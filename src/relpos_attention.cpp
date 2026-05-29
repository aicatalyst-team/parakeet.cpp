#include "relpos_attention.hpp"
#include "ggml_graph.hpp"
#include "backend.hpp"
#include "ggml.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace pk {

// Weights from the loader are referenced DIRECTLY as graph leaves via the shared
// pk::clone_weight (backend.cpp; zero-copy via the loader's CPU backend buffer).
// Allowlisted attention linears (linear_q/k/v/out/pos.weight) may be f16/q8_0
// and are fed into ggml_mul_mat, which dequantizes src0 on the fly. pos_bias_u/v
// and every other weight stay F32. A std::string-name overload keeps the call
// sites (pre + suffix) unchanged.
static ggml_tensor* clone_weight(ggml_context* ctx, const ModelLoader& ml,
                                 const std::string& name) {
    return pk::clone_weight(ctx, ml, name.c_str());
}

RelPosAttention::RelPosAttention(const ModelLoader& ml, int layer_idx)
    : ml_(ml), layer_idx_(layer_idx) {
    d_model_ = (int)ml.config().d_model;
    n_heads_ = (int)ml.config().n_heads;
    assert(n_heads_ > 0 && d_model_ % n_heads_ == 0);
    d_head_ = d_model_ / n_heads_;
    // Chunked-limited attention (NeMo att_context_style=="chunked_limited",
    // e.g. parakeet_realtime_eou_120m-v1 with att_context_size=[70,1]). The
    // offline forward applies the SAME additive -inf mask NeMo builds in
    // ConformerEncoder._create_masks. Offline models use "regular" (full context).
    chunked_limited_ = (ml.config().att_context_style == "chunked_limited" &&
                        ml.config().att_context_right >= 0);
    att_left_  = ml.config().att_context_left;
    att_right_ = ml.config().att_context_right;
}

ggml_tensor* RelPosAttention::build_graph(ggml_context* ctx, ggml_tensor* xt,
                                          int T, ggml_tensor* pe, int pos_len,
                                          int valid_len,
                                          GraphInputPool& pool) const {
    const int D  = d_model_;
    const int H  = n_heads_;
    const int dk = d_head_;
    const float scale = 1.0f / std::sqrt((float)dk);
    assert(pos_len == 2 * T - 1);

    const std::string pre = "encoder.layers." + std::to_string(layer_idx_) + ".self_attn.";
    const ModelLoader& ml = ml_;

    // ---- linear projections (nn.Linear: ggml W ne=[in,out]) ----
    // The bias is added only when requested AND present: NeMo configures the
    // attention linears with bias=False in some checkpoints
    // (parakeet-tdt-0.6b-v2/-v3) and bias=True in others (110m).
    auto linear = [&](const char* w, const char* b, ggml_tensor* in) {
        ggml_tensor* W = clone_weight(ctx, ml, pre + w);
        ggml_tensor* y = ggml_mul_mat(ctx, W, in);  // [out, *]
        if (b && ml.tensor(pre + b)) {
            ggml_tensor* B = clone_weight(ctx, ml, pre + b);
            y = ggml_add(ctx, y, B);                // broadcast [out] over cols
        }
        return y;
    };
    ggml_tensor* q = linear("linear_q.weight", "linear_q.bias", xt); // [D, T]
    ggml_tensor* k = linear("linear_k.weight", "linear_k.bias", xt); // [D, T]
    ggml_tensor* v = linear("linear_v.weight", "linear_v.bias", xt); // [D, T]
    ggml_tensor* p = linear("linear_pos.weight", nullptr, pe);       // [D, P]

    // ---- split into heads: [D, *] -> [dk, H, *] -> [dk, *, H] ----
    auto to_heads = [&](ggml_tensor* t, int n) {
        t = ggml_reshape_3d(ctx, t, dk, H, n);                 // [dk, H, n]
        t = ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));  // [dk, n, H]
        return t;
    };
    ggml_tensor* qh = to_heads(q, T);        // [dk, T, H]
    ggml_tensor* kh = to_heads(k, T);        // [dk, T, H]
    ggml_tensor* vh = to_heads(v, T);        // [dk, T, H]
    ggml_tensor* ph = to_heads(p, pos_len);  // [dk, P, H]

    // ---- pos_bias_u/v: ne [dk, H] -> [dk, 1, H] to broadcast over T ----
    ggml_tensor* bu = clone_weight(ctx, ml, pre + "pos_bias_u"); // [dk, H]
    ggml_tensor* bv = clone_weight(ctx, ml, pre + "pos_bias_v"); // [dk, H]
    bu = ggml_reshape_3d(ctx, bu, dk, 1, H);
    bv = ggml_reshape_3d(ctx, bv, dk, 1, H);
    ggml_tensor* qu = ggml_add(ctx, qh, bu); // [dk, T, H]
    ggml_tensor* qv = ggml_add(ctx, qh, bv); // [dk, T, H]

    // ---- ac = q_u @ k^T : ggml_mul_mat([dk,T,H],[dk,T,H]) -> [T_k, T_q, H] ----
    ggml_tensor* ac = ggml_mul_mat(ctx, kh, qu); // [T(key), T(query), H]

    // ---- bd = q_v @ p^T -> [P(pos), T(query), H], then rel_shift -> [T,T,H] ----
    ggml_tensor* bd = ggml_mul_mat(ctx, ph, qv); // [P, T, H]
    bd = ggml_pad_ext(ctx, bd, /*lp0*/1, /*rp0*/0, 0,0, 0,0, 0,0); // [P+1=2T, T, H]
    bd = ggml_reshape_3d(ctx, bd, T, 2 * T, H);                    // [T, 2T, H]
    bd = ggml_view_3d(ctx, bd, T, 2 * T - 1, H,
                      bd->nb[1], bd->nb[2], bd->nb[1]);            // [T, 2T-1, H]
    bd = ggml_cont(ctx, bd);
    bd = ggml_reshape_3d(ctx, bd, 2 * T - 1, T, H);               // [2T-1, T, H]
    bd = ggml_view_3d(ctx, bd, T, T, H, bd->nb[1], bd->nb[2], 0);
    bd = ggml_cont(ctx, bd);

    // ---- scores = ac + bd ; softmax(scores*scale + mask) ----
    ggml_tensor* scores = ggml_add(ctx, ac, bd); // [T_k, T_q, H]

    // Additive mask [T_k, T_q]: 0 where query qi may attend to key kj, -inf
    // otherwise. (1) pad mask: key kj valid iff kj < valid_len. (2) chunked-
    // limited window for streaming models. See header / NeMo _create_masks.
    const int chunk_size  = chunked_limited_ ? (att_right_ + 1) : 0;
    const int left_chunks = (chunked_limited_ && chunk_size > 0)
                            ? (att_left_ / chunk_size) : 0;
    std::vector<float>& mask_host = pool.alloc_f32((size_t)T * T);
    {
        float* md = mask_host.data();
        const float ninf = -INFINITY;
        for (int qi = 0; qi < T; ++qi) {
            const int cq = chunked_limited_ ? (qi / chunk_size) : 0;
            for (int kj = 0; kj < T; ++kj) {
                bool ok = (kj < valid_len);
                if (ok && chunked_limited_) {
                    const int ck = kj / chunk_size;
                    const int diff = cq - ck;
                    ok = (diff >= 0 && diff <= left_chunks);
                }
                md[(size_t)qi * T + kj] = ok ? 0.0f : ninf;
            }
        }
    }
    int64_t mask_ne[2] = {T, T};
    ggml_tensor* mask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, mask_ne,
                            mask_host.data(), mask_host.size() * sizeof(float));
    ggml_tensor* attn = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f); // [T_k, T_q, H]

    // ---- context = attn @ v -> [dk, T_q, H] ----
    ggml_tensor* vtk = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3)); // [T_k, dk, H]
    ggml_tensor* ctxh = ggml_mul_mat(ctx, vtk, attn); // [dk, T_q, H]

    // ---- concat heads: [dk, T, H] -> [dk, H, T] -> [D, T] ----
    ggml_tensor* merged = ggml_cont(ctx, ggml_permute(ctx, ctxh, 0, 2, 1, 3)); // [dk, H, T]
    merged = ggml_reshape_2d(ctx, merged, D, T); // [D, T]

    // Zero the context for PADDED query rows (NeMo masks padded query rows fully
    // -> output reduces to linear_out.bias). Apply a query-row mask [1, T].
    if (valid_len < T) {
        std::vector<float>& qmask_host = pool.alloc_f32(T);
        for (int qi = 0; qi < T; ++qi)
            qmask_host[qi] = (qi < valid_len) ? 1.0f : 0.0f;
        int64_t qm_ne[2] = {1, T};
        ggml_tensor* qmask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, qm_ne,
                                 qmask_host.data(), qmask_host.size() * sizeof(float));
        merged = ggml_mul(ctx, merged, qmask); // broadcast over D
    }

    // ---- output projection ----
    return linear("linear_out.weight", "linear_out.bias", merged); // [D, T]
}

void RelPosAttention::forward(const std::vector<float>& x, int T,
                              const std::vector<float>& pos_emb, int pos_len,
                              int valid_len,
                              std::vector<float>& out) const {
    const int D = d_model_;
    assert((int)x.size() == T * D);
    assert((int)pos_emb.size() == pos_len * D);
    assert(pos_len == 2 * T - 1);

    // Thin wrapper over the graph-builder: build JUST the attention sub-graph
    // with x/pos_emb fed as inputs and compute it on the persistent Backend.
    // Used by the unit test (the fused conformer layer uses build_graph).
    GraphInputPool pool;
    bool ok = pk::run_graph(/*mem_bytes*/0, /*n_threads*/4,
        [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t xt_ne[2] = {D, T};
            ggml_tensor* xt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, xt_ne,
                                  x.data(), (size_t)T * D * sizeof(float));
            int64_t pe_ne[2] = {D, pos_len};
            ggml_tensor* pe = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, pe_ne,
                                  pos_emb.data(), (size_t)pos_len * D * sizeof(float));
            return build_graph(ctx, xt, T, pe, pos_len, valid_len, pool);
        }, out);
    assert(ok && "relpos attention graph failed");
    (void)ok;
}

} // namespace pk
