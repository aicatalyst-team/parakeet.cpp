#include "relpos_attention.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace pk {

// Clone a tensor from the loader context into the compute context, preserving
// ne[] layout and f32 data (same helper pattern as subsampling.cpp).
static ggml_tensor* clone_weight(ggml_context* ctx, const ModelLoader& ml,
                                 const std::string& name) {
    ggml_tensor* src = ml.tensor(name);
    assert(src && "missing tensor");
    assert(src->type == GGML_TYPE_F32 && "expected f32 weight");
    const int nd = ggml_n_dims(src);
    int64_t ne[4] = {1, 1, 1, 1};
    for (int i = 0; i < nd; ++i) ne[i] = src->ne[i];
    ggml_tensor* dst = ggml_new_tensor(ctx, GGML_TYPE_F32, nd, ne);
    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    return dst;
}

RelPosAttention::RelPosAttention(const ModelLoader& ml, int layer_idx)
    : ml_(ml), layer_idx_(layer_idx) {
    d_model_ = (int)ml.config().d_model;
    n_heads_ = (int)ml.config().n_heads;
    assert(n_heads_ > 0 && d_model_ % n_heads_ == 0);
    d_head_ = d_model_ / n_heads_;
}

void RelPosAttention::forward(const std::vector<float>& x, int T,
                              const std::vector<float>& pos_emb, int pos_len,
                              int valid_len,
                              std::vector<float>& out) const {
    const int D  = d_model_;
    const int H  = n_heads_;
    const int dk = d_head_;
    const float scale = 1.0f / std::sqrt((float)dk);

    assert((int)x.size() == T * D);
    assert((int)pos_emb.size() == pos_len * D);
    assert(pos_len == 2 * T - 1);

    const std::string pre = "encoder.layers." + std::to_string(layer_idx_) + ".self_attn.";
    const ModelLoader& ml = ml_;

    // Generous compute budget: a handful of [T,T,H] and [2T-1,T,H] intermediates.
    const size_t mem_bytes =
        (size_t)128 * 1024 * 1024 +
        (size_t)H * (size_t)T * (size_t)pos_len * 64 * sizeof(float);

    bool ok = pk::run_graph(mem_bytes, /*n_threads*/4,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // ---- inputs ----
            // x: ne [D, T] (d_model fastest), fed time-major x[t*D + c].
            ggml_tensor* xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
            std::memcpy(xt->data, x.data(), (size_t)T * D * sizeof(float));
            // pos_emb: ne [D, P] (d_model fastest), pe[p*D + c].
            ggml_tensor* pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, pos_len);
            std::memcpy(pe->data, pos_emb.data(), (size_t)pos_len * D * sizeof(float));

            // ---- linear projections (nn.Linear: ggml W ne=[in,out]) ----
            // The bias is added only when requested AND present: NeMo configures
            // the attention linears with bias=False in some checkpoints
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
            // NeMo view(B,T,H,dk): head h occupies channels [h*dk : (h+1)*dk], so
            // reshaping ne0=D into (dk, H) places head h at the h-th dk block.
            auto to_heads = [&](ggml_tensor* t, int n) {
                t = ggml_reshape_3d(ctx, t, dk, H, n);             // [dk, H, n]
                t = ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3)); // [dk, n, H]
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
            // rel_shift (NeMo): pad front of pos dim by 1, reinterpret, drop first
            // row, slice to first T. See unit-validated numpy trace.
            bd = ggml_pad_ext(ctx, bd, /*lp0*/1, /*rp0*/0, 0,0, 0,0, 0,0); // [P+1=2T, T, H]
            bd = ggml_reshape_3d(ctx, bd, T, 2 * T, H);                    // [T, 2T, H]
            // drop first row along the 2T dim: view [T, 2T-1, H] offset by one row.
            bd = ggml_view_3d(ctx, bd, T, 2 * T - 1, H,
                              bd->nb[1], bd->nb[2], bd->nb[1]);            // [T, 2T-1, H]
            bd = ggml_cont(ctx, bd);
            bd = ggml_reshape_3d(ctx, bd, 2 * T - 1, T, H);               // [2T-1, T, H]
            // slice first T columns of the pos dim -> [T(key), T(query), H].
            bd = ggml_view_3d(ctx, bd, T, T, H, bd->nb[1], bd->nb[2], 0);
            bd = ggml_cont(ctx, bd);

            // ---- scores = ac + bd ; softmax(scores*scale + mask) ----
            ggml_tensor* scores = ggml_add(ctx, ac, bd); // [T_k, T_q, H]

            // Additive mask [T_k, T_q]: 0 where key j is valid, -inf where padded.
            // (broadcasts over heads). Matches NeMo masking of key column j>=valid.
            ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, T);
            {
                float* md = (float*)mask->data;
                const float ninf = -INFINITY;
                for (int qi = 0; qi < T; ++qi)
                    for (int kj = 0; kj < T; ++kj)
                        md[(size_t)qi * T + kj] = (kj < valid_len) ? 0.0f : ninf;
            }
            // softmax(scores*scale + mask), normalized over keys (ne0).
            ggml_tensor* attn = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f); // [T_k, T_q, H]

            // ---- context = attn @ v : per head sum_k attn[k,q]*v[k] -> [dk, T_q, H] ----
            // ggml_mul_mat(A[dk?..]) needs contraction on ne0. attn ne0=T_k, vh ne0=dk.
            // Use mul_mat(vh^T?, ...): we want out[d, q] = sum_k v[d,k]*attn[k,q].
            // ggml_mul_mat(a,b): out[i,j] = sum_r a[r,i]*b[r,j], contraction on ne0(r).
            // Set a = attn with ne0=T_k (a[k, q]) and b = v_kd with ne0=T_k (b[k, d]):
            // out[q, d]. We instead want [dk, T_q]. Take a = v as [T_k, dk, H]
            // (v transposed) and b = attn [T_k, T_q, H] -> out[dk, T_q, H].
            ggml_tensor* vtk = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3)); // [T_k, dk, H]
            ggml_tensor* ctxh = ggml_mul_mat(ctx, vtk, attn); // [dk, T_q, H]

            // ---- concat heads: [dk, T, H] -> [dk, H, T] -> [D, T] ----
            ggml_tensor* merged = ggml_cont(ctx, ggml_permute(ctx, ctxh, 0, 2, 1, 3)); // [dk, H, T]
            merged = ggml_reshape_2d(ctx, merged, D, T); // [D, T]

            // Zero the context for PADDED query rows. NeMo masks padded query rows
            // fully (att_mask row j>=valid is all True) and masked_fill the softmax
            // output to 0, so attn@v == 0 for those rows -> output reduces to
            // linear_out.bias. The key-column mask above doesn't cover this (it only
            // zeros padded keys), so apply a query-row mask [1, T] over the context.
            if (valid_len < T) {
                ggml_tensor* qmask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, T);
                float* qd = (float*)qmask->data;
                for (int qi = 0; qi < T; ++qi) qd[qi] = (qi < valid_len) ? 1.0f : 0.0f;
                merged = ggml_mul(ctx, merged, qmask); // broadcast over D
            }

            // ---- output projection ----
            ggml_tensor* y = linear("linear_out.weight", "linear_out.bias", merged); // [D, T]
            return y;
        },
        out);

    assert(ok && "relpos attention graph failed");
    (void)ok;
}

} // namespace pk
