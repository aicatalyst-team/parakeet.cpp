#include "conformer.hpp"
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
// pk::clone_weight / pk::clone_weight_opt (backend.cpp; zero-copy via the
// loader's CPU backend buffer). Quantization-allowlist linears may be f16/q8_0
// (ggml_mul_mat dequantizes src0 on the fly). Optional weights (conv biases,
// absent in bias=False checkpoints) use clone_weight_opt. std::string-name
// overloads keep the call sites (pre + suffix) unchanged.
static ggml_tensor* clone_weight(ggml_context* ctx, const ModelLoader& ml,
                                 const std::string& name) {
    return pk::clone_weight(ctx, ml, name.c_str());
}
static ggml_tensor* clone_weight_opt(ggml_context* ctx, const ModelLoader& ml,
                                     const std::string& name) {
    return pk::clone_weight_opt(ctx, ml, name.c_str());
}

// Build the ConformerConvolution sub-graph (everything AFTER norm_conv) on the
// conv input `c` (= norm_conv(residual)), ne [D, T] (channels fastest). Returns
// the conv output tensor, ne [D, T] (row-major [T, D]). See header for the NeMo
// `ConformerConvolution.forward` mapping. `pool` keeps host-side mask/scale/
// shift buffers alive until the enclosing Backend::compute finishes.
static ggml_tensor* build_conv_module(ggml_context* ctx, const ModelLoader& ml,
                                      const std::string& pre, ggml_tensor* c,
                                      int D, int T, int K, int valid_len,
                                      const std::string& conv_norm_type,
                                      bool conv_causal, GraphInputPool& pool) {
    const float ln_eps = 1e-5f;
    const float bn_eps = 1e-5f;
    const int pad = (K - 1) / 2;  // symmetric padding (offline model)

    // -- pointwise_conv1 (Conv1d d->2d, k=1): 1x1 conv == linear over channels.
    ggml_tensor* pw1w = clone_weight(ctx, ml, pre + "conv.pointwise_conv1.weight");
    pw1w = ggml_reshape_2d(ctx, pw1w, D, 2 * D); // [in=d, out=2d]
    ggml_tensor* pw1b = clone_weight_opt(ctx, ml, pre + "conv.pointwise_conv1.bias");
    ggml_tensor* y = ggml_mul_mat(ctx, pw1w, c); // [2d, T]
    if (pw1b) y = ggml_add(ctx, y, pw1b);

    // -- GLU over channel dim (NeMo F.glu(x, dim=1)).
    ggml_tensor* a = ggml_view_2d(ctx, y, D, T, y->nb[1], 0);
    ggml_tensor* b = ggml_view_2d(ctx, y, D, T, y->nb[1], (size_t)D * y->nb[0]);
    ggml_tensor* glu = ggml_mul(ctx, ggml_cont(ctx, a),
                                ggml_sigmoid(ctx, ggml_cont(ctx, b))); // [d, T]

    // -- pad_mask: zero padded time positions before depthwise conv.
    if (valid_len < T) {
        std::vector<float>& md = pool.alloc_f32(T);
        for (int t = 0; t < T; ++t) md[t] = (t < valid_len) ? 1.0f : 0.0f;
        int64_t tm_ne[2] = {1, T};
        ggml_tensor* tmask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, tm_ne,
                                 md.data(), md.size() * sizeof(float));
        glu = ggml_mul(ctx, glu, tmask);
    }

    // -- depthwise_conv (Conv1d d->d, k=K, groups=d). F32 im2col throughout.
    ggml_tensor* glu_tc = ggml_cont(ctx, ggml_transpose(ctx, glu)); // [T, C]
    ggml_tensor* dww = clone_weight(ctx, ml, pre + "conv.depthwise_conv.weight"); // [K,1,C]
    ggml_tensor* dw;
    {
        ggml_tensor* nb = ggml_reshape_4d(ctx, glu_tc,
                              glu_tc->ne[0], 1, glu_tc->ne[1], 1); // [T,1,C,1]
        ggml_tensor* ic;
        if (conv_causal) {
            ggml_tensor* nbp = ggml_pad_ext(ctx, nb, /*lp0*/K - 1, /*rp0*/0,
                                            0, 0, 0, 0, 0, 0); // [T+K-1,1,C,1]
            ic = ggml_im2col(ctx, dww, nbp, /*s0*/1, /*s1*/0,
                             /*p0*/0, /*p1*/0, /*d0*/1, /*d1*/0,
                             /*is_2D*/false, GGML_TYPE_F32);
        } else {
            ic = ggml_im2col(ctx, dww, nb, /*s0*/1, /*s1*/0,
                             /*p0*/pad, /*p1*/0, /*d0*/1, /*d1*/0,
                             /*is_2D*/false, GGML_TYPE_F32);
        }
        ggml_tensor* r2 = ggml_mul_mat(ctx, ic, dww);
        dw = ggml_reshape_3d(ctx, r2, r2->ne[0], r2->ne[2], 1); // [OW=T, C, 1]
    }
    dw = ggml_reshape_2d(ctx, dw, T, D);                  // [T, C]
    ggml_tensor* dwb = clone_weight_opt(ctx, ml, pre + "conv.depthwise_conv.bias"); // [C]
    ggml_tensor* dwt = ggml_cont(ctx, ggml_transpose(ctx, dw)); // [C, T]
    if (dwb) dwt = ggml_add(ctx, dwt, dwb);               // broadcast [C] over T

    // -- norm (between depthwise conv and SiLU).
    ggml_tensor* normed;
    if (conv_norm_type == "layer_norm") {
        ggml_tensor* g = clone_weight(ctx, ml, pre + "conv.batch_norm.weight"); // [C]
        ggml_tensor* bb = clone_weight(ctx, ml, pre + "conv.batch_norm.bias");  // [C]
        normed = ggml_norm(ctx, dwt, ln_eps);             // normalize over ne0=C
        normed = ggml_mul(ctx, normed, g);                // *gamma  (broadcast [C] over T)
        normed = ggml_add(ctx, normed, bb);               // +beta
    } else {
        // batch_norm (inference): fold into per-channel scale/shift constants:
        //   scale = g / sqrt(var+eps); shift = b - mean*scale.  Computed host-side.
        std::vector<float>& sc = pool.alloc_f32(D);
        std::vector<float>& sh = pool.alloc_f32(D);
        std::vector<float> g, bb, m, var;
        pk::weight_to_host_f32(ml, (pre + "conv.batch_norm.weight").c_str(), g);
        pk::weight_to_host_f32(ml, (pre + "conv.batch_norm.bias").c_str(), bb);
        pk::weight_to_host_f32(ml, (pre + "conv.batch_norm.running_mean").c_str(), m);
        pk::weight_to_host_f32(ml, (pre + "conv.batch_norm.running_var").c_str(), var);
        for (int cc = 0; cc < D; ++cc) {
            sc[cc] = g[cc] / std::sqrt(var[cc] + bn_eps);
            sh[cc] = bb[cc] - m[cc] * sc[cc];
        }
        int64_t d_ne[1] = {D};
        ggml_tensor* scale = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, d_ne,
                                 sc.data(), sc.size() * sizeof(float));
        ggml_tensor* shift = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, d_ne,
                                 sh.data(), sh.size() * sizeof(float));
        normed = ggml_add(ctx, ggml_mul(ctx, dwt, scale), shift); // [C, T]
    }

    // -- SiLU (Swish), then pointwise_conv2 (Conv1d d->d, k=1).
    normed = ggml_silu(ctx, normed);
    ggml_tensor* pw2w = clone_weight(ctx, ml, pre + "conv.pointwise_conv2.weight");
    pw2w = ggml_reshape_2d(ctx, pw2w, D, D); // [in=d, out=d]
    ggml_tensor* pw2b = clone_weight_opt(ctx, ml, pre + "conv.pointwise_conv2.bias");
    ggml_tensor* cout = ggml_mul_mat(ctx, pw2w, normed); // [d, T]
    if (pw2b) cout = ggml_add(ctx, cout, pw2b);
    return cout; // [D, T] -> row-major [T, D]; this is layers[i].conv output
}

ConformerLayer::ConformerLayer(const ModelLoader& ml, int layer_idx)
    : ml_(ml), layer_idx_(layer_idx) {
    d_model_     = (int)ml.config().d_model;
    n_heads_     = (int)ml.config().n_heads;
    ff_dim_      = (int)ml.config().ff_dim;
    conv_kernel_ = (int)ml.config().conv_kernel;
    conv_norm_type_ = ml.config().conv_norm_type;
    conv_causal_    = ml.config().conv_causal;
    assert((conv_norm_type_ == "batch_norm" || conv_norm_type_ == "layer_norm") &&
           "ConformerLayer supports conv_norm_type in {batch_norm, layer_norm}");
    assert(n_heads_ > 0 && d_model_ % n_heads_ == 0);
    assert((conv_kernel_ - 1) % 2 == 0 && "depthwise kernel must be odd");
}

ggml_tensor* ConformerLayer::build_graph(ggml_context* ctx, ggml_tensor* xt,
                                         int T, ggml_tensor* pe, int pos_len,
                                         int valid_len,
                                         GraphInputPool& pool) const {
    const int D  = d_model_;
    const int K  = conv_kernel_;
    const float ln_eps = 1e-5f;            // LayerNorm eps (NeMo nn.LayerNorm default)
    assert(pos_len == 2 * T - 1);

    const std::string pre = "encoder.layers." + std::to_string(layer_idx_) + ".";
    const ModelLoader& ml = ml_;

    // LayerNorm over the channel dim (ne0 = D), affine. Input ne [D, T].
    auto layer_norm = [&](ggml_tensor* in, const std::string& nm) {
        ggml_tensor* g = clone_weight(ctx, ml, pre + nm + ".weight"); // [D]
        ggml_tensor* b = clone_weight(ctx, ml, pre + nm + ".bias");   // [D]
        ggml_tensor* y = ggml_norm(ctx, in, ln_eps);                  // normalize over ne0
        y = ggml_mul(ctx, y, g);                                      // broadcast [D] over T
        y = ggml_add(ctx, y, b);
        return y;
    };
    // nn.Linear: ggml weight ne = [in, out]. in ne [in, T] -> [out, T].
    auto linear = [&](ggml_tensor* in, const std::string& nm, bool bias) {
        ggml_tensor* W = clone_weight(ctx, ml, pre + nm + ".weight");
        ggml_tensor* y = ggml_mul_mat(ctx, W, in);
        if (bias) {
            ggml_tensor* B = clone_weight_opt(ctx, ml, pre + nm + ".bias");
            if (B) y = ggml_add(ctx, y, B);
        }
        return y;
    };
    // ConformerFeedForward: linear1(d->ff) -> SiLU -> linear2(ff->d). in [D, T].
    auto feed_forward = [&](ggml_tensor* in, const std::string& ff) {
        ggml_tensor* h = linear(in, ff + ".linear1", /*bias*/true); // [FF, T]
        h = ggml_silu(ctx, h);                                      // Swish == SiLU
        h = linear(h, ff + ".linear2", /*bias*/true);               // [D, T]
        return h;
    };

    // === Stage A: r = x + 0.5 * FFN1(norm_ff1(x)). ===
    ggml_tensor* h1 = layer_norm(xt, "norm_feed_forward1");
    h1 = feed_forward(h1, "feed_forward1");
    h1 = ggml_scale(ctx, h1, 0.5f);          // fc_factor
    ggml_tensor* r = ggml_add(ctx, xt, h1);  // [D, T]

    // === Stage B: r = r + self_attn(norm_self_att(r)). ===
    ggml_tensor* attn_in = layer_norm(r, "norm_self_att");
    RelPosAttention attn(ml_, layer_idx_);
    ggml_tensor* attn_out = attn.build_graph(ctx, attn_in, T, pe, pos_len,
                                             valid_len, pool); // [D, T]
    r = ggml_add(ctx, r, attn_out);

    // === Stage C: r = r + conv(norm_conv(r)). ===
    ggml_tensor* c = layer_norm(r, "norm_conv"); // [D, T]
    ggml_tensor* conv_out = build_conv_module(ctx, ml, pre, c, D, T, K, valid_len,
                                              conv_norm_type_, conv_causal_, pool);
    r = ggml_add(ctx, r, conv_out);

    // === Stage D: r = r + 0.5 * FFN2(norm_ff2(r)); out = norm_out(r). ===
    ggml_tensor* h2 = layer_norm(r, "norm_feed_forward2");
    h2 = feed_forward(h2, "feed_forward2");
    h2 = ggml_scale(ctx, h2, 0.5f);
    r = ggml_add(ctx, r, h2);
    r = layer_norm(r, "norm_out");
    return r; // [D, T] -> row-major [T, D]
}

void ConformerLayer::forward(const std::vector<float>& x, int T,
                             const std::vector<float>& pos_emb, int pos_len,
                             int valid_len,
                             std::vector<float>& out) const {
    std::vector<float> conv_out_unused;
    forward_with_conv(x, T, pos_emb, pos_len, valid_len, out, conv_out_unused);
}

void ConformerLayer::forward_with_conv(const std::vector<float>& x, int T,
                                       const std::vector<float>& pos_emb, int pos_len,
                                       int valid_len,
                                       std::vector<float>& out,
                                       std::vector<float>& conv_out) const {
    const int D = d_model_;
    assert((int)x.size() == T * D);
    assert((int)pos_emb.size() == pos_len * D);
    assert(pos_len == 2 * T - 1);

    // Unit-test entry point: build the full layer as ONE sub-graph and compute
    // it on the persistent Backend (the fused encoder calls build_graph
    // directly). The conv-module output (for localization) is recomputed via a
    // tiny separate sub-graph below; the test only needs it for a print, not the
    // hot path, so the extra graph is acceptable.
    {
        GraphInputPool pool;
        bool ok = pk::run_graph(/*mem_bytes*/0, /*n_threads*/4,
            [&](ggml_context* ctx) -> ggml_tensor* {
                int64_t xt_ne[2] = {D, T};
                ggml_tensor* xt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
                                      xt_ne, x.data(), (size_t)T * D * sizeof(float));
                int64_t pe_ne[2] = {D, pos_len};
                ggml_tensor* pe = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
                                      pe_ne, pos_emb.data(),
                                      (size_t)pos_len * D * sizeof(float));
                return build_graph(ctx, xt, T, pe, pos_len, valid_len, pool);
            }, out);
        assert(ok && "conformer layer graph failed"); (void)ok;
    }
    // Conv-module output localization (norm_conv -> conv): recompute r up to the
    // conv input and emit the conv-module output. r here = x + FFN1 + attn (same
    // as the fused path's residual into norm_conv).
    {
        GraphInputPool pool;
        bool ok = pk::run_graph(/*mem_bytes*/0, /*n_threads*/4,
            [&](ggml_context* ctx) -> ggml_tensor* {
                int64_t xt_ne[2] = {D, T};
                ggml_tensor* xt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
                                      xt_ne, x.data(), (size_t)T * D * sizeof(float));
                int64_t pe_ne[2] = {D, pos_len};
                ggml_tensor* pe = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
                                      pe_ne, pos_emb.data(),
                                      (size_t)pos_len * D * sizeof(float));
                const std::string pre = "encoder.layers." +
                                        std::to_string(layer_idx_) + ".";
                auto layer_norm = [&](ggml_tensor* in, const std::string& nm) {
                    ggml_tensor* g = clone_weight(ctx, ml_, pre + nm + ".weight");
                    ggml_tensor* b = clone_weight(ctx, ml_, pre + nm + ".bias");
                    ggml_tensor* y = ggml_norm(ctx, in, 1e-5f);
                    y = ggml_mul(ctx, y, g);
                    y = ggml_add(ctx, y, b);
                    return y;
                };
                auto linear = [&](ggml_tensor* in, const std::string& nm, bool bias) {
                    ggml_tensor* W = clone_weight(ctx, ml_, pre + nm + ".weight");
                    ggml_tensor* y = ggml_mul_mat(ctx, W, in);
                    if (bias) { ggml_tensor* B = clone_weight_opt(ctx, ml_, pre + nm + ".bias");
                                if (B) y = ggml_add(ctx, y, B); }
                    return y;
                };
                auto feed_forward = [&](ggml_tensor* in, const std::string& ff) {
                    ggml_tensor* h = linear(in, ff + ".linear1", true);
                    h = ggml_silu(ctx, h);
                    return linear(h, ff + ".linear2", true);
                };
                ggml_tensor* h1 = feed_forward(layer_norm(xt, "norm_feed_forward1"),
                                               "feed_forward1");
                h1 = ggml_scale(ctx, h1, 0.5f);
                ggml_tensor* r = ggml_add(ctx, xt, h1);
                ggml_tensor* attn_in = layer_norm(r, "norm_self_att");
                RelPosAttention attn(ml_, layer_idx_);
                ggml_tensor* attn_out = attn.build_graph(ctx, attn_in, T, pe,
                                            pos_len, valid_len, pool);
                r = ggml_add(ctx, r, attn_out);
                ggml_tensor* c = layer_norm(r, "norm_conv");
                return build_conv_module(ctx, ml_, pre, c, D, T, conv_kernel_,
                                         valid_len, conv_norm_type_, conv_causal_,
                                         pool);
            }, conv_out);
        assert(ok && "conformer conv localization graph failed"); (void)ok;
    }
}

void ConformerLayer::conv_module_forward(const std::vector<float>& conv_in, int T,
                                         int valid_len,
                                         std::vector<float>& out) const {
    const int D = d_model_;
    const int K = conv_kernel_;
    assert((int)conv_in.size() == T * D);

    const std::string pre = "encoder.layers." + std::to_string(layer_idx_) + ".";
    const ModelLoader& ml = ml_;

    // conv_in is norm_conv(residual), row-major [T, D] -> ggml [D, T]; feed it
    // straight into the conv sub-module (NeMo's ConformerConvolution.forward,
    // which starts AFTER norm_conv).
    GraphInputPool pool;
    bool ok = pk::run_graph(/*mem_bytes*/0, /*n_threads*/4,
        [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t c_ne[2] = {D, T};
            ggml_tensor* c = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, c_ne,
                                 conv_in.data(), (size_t)T * D * sizeof(float));
            return build_conv_module(ctx, ml, pre, c, D, T, K, valid_len,
                                     conv_norm_type_, conv_causal_, pool);
        }, out);
    assert(ok && "conv_module_forward graph failed"); (void)ok;
}

} // namespace pk
