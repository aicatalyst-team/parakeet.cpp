#include "conformer.hpp"
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
// ne[] layout and f32 data (same helper pattern as relpos_attention.cpp).
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

// Like clone_weight but returns nullptr when the tensor is absent (optional
// weight). The conformer conv submodule's Conv1d biases are present in some
// checkpoints (e.g. parakeet-tdt_ctc-110m) and absent in others where NeMo
// configures the convolutions with bias=False (e.g. parakeet-tdt-0.6b-v2/-v3).
// Honour whatever the checkpoint actually carries instead of assuming a bias.
static ggml_tensor* clone_weight_opt(ggml_context* ctx, const ModelLoader& ml,
                                     const std::string& name) {
    if (!ml.tensor(name)) return nullptr;
    return clone_weight(ctx, ml, name);
}

ConformerLayer::ConformerLayer(const ModelLoader& ml, int layer_idx)
    : ml_(ml), layer_idx_(layer_idx) {
    d_model_     = (int)ml.config().d_model;
    n_heads_     = (int)ml.config().n_heads;
    ff_dim_      = (int)ml.config().ff_dim;
    conv_kernel_ = (int)ml.config().conv_kernel;
    // This implementation only handles the batch_norm conv module (the affine
    // running-stat folding below). The layer_norm path transposes around the
    // norm and is a different model variant — out of scope here.
    assert(ml.config().conv_norm_type == "batch_norm" &&
           "ConformerLayer only supports conv_norm_type=batch_norm");
    assert(n_heads_ > 0 && d_model_ % n_heads_ == 0);
    assert((conv_kernel_ - 1) % 2 == 0 && "depthwise kernel must be odd");
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
    const int D  = d_model_;
    const int FF = ff_dim_;
    const int K  = conv_kernel_;
    const int pad = (K - 1) / 2;          // symmetric padding (offline model)
    const float ln_eps = 1e-5f;
    const float bn_eps = 1e-5f;

    assert((int)x.size() == T * D);
    assert((int)pos_emb.size() == pos_len * D);
    assert(pos_len == 2 * T - 1);

    const std::string pre = "encoder.layers." + std::to_string(layer_idx_) + ".";
    const ModelLoader& ml = ml_;

    // Memory budget: FFN intermediates are the largest ([FF, T]).
    const size_t mem_bytes =
        (size_t)128 * 1024 * 1024 +
        (size_t)FF * (size_t)T * 64 * sizeof(float);

    // ---- shared graph helpers (capture-by-value of ctx/ml/pre inside lambdas) ----
    // LayerNorm over the channel dim (ne0 = D), affine. Input ne [D, T].
    auto layer_norm = [&](ggml_context* ctx, ggml_tensor* in, const std::string& nm) {
        ggml_tensor* g = clone_weight(ctx, ml, pre + nm + ".weight"); // [D]
        ggml_tensor* b = clone_weight(ctx, ml, pre + nm + ".bias");   // [D]
        ggml_tensor* y = ggml_norm(ctx, in, ln_eps);                  // normalize over ne0
        y = ggml_mul(ctx, y, g);                                      // broadcast [D] over T
        y = ggml_add(ctx, y, b);
        return y;
    };
    // nn.Linear: ggml weight ne = [in, out]. in ne [in, T] -> [out, T].
    // The bias is added only when both requested AND present in the checkpoint:
    // NeMo configures the FastConformer FFN/attention linears with bias=False in
    // some models (parakeet-tdt-0.6b-v2/-v3) and bias=True in others (110m).
    auto linear = [&](ggml_context* ctx, ggml_tensor* in,
                      const std::string& nm, bool bias) {
        ggml_tensor* W = clone_weight(ctx, ml, pre + nm + ".weight");
        ggml_tensor* y = ggml_mul_mat(ctx, W, in);
        if (bias) {
            ggml_tensor* B = clone_weight_opt(ctx, ml, pre + nm + ".bias");
            if (B) y = ggml_add(ctx, y, B);
        }
        return y;
    };
    // ConformerFeedForward: linear1(d->ff) -> SiLU -> linear2(ff->d). in [D, T].
    auto feed_forward = [&](ggml_context* ctx, ggml_tensor* in, const std::string& ff) {
        ggml_tensor* h = linear(ctx, in, ff + ".linear1", /*bias*/true); // [FF, T]
        h = ggml_silu(ctx, h);                                           // Swish == SiLU
        h = linear(ctx, h, ff + ".linear2", /*bias*/true);              // [D, T]
        return h;
    };

    // === Stage A: r = x + 0.5 * FFN1(norm_ff1(x)); emit norm_self_att(r). ===
    // We need two outputs (r and its norm). run_graph returns one flat buffer,
    // so run two graphs that share the cheap recompute.
    std::vector<float> r;       // residual after FFN1, [T, D] row-major
    {
        bool ok = pk::run_graph(mem_bytes, /*n_threads*/4,
            [&](ggml_context* ctx) -> ggml_tensor* {
                ggml_tensor* xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
                std::memcpy(xt->data, x.data(), (size_t)T * D * sizeof(float));
                ggml_tensor* h = layer_norm(ctx, xt, "norm_feed_forward1");
                h = feed_forward(ctx, h, "feed_forward1");
                h = ggml_scale(ctx, h, 0.5f);          // fc_factor
                ggml_tensor* res = ggml_add(ctx, xt, h);
                return res;                            // [D, T] -> row-major [T, D]
            }, r);
        assert(ok && "conformer FFN1 graph failed"); (void)ok;
    }

    // attention input = norm_self_att(r)
    std::vector<float> attn_in;
    {
        bool ok = pk::run_graph(mem_bytes, /*n_threads*/4,
            [&](ggml_context* ctx) -> ggml_tensor* {
                ggml_tensor* rt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
                std::memcpy(rt->data, r.data(), (size_t)T * D * sizeof(float));
                return layer_norm(ctx, rt, "norm_self_att");
            }, attn_in);
        assert(ok && "conformer norm_self_att graph failed"); (void)ok;
    }

    // === Stage B: MHSA via the shared RelPosAttention module. ===
    std::vector<float> attn_out;
    {
        RelPosAttention attn(ml_, layer_idx_);
        attn.forward(attn_in, T, pos_emb, pos_len, valid_len, attn_out);
    }
    // r = r + attn_out
    for (size_t i = 0; i < r.size(); ++i) r[i] += attn_out[i];

    // === Stage C: conv module on norm_conv(r); r = r + conv_out. ===
    {
        bool ok = pk::run_graph(mem_bytes, /*n_threads*/4,
            [&](ggml_context* ctx) -> ggml_tensor* {
                ggml_tensor* rt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
                std::memcpy(rt->data, r.data(), (size_t)T * D * sizeof(float));

                // norm_conv -> [D, T]
                ggml_tensor* c = layer_norm(ctx, rt, "norm_conv");

                // ConformerConvolution operates on [channels, time]. Our [D, T]
                // layout (D fastest) is exactly that: ne0=D=channels, ne1=T=time.
                // -- pointwise_conv1 (Conv1d d->2d, k=1): 1x1 conv == linear over
                //    channels. weight torch [2d, d, 1] -> ggml [1, d, 2d]; squeeze
                //    KW=1 to get a [d, 2d] mul_mat weight.
                ggml_tensor* pw1w = clone_weight(ctx, ml, pre + "conv.pointwise_conv1.weight");
                pw1w = ggml_reshape_2d(ctx, pw1w, D, 2 * D); // [in=d, out=2d]
                ggml_tensor* pw1b = clone_weight_opt(ctx, ml, pre + "conv.pointwise_conv1.bias");
                ggml_tensor* y = ggml_mul_mat(ctx, pw1w, c); // [2d, T]
                if (pw1b) y = ggml_add(ctx, y, pw1b);

                // -- GLU over channel dim (NeMo F.glu(x, dim=1)): first half *
                //    sigmoid(second half). y ne0 spans 2d channels [0:d]|[d:2d].
                ggml_tensor* a = ggml_view_2d(ctx, y, D, T, y->nb[1], 0);
                ggml_tensor* b = ggml_view_2d(ctx, y, D, T, y->nb[1], (size_t)D * y->nb[0]);
                ggml_tensor* glu = ggml_mul(ctx, ggml_cont(ctx, a),
                                            ggml_sigmoid(ctx, ggml_cont(ctx, b))); // [d, T]

                // -- pad_mask: zero padded time positions before depthwise conv so
                //    padding does not smear into valid frames. Multiply by time
                //    mask [1, T] (broadcasts over channels).
                if (valid_len < T) {
                    ggml_tensor* tmask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, T);
                    float* md = (float*)tmask->data;
                    for (int t = 0; t < T; ++t) md[t] = (t < valid_len) ? 1.0f : 0.0f;
                    glu = ggml_mul(ctx, glu, tmask);
                }

                // -- depthwise_conv (Conv1d d->d, k=K, groups=d, symmetric pad).
                //    ggml_conv_1d_dw kernel a ne=[KW,1,C], data b ne=[W=T,C].
                //    glu is [d=C, T]; transpose to [T, C] for the conv data layout.
                ggml_tensor* glu_tc = ggml_cont(ctx, ggml_transpose(ctx, glu)); // [T, C]
                ggml_tensor* dww = clone_weight(ctx, ml, pre + "conv.depthwise_conv.weight"); // [K,1,C]
                // Depthwise conv, F32 throughout. ggml_conv_1d_dw() hardcodes an
                // F16 im2col, which loses precision on large activations and
                // would compound across stacked layers; replicate its internals
                // with an F32 im2col instead. (kernel a=[K,1,C], data b=[T,1,C,1]
                // -> im2col [K, T, C] -> mul_mat with a -> [1, T, C].)
                ggml_tensor* dw;
                {
                    // mirror ggml_conv_1d_dw() internals (b reshaped to [W,1,IC,N]).
                    ggml_tensor* nb = ggml_reshape_4d(ctx, glu_tc,
                                          glu_tc->ne[0], 1, glu_tc->ne[1], 1); // [T,1,C,1]
                    ggml_tensor* ic = ggml_im2col(ctx, dww, nb, /*s0*/1, /*s1*/0,
                                                  /*p0*/pad, /*p1*/0, /*d0*/1, /*d1*/0,
                                                  /*is_2D*/false, GGML_TYPE_F32);
                    ggml_tensor* r2 = ggml_mul_mat(ctx, ic, dww);
                    dw = ggml_reshape_3d(ctx, r2, r2->ne[0], r2->ne[2], 1); // [OW=T, C, 1]
                }
                // dw ne = [OW=T, C, 1]; bias [C] added per channel, then back to
                // [C, T] (channels fastest) for batch_norm / activation / pw2.
                dw = ggml_reshape_2d(ctx, dw, T, D);                  // [T, C]
                ggml_tensor* dwb = clone_weight_opt(ctx, ml, pre + "conv.depthwise_conv.bias"); // [C]
                ggml_tensor* dwt = ggml_cont(ctx, ggml_transpose(ctx, dw)); // [C, T]
                if (dwb) dwt = ggml_add(ctx, dwt, dwb);               // broadcast [C] over T

                // -- batch_norm (inference): y = (x-mean)/sqrt(var+eps)*g + b,
                //    per-channel. Fold into constant scale/shift [C] in C++
                //    (params are inference constants):
                //      scale = g / sqrt(var+eps); shift = b - mean*scale.
                ggml_tensor* scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
                ggml_tensor* shift = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
                {
                    const float* g = ggml_get_data_f32(ml.tensor(pre + "conv.batch_norm.weight"));
                    const float* bb = ggml_get_data_f32(ml.tensor(pre + "conv.batch_norm.bias"));
                    const float* m = ggml_get_data_f32(ml.tensor(pre + "conv.batch_norm.running_mean"));
                    const float* var = ggml_get_data_f32(ml.tensor(pre + "conv.batch_norm.running_var"));
                    float* sc = (float*)scale->data;
                    float* sh = (float*)shift->data;
                    for (int c = 0; c < D; ++c) {
                        sc[c] = g[c] / std::sqrt(var[c] + bn_eps);
                        sh[c] = bb[c] - m[c] * sc[c];
                    }
                }
                ggml_tensor* bn = ggml_add(ctx, ggml_mul(ctx, dwt, scale), shift); // [C, T]

                // -- SiLU (Swish), then pointwise_conv2 (Conv1d d->d, k=1).
                bn = ggml_silu(ctx, bn);
                ggml_tensor* pw2w = clone_weight(ctx, ml, pre + "conv.pointwise_conv2.weight");
                pw2w = ggml_reshape_2d(ctx, pw2w, D, D); // [in=d, out=d]
                ggml_tensor* pw2b = clone_weight_opt(ctx, ml, pre + "conv.pointwise_conv2.bias");
                ggml_tensor* cout = ggml_mul_mat(ctx, pw2w, bn); // [d, T]
                if (pw2b) cout = ggml_add(ctx, cout, pw2b);
                return cout; // [D, T] -> row-major [T, D]; this is layers[i].conv output
            }, conv_out);
        assert(ok && "conformer conv graph failed"); (void)ok;
    }
    // r = r + conv_out
    for (size_t i = 0; i < r.size(); ++i) r[i] += conv_out[i];

    // === Stage D: r = r + 0.5 * FFN2(norm_ff2(r)); out = norm_out(r). ===
    {
        bool ok = pk::run_graph(mem_bytes, /*n_threads*/4,
            [&](ggml_context* ctx) -> ggml_tensor* {
                ggml_tensor* rt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
                std::memcpy(rt->data, r.data(), (size_t)T * D * sizeof(float));
                ggml_tensor* h = layer_norm(ctx, rt, "norm_feed_forward2");
                h = feed_forward(ctx, h, "feed_forward2");
                h = ggml_scale(ctx, h, 0.5f);
                ggml_tensor* res = ggml_add(ctx, rt, h);
                res = layer_norm(ctx, res, "norm_out");
                return res; // [D, T] -> row-major [T, D]
            }, out);
        assert(ok && "conformer FFN2/out graph failed"); (void)ok;
    }
}

} // namespace pk
