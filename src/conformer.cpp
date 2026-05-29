#include "conformer.hpp"
#include "relpos_attention.hpp"
#include "ggml_graph.hpp"
#include "backend.hpp"
#include "ggml.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace pk {

// Weights from the loader are brought into the graph as inputs via the shared
// pk::clone_weight / pk::clone_weight_opt (backend.cpp). Quantization-allowlist
// linears may be f16/q8_0 (ggml_mul_mat dequantizes src0 on the fly; raw bytes
// copied through). Optional weights (conv biases, absent in bias=False
// checkpoints) use clone_weight_opt. std::string-name overloads keep the
// existing call sites (pre + suffix) unchanged.
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
// the conv output tensor, ne [D, T] (row-major [T, D]).
//
// NeMo `ConformerConvolution.forward`:
//   x = pointwise_conv1(x)            # Conv1d d->2d, k=1
//   x = F.glu(x, dim=1)               # GLU over the channel dim
//   x = masked_fill(pad_mask, 0)      # zero padded time frames
//   x = depthwise_conv(x)             # CausalConv1D d->d, k=K, groups=d
//   if layer_norm: x = LN(x.T).T      # transpose to channels-last, nn.LayerNorm
//   else:          x = batch_norm(x)  # per-channel affine (running-stat fold)
//   x = SiLU(x); x = pointwise_conv2(x)
//
// `conv_norm_type`: "layer_norm" normalizes over the d_model channel dim per
// time-frame (eps 1e-5) with gamma/beta = conv.batch_norm.{weight,bias} — the
// SAME tensor names as batch_norm (NeMo always names the attr `batch_norm`).
// "batch_norm" folds the inference running stats into a per-channel scale/shift.
//
// `conv_causal`: causal depthwise conv pads the time axis LEFT by K-1, RIGHT 0
// (vs the symmetric (K-1)/2 each side). For the causal case the input is padded
// explicitly with ggml_pad_ext (im2col takes one padding value per axis); the
// symmetric case keeps the existing im2col p0=(K-1)/2 path byte-identically.
// `host_pool` keeps host-side mask/scale/shift buffers alive until the enclosing
// Backend::compute finishes (their bytes are copied into the graph AFTER alloc,
// so they must outlive the build lambda). Each entry is owned via unique_ptr so
// the storage addresses stay stable as more are appended.
static ggml_tensor* build_conv_module(ggml_context* ctx, const ModelLoader& ml,
                                      const std::string& pre, ggml_tensor* c,
                                      int D, int T, int K, int valid_len,
                                      const std::string& conv_norm_type,
                                      bool conv_causal,
                                      std::vector<std::unique_ptr<std::vector<float>>>& host_pool) {
    const float ln_eps = 1e-5f;
    const float bn_eps = 1e-5f;
    const int pad = (K - 1) / 2;  // symmetric padding (offline model)

    // -- pointwise_conv1 (Conv1d d->2d, k=1): 1x1 conv == linear over channels.
    //    weight torch [2d, d, 1] -> ggml [1, d, 2d]; squeeze KW=1 to [d, 2d].
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

    // -- pad_mask: zero padded time positions before depthwise conv so padding
    //    does not smear into valid frames. Multiply by time mask [1, T]
    //    (broadcasts over channels).
    if (valid_len < T) {
        host_pool.emplace_back(new std::vector<float>(T));
        std::vector<float>& md = *host_pool.back();
        for (int t = 0; t < T; ++t) md[t] = (t < valid_len) ? 1.0f : 0.0f;
        int64_t tm_ne[2] = {1, T};
        ggml_tensor* tmask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, tm_ne,
                                 md.data(), md.size() * sizeof(float));
        glu = ggml_mul(ctx, glu, tmask);
    }

    // -- depthwise_conv (Conv1d d->d, k=K, groups=d).
    //    ggml_conv_1d_dw kernel a ne=[KW,1,C], data b ne=[W=T,C].
    //    glu is [d=C, T]; transpose to [T, C] for the conv data layout.
    ggml_tensor* glu_tc = ggml_cont(ctx, ggml_transpose(ctx, glu)); // [T, C]
    ggml_tensor* dww = clone_weight(ctx, ml, pre + "conv.depthwise_conv.weight"); // [K,1,C]
    // Depthwise conv, F32 throughout. ggml_conv_1d_dw() hardcodes an F16 im2col,
    // which loses precision on large activations and would compound across
    // stacked layers; replicate its internals with an F32 im2col instead.
    // (kernel a=[K,1,C], data b=[T,1,C,1] -> im2col [K,OW,C] -> mul_mat -> [1,OW,C].)
    ggml_tensor* dw;
    {
        ggml_tensor* nb = ggml_reshape_4d(ctx, glu_tc,
                              glu_tc->ne[0], 1, glu_tc->ne[1], 1); // [T,1,C,1]
        ggml_tensor* ic;
        if (conv_causal) {
            // Causal: pad the time axis (ne0=W) LEFT by K-1, RIGHT by 0. im2col
            // takes a single symmetric p0, so pad explicitly and use p0=0. With
            // OW = T + (K-1) - (K-1) = T, output time length is preserved.
            ggml_tensor* nbp = ggml_pad_ext(ctx, nb, /*lp0*/K - 1, /*rp0*/0,
                                            0, 0, 0, 0, 0, 0); // [T+K-1,1,C,1]
            ic = ggml_im2col(ctx, dww, nbp, /*s0*/1, /*s1*/0,
                             /*p0*/0, /*p1*/0, /*d0*/1, /*d1*/0,
                             /*is_2D*/false, GGML_TYPE_F32);
        } else {
            // Symmetric (offline): pad (K-1)/2 each side directly via im2col.
            ic = ggml_im2col(ctx, dww, nb, /*s0*/1, /*s1*/0,
                             /*p0*/pad, /*p1*/0, /*d0*/1, /*d1*/0,
                             /*is_2D*/false, GGML_TYPE_F32);
        }
        ggml_tensor* r2 = ggml_mul_mat(ctx, ic, dww);
        dw = ggml_reshape_3d(ctx, r2, r2->ne[0], r2->ne[2], 1); // [OW=T, C, 1]
    }
    // dw ne = [OW=T, C, 1]; bias [C] added per channel, then back to [C, T]
    // (channels fastest) for the norm / activation / pw2.
    dw = ggml_reshape_2d(ctx, dw, T, D);                  // [T, C]
    ggml_tensor* dwb = clone_weight_opt(ctx, ml, pre + "conv.depthwise_conv.bias"); // [C]
    ggml_tensor* dwt = ggml_cont(ctx, ggml_transpose(ctx, dw)); // [C, T]
    if (dwb) dwt = ggml_add(ctx, dwt, dwb);               // broadcast [C] over T

    // -- norm (between depthwise conv and SiLU).
    ggml_tensor* normed;
    if (conv_norm_type == "layer_norm") {
        // NeMo: x = x.transpose(1,2); x = LayerNorm(d_model)(x); x = x.transpose(1,2).
        // i.e. normalize over the CHANNEL dim per time-frame. dwt is [C, T]
        // (ne0=C=channels), so ggml_norm normalizes over ne0=C directly — exactly
        // channels-last LayerNorm. gamma/beta = conv.batch_norm.{weight,bias}.
        ggml_tensor* g = clone_weight(ctx, ml, pre + "conv.batch_norm.weight"); // [C]
        ggml_tensor* bb = clone_weight(ctx, ml, pre + "conv.batch_norm.bias");  // [C]
        normed = ggml_norm(ctx, dwt, ln_eps);             // normalize over ne0=C
        normed = ggml_mul(ctx, normed, g);                // *gamma  (broadcast [C] over T)
        normed = ggml_add(ctx, normed, bb);               // +beta
    } else {
        // batch_norm (inference): y = (x-mean)/sqrt(var+eps)*g + b, per-channel.
        // Fold into constant scale/shift [C] (inference constants):
        //   scale = g / sqrt(var+eps); shift = b - mean*scale.
        // Computed host-side from the loader running stats, fed as inputs.
        host_pool.emplace_back(new std::vector<float>(D));
        host_pool.emplace_back(new std::vector<float>(D));
        std::vector<float>& sc = *host_pool[host_pool.size() - 2];
        std::vector<float>& sh = *host_pool[host_pool.size() - 1];
        const float* g = ggml_get_data_f32(ml.tensor(pre + "conv.batch_norm.weight"));
        const float* bb = ggml_get_data_f32(ml.tensor(pre + "conv.batch_norm.bias"));
        const float* m = ggml_get_data_f32(ml.tensor(pre + "conv.batch_norm.running_mean"));
        const float* var = ggml_get_data_f32(ml.tensor(pre + "conv.batch_norm.running_var"));
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
    // Two conv-module norm variants are supported (selected by the conv_norm_type
    // KV): the offline models use batch_norm (the affine running-stat fold below);
    // the streaming models (e.g. parakeet_realtime_eou_120m-v1) use layer_norm
    // (transpose -> nn.LayerNorm over the channel dim -> transpose back). NeMo
    // stores the layer_norm gamma/beta under the SAME tensor names
    // `conv.batch_norm.{weight,bias}` (the module attr is always `batch_norm`).
    assert((conv_norm_type_ == "batch_norm" || conv_norm_type_ == "layer_norm") &&
           "ConformerLayer supports conv_norm_type in {batch_norm, layer_norm}");
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
    const float ln_eps = 1e-5f;            // LayerNorm eps (NeMo nn.LayerNorm default)

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
                int64_t xt_ne[2] = {D, T};
                ggml_tensor* xt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
                                      xt_ne, x.data(), (size_t)T * D * sizeof(float));
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
                int64_t rt_ne[2] = {D, T};
                ggml_tensor* rt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
                                      rt_ne, r.data(), (size_t)T * D * sizeof(float));
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
    // ConformerConvolution operates on [channels, time]. Our [D, T] layout
    // (D fastest) is exactly that: ne0=D=channels, ne1=T=time.
    {
        std::vector<std::unique_ptr<std::vector<float>>> host_pool;
        bool ok = pk::run_graph(mem_bytes, /*n_threads*/4,
            [&](ggml_context* ctx) -> ggml_tensor* {
                int64_t rt_ne[2] = {D, T};
                ggml_tensor* rt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
                                      rt_ne, r.data(), (size_t)T * D * sizeof(float));
                ggml_tensor* c = layer_norm(ctx, rt, "norm_conv"); // -> [D, T]
                return build_conv_module(ctx, ml, pre, c, D, T, K, valid_len,
                                         conv_norm_type_, conv_causal_, host_pool);
            }, conv_out);
        assert(ok && "conformer conv graph failed"); (void)ok;
    }
    // r = r + conv_out
    for (size_t i = 0; i < r.size(); ++i) r[i] += conv_out[i];

    // === Stage D: r = r + 0.5 * FFN2(norm_ff2(r)); out = norm_out(r). ===
    {
        bool ok = pk::run_graph(mem_bytes, /*n_threads*/4,
            [&](ggml_context* ctx) -> ggml_tensor* {
                int64_t rt_ne[2] = {D, T};
                ggml_tensor* rt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
                                      rt_ne, r.data(), (size_t)T * D * sizeof(float));
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

void ConformerLayer::conv_module_forward(const std::vector<float>& conv_in, int T,
                                         int valid_len,
                                         std::vector<float>& out) const {
    const int D = d_model_;
    const int K = conv_kernel_;
    assert((int)conv_in.size() == T * D);

    const std::string pre = "encoder.layers." + std::to_string(layer_idx_) + ".";
    const ModelLoader& ml = ml_;
    const size_t mem_bytes =
        (size_t)128 * 1024 * 1024 + (size_t)D * (size_t)T * 64 * sizeof(float);

    // conv_in is norm_conv(residual), row-major [T, D] -> ggml [D, T]
    // (channels fastest); feed it straight into the conv sub-module (NeMo's
    // ConformerConvolution.forward, which starts AFTER norm_conv).
    std::vector<std::unique_ptr<std::vector<float>>> host_pool;
    bool ok = pk::run_graph(mem_bytes, /*n_threads*/4,
        [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t c_ne[2] = {D, T};
            ggml_tensor* c = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, c_ne,
                                 conv_in.data(), (size_t)T * D * sizeof(float));
            return build_conv_module(ctx, ml, pre, c, D, T, K, valid_len,
                                     conv_norm_type_, conv_causal_, host_pool);
        }, out);
    assert(ok && "conv_module_forward graph failed"); (void)ok;
}

} // namespace pk
