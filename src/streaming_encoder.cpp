#include "streaming_encoder.hpp"
#include "subsampling.hpp"
#include "pos_enc.hpp"
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
// pk::clone_weight / pk::clone_weight_opt (backend.cpp): allowlisted linears may
// be f16/q8_0 and are dequantized by ggml_mul_mat. std::string-name overloads
// keep the existing call sites unchanged.
static ggml_tensor* clone_weight(ggml_context* ctx, const ModelLoader& ml,
                                 const std::string& name) {
    return pk::clone_weight(ctx, ml, name.c_str());
}
static ggml_tensor* clone_weight_opt(ggml_context* ctx, const ModelLoader& ml,
                                     const std::string& name) {
    return pk::clone_weight_opt(ctx, ml, name.c_str());
}

StreamingEncoder::StreamingEncoder(const ModelLoader& ml) : ml_(ml) {
    const ParakeetConfig& c = ml.config();
    d_model_     = (int)c.d_model;
    n_layers_    = (int)c.n_layers;
    n_heads_     = (int)c.n_heads;
    conv_kernel_ = (int)c.conv_kernel;
    xscaling_    = c.xscaling;
    assert(n_heads_ > 0 && d_model_ % n_heads_ == 0);
    d_head_ = d_model_ / n_heads_;
    left_pad_ = conv_kernel_ - 1;   // causal depthwise conv left-context (8)
    assert(c.streaming.present && "StreamingEncoder requires a streaming model");
    const StreamingCfg& s = c.streaming;
    // chunk_size/shift_size/pre_encode_cache_size are stored as the [a,b] arrays;
    // the cache-aware streaming buffer uses the [0] value for the FIRST chunk and
    // the [1] value thereafter (CacheAwareStreamingAudioBuffer.__iter__).
    auto pick = [](const std::vector<int32_t>& v, int idx, int dflt) {
        return (idx < (int)v.size()) ? (int)v[idx] : dflt;
    };
    chunk_first_       = pick(s.chunk_size, 0, 9);
    chunk_main_        = pick(s.chunk_size, 1, 16);
    pre_cache_         = pick(s.pre_encode_cache_size, 1, 9);
    drop_extra_        = s.drop_extra_pre_encoded;
    last_channel_cache_= s.last_channel_cache_size;
    valid_out_len_     = s.valid_out_len;
    att_left_  = c.att_context_left;
    att_right_ = c.att_context_right;
    assert(c.causal_downsampling && "streaming model expects causal subsampling");
    assert(c.conv_causal && "streaming model expects causal depthwise conv");
    assert(c.att_context_style == "chunked_limited");
    reset();
}

void StreamingEncoder::reset() {
    step_ = 0;
    clc_len_ = 0;
    cache_time_.assign(n_layers_,
                       std::vector<float>((size_t)left_pad_ * d_model_, 0.0f));
    cache_channel_.assign(n_layers_,
                          std::vector<float>((size_t)last_channel_cache_ * d_model_, 0.0f));
}

// One conformer layer in streaming mode. Mirrors ConformerLayer::forward but:
//   - MHSA prepends the [cache_len, d_model] attention K/V cache (NeMo
//     MultiHeadAttention.update_cache) and uses the streaming chunked-limited +
//     pad/offset mask over [cache_len+Tc] keys.
//   - Conv prepends the [d_model, left_pad] depthwise-conv cache (NeMo
//     CausalConv1D.update_cache) before the depthwise conv (causal, no pad).
//   - both caches are updated in place after the layer runs.
// `x` is the chunk input [Tc, d_model] row-major; pos_emb is [pos_len, d_model]
// for pos_len = 2*(Tc+cache_len)-1. cache_len = last_channel_cache_ (70).
std::vector<float> StreamingEncoder::layer_step(int layer_idx,
                                                const std::vector<float>& x,
                                                int Tc,
                                                const std::vector<float>& pos_emb,
                                                int pos_len, int cache_len) {
    const int D  = d_model_;
    const int H  = n_heads_;
    const int dk = d_head_;
    const int K  = conv_kernel_;
    const int LP = left_pad_;
    const float ln_eps = 1e-5f;
    const float scale = 1.0f / std::sqrt((float)dk);
    const int Tk = cache_len + Tc;            // attention key length
    assert((int)x.size() == Tc * D);
    assert(pos_len == 2 * (Tc + cache_len) - 1);

    const std::string pre = "encoder.layers." + std::to_string(layer_idx) + ".";
    const ModelLoader& ml = ml_;
    const ParakeetConfig& cfg = ml.config();
    const std::string cnt = cfg.conv_norm_type;
    const size_t mem_bytes =
        (size_t)192 * 1024 * 1024 +
        (size_t)H * (size_t)(Tc + 1) * (size_t)pos_len * 64 * sizeof(float) +
        (size_t)d_model_ * (size_t)(Tk + 64) * 64 * sizeof(float);

    auto layer_norm = [&](ggml_context* ctx, ggml_tensor* in, const std::string& nm) {
        ggml_tensor* g = clone_weight(ctx, ml, pre + nm + ".weight");
        ggml_tensor* b = clone_weight(ctx, ml, pre + nm + ".bias");
        ggml_tensor* y = ggml_norm(ctx, in, ln_eps);
        y = ggml_mul(ctx, y, g);
        y = ggml_add(ctx, y, b);
        return y;
    };
    auto linear = [&](ggml_context* ctx, ggml_tensor* in,
                      const std::string& nm, bool bias) {
        ggml_tensor* W = clone_weight(ctx, ml, pre + nm + ".weight");
        ggml_tensor* y = ggml_mul_mat(ctx, W, in);
        if (bias) { ggml_tensor* B = clone_weight_opt(ctx, ml, pre + nm + ".bias");
                    if (B) y = ggml_add(ctx, y, B); }
        return y;
    };
    auto feed_forward = [&](ggml_context* ctx, ggml_tensor* in, const std::string& ff) {
        ggml_tensor* h = linear(ctx, in, ff + ".linear1", true);
        h = ggml_silu(ctx, h);
        h = linear(ctx, h, ff + ".linear2", true);
        return h;
    };

    // === Stage A: r = x + 0.5*FFN1(norm_ff1(x)); attn_in = norm_self_att(r). ===
    std::vector<float> r;
    {
        bool ok = pk::run_graph(mem_bytes, 4, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t xt_ne[2] = {D, Tc};
            ggml_tensor* xt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, xt_ne,
                                  x.data(), (size_t)Tc * D * sizeof(float));
            ggml_tensor* h = layer_norm(ctx, xt, "norm_feed_forward1");
            h = feed_forward(ctx, h, "feed_forward1");
            h = ggml_scale(ctx, h, 0.5f);
            return ggml_add(ctx, xt, h);
        }, r);
        assert(ok && "stream FFN1 graph failed"); (void)ok;
    }
    std::vector<float> attn_in;
    {
        bool ok = pk::run_graph(mem_bytes, 4, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t rt_ne[2] = {D, Tc};
            ggml_tensor* rt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, rt_ne,
                                  r.data(), (size_t)Tc * D * sizeof(float));
            return layer_norm(ctx, rt, "norm_self_att");
        }, attn_in);
        assert(ok && "stream norm_self_att graph failed"); (void)ok;
    }

    // === Stage B: streaming MHSA (RelPosAttention with attention cache). ===
    // K/V = cat([cache_channel(cache_len), attn_in(Tc)]); Q = attn_in(Tc).
    std::vector<float> attn_out;
    {
        const std::string ap = pre + "self_attn.";
        // kv host buffer [D, Tk] = [cache (cache_len) ; attn_in (Tc)] assembled
        // on the host (the deferred input copy is a single contiguous block).
        std::vector<float> kv_host((size_t)Tk * D);
        std::memcpy(kv_host.data(), cache_channel_[layer_idx].data(),
                    (size_t)cache_len * D * sizeof(float));
        std::memcpy(kv_host.data() + (size_t)cache_len * D, attn_in.data(),
                    (size_t)Tc * D * sizeof(float));
        std::vector<float> mask_host;  // filled below, alive for the call.
        bool ok = pk::run_graph(mem_bytes, 4, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t kvt_ne[2] = {D, Tk};
            ggml_tensor* kvt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, kvt_ne,
                                   kv_host.data(), kv_host.size() * sizeof(float));
            int64_t qt_ne[2] = {D, Tc};
            ggml_tensor* qt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, qt_ne,
                                  attn_in.data(), (size_t)Tc * D * sizeof(float));
            int64_t pe_ne[2] = {D, pos_len};
            ggml_tensor* pe = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, pe_ne,
                                  pos_emb.data(), (size_t)pos_len * D * sizeof(float));

            auto lin = [&](const char* w, const char* b, ggml_tensor* in) {
                ggml_tensor* W = clone_weight(ctx, ml, ap + w);
                ggml_tensor* y = ggml_mul_mat(ctx, W, in);
                if (b && ml.tensor(ap + b)) y = ggml_add(ctx, y, clone_weight(ctx, ml, ap + b));
                return y;
            };
            ggml_tensor* q = lin("linear_q.weight", "linear_q.bias", qt);  // [D, Tc]
            ggml_tensor* k = lin("linear_k.weight", "linear_k.bias", kvt); // [D, Tk]
            ggml_tensor* v = lin("linear_v.weight", "linear_v.bias", kvt); // [D, Tk]
            ggml_tensor* p = lin("linear_pos.weight", nullptr, pe);        // [D, P]

            auto to_heads = [&](ggml_tensor* t, int n) {
                t = ggml_reshape_3d(ctx, t, dk, H, n);
                return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3)); // [dk, n, H]
            };
            ggml_tensor* qh = to_heads(q, Tc);
            ggml_tensor* kh = to_heads(k, Tk);
            ggml_tensor* ph = to_heads(p, pos_len);

            ggml_tensor* bu = clone_weight(ctx, ml, ap + "pos_bias_u");
            ggml_tensor* bv = clone_weight(ctx, ml, ap + "pos_bias_v");
            bu = ggml_reshape_3d(ctx, bu, dk, 1, H);
            bv = ggml_reshape_3d(ctx, bv, dk, 1, H);
            ggml_tensor* qu = ggml_add(ctx, qh, bu); // [dk, Tc, H]
            ggml_tensor* qv = ggml_add(ctx, qh, bv); // [dk, Tc, H]

            // ac = q_u @ k^T -> [Tk, Tc, H].
            ggml_tensor* ac = ggml_mul_mat(ctx, kh, qu);

            // bd = q_v @ p^T -> [P, Tc, H]; rel_shift; slice first Tk pos cols.
            // pos_len = 2*Pn-1, Pn = Tc+cache_len. NeMo rel_shift over query Tc and
            // pos_len P, then drops to k.size = Tk.
            ggml_tensor* bd = ggml_mul_mat(ctx, ph, qv); // [P, Tc, H]
            bd = ggml_pad_ext(ctx, bd, 1, 0, 0, 0, 0, 0, 0, 0);      // [P+1, Tc, H]
            bd = ggml_reshape_3d(ctx, bd, Tc, pos_len + 1, H);        // [Tc, P+1, H]
            bd = ggml_view_3d(ctx, bd, Tc, pos_len, H,
                              bd->nb[1], bd->nb[2], bd->nb[1]);       // drop first row
            bd = ggml_cont(ctx, bd);
            bd = ggml_reshape_3d(ctx, bd, pos_len, Tc, H);            // [P, Tc, H]
            bd = ggml_view_3d(ctx, bd, Tk, Tc, H, bd->nb[1], bd->nb[2], 0); // first Tk pos
            bd = ggml_cont(ctx, bd);

            ggml_tensor* scores = ggml_add(ctx, ac, bd); // [Tk, Tc, H]

            // Streaming attention mask [Tk(key), Tc(query)] (0 visible / -inf):
            //   global key index   gk = kj            in [0, Tk)
            //   global query index gq = cache_len + qi in [cache_len, cache_len+Tc)
            //   (1) empty-cache mask: cache cols [0, cache_len-clc_len) are not yet
            //       filled -> masked.
            //   (2) chunked_limited (chunk=att_right+1=2, left=att_left/chunk=35):
            //       visible iff 0 <= gq/chunk - gk/chunk <= left_chunks.
            const int chunk = att_right_ + 1;
            const int left_chunks = (chunk > 0) ? (att_left_ / chunk) : 0;
            const int empty_cache = cache_len - clc_len_;
            mask_host.assign((size_t)Tk * Tc, 0.0f);
            {
                float* md = mask_host.data();
                const float ninf = -INFINITY;
                for (int qi = 0; qi < Tc; ++qi) {
                    const int gq = cache_len + qi;
                    const int cq = gq / chunk;
                    for (int kj = 0; kj < Tk; ++kj) {
                        bool vis = (kj >= empty_cache);
                        if (vis) {
                            const int ck = kj / chunk;
                            const int diff = cq - ck;
                            vis = (diff >= 0 && diff <= left_chunks);
                        }
                        md[(size_t)qi * Tk + kj] = vis ? 0.0f : ninf;
                    }
                }
            }
            int64_t mask_ne[2] = {Tk, Tc};
            ggml_tensor* mask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
                                    mask_ne, mask_host.data(),
                                    mask_host.size() * sizeof(float));
            ggml_tensor* attn = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);

            ggml_tensor* vh = to_heads(v, Tk);           // [dk, Tk, H]
            ggml_tensor* vtk = ggml_cont(ctx, ggml_permute(ctx, vh, 1, 0, 2, 3)); // [Tk, dk, H]
            ggml_tensor* ctxh = ggml_mul_mat(ctx, vtk, attn); // [dk, Tc, H]
            ggml_tensor* merged = ggml_cont(ctx, ggml_permute(ctx, ctxh, 0, 2, 1, 3));
            merged = ggml_reshape_2d(ctx, merged, D, Tc);
            return lin("linear_out.weight", "linear_out.bias", merged); // [D, Tc]
        }, attn_out);
        assert(ok && "stream MHSA graph failed"); (void)ok;
    }
    for (size_t i = 0; i < r.size(); ++i) r[i] += attn_out[i];

    // Update attention cache (NeMo update_cache, cache_drop_size=0):
    //   next = cat([cache[Tc:], attn_in[:Tc]]) keep last cache_len. With cache_len
    // fixed and Tc<=cache_len: drop oldest Tc, append the Tc current frames.
    // clc_len grows by Tc (clamped to cache_len).
    {
        std::vector<float>& cc = cache_channel_[layer_idx];
        std::vector<float> nc((size_t)cache_len * D);
        const int keep = cache_len - Tc;
        if (keep > 0)
            std::memcpy(nc.data(), cc.data() + (size_t)Tc * D,
                        (size_t)keep * D * sizeof(float));
        std::memcpy(nc.data() + (size_t)keep * D, attn_in.data(),
                    (size_t)Tc * D * sizeof(float));
        cc.swap(nc);
    }

    // === Stage C: streaming conv module (causal depthwise + conv cache). ===
    // The conv graph returns BOTH the conv output [D, Tc] (rows 0..Tc-1 after
    // transpose -> [Tc, D]) AND the GLU output [Tc, D] (the depthwise-conv input)
    // so we can update the conv cache (last LP frames of [cache ; glu]). We pack
    // them by returning a [D, Tc + Tc] tensor: cols [0,Tc) = conv_out, cols
    // [Tc,2Tc) = glu. run_graph flattens row-major [2Tc, D].
    std::vector<float> packed; // [2Tc, D] row-major: [conv_out(Tc) ; glu(Tc)]
    {
        // batch_norm fold scale/shift host buffers (filled below, alive for the
        // call). cache_time is read directly from the layer cache (alive too).
        std::vector<float> sc_host, sh_host;
        bool ok = pk::run_graph(mem_bytes, 4, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t rt_ne[2] = {D, Tc};
            ggml_tensor* rt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, rt_ne,
                                  r.data(), (size_t)Tc * D * sizeof(float));
            // NOTE: r already includes attn_out; norm_conv operates on that
            // residual (NeMo: residual after MHSA -> norm_conv -> conv).
            ggml_tensor* c = layer_norm(ctx, rt, "norm_conv"); // [D, Tc]

            ggml_tensor* pw1w = clone_weight(ctx, ml, pre + "conv.pointwise_conv1.weight");
            pw1w = ggml_reshape_2d(ctx, pw1w, D, 2 * D);
            ggml_tensor* pw1b = clone_weight_opt(ctx, ml, pre + "conv.pointwise_conv1.bias");
            ggml_tensor* y = ggml_mul_mat(ctx, pw1w, c); // [2D, Tc]
            if (pw1b) y = ggml_add(ctx, y, pw1b);
            ggml_tensor* a = ggml_view_2d(ctx, y, D, Tc, y->nb[1], 0);
            ggml_tensor* b = ggml_view_2d(ctx, y, D, Tc, y->nb[1], (size_t)D * y->nb[0]);
            ggml_tensor* glu = ggml_cont(ctx, ggml_mul(ctx, ggml_cont(ctx, a),
                                ggml_sigmoid(ctx, ggml_cont(ctx, b)))); // [D, Tc]

            // dw_in [D, LP+Tc] = [cache_time(LP) ; glu(Tc)].
            int64_t cache_ne[2] = {D, LP};
            ggml_tensor* cache_t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
                                       cache_ne, cache_time_[layer_idx].data(),
                                       (size_t)LP * D * sizeof(float));
            ggml_tensor* dw_in = ggml_concat(ctx, cache_t, glu, /*dim=*/1); // [D, LP+Tc]

            ggml_tensor* glu_tc = ggml_cont(ctx, ggml_transpose(ctx, dw_in)); // [LP+Tc, D]
            ggml_tensor* dww = clone_weight(ctx, ml, pre + "conv.depthwise_conv.weight"); // [K,1,C]
            ggml_tensor* dw;
            {
                ggml_tensor* nb = ggml_reshape_4d(ctx, glu_tc, glu_tc->ne[0], 1, glu_tc->ne[1], 1);
                ggml_tensor* ic = ggml_im2col(ctx, dww, nb, 1, 0, 0, 0, 1, 0,
                                              /*is_2D*/false, GGML_TYPE_F32);
                ggml_tensor* r2 = ggml_mul_mat(ctx, ic, dww);
                dw = ggml_reshape_3d(ctx, r2, r2->ne[0], r2->ne[2], 1); // [Tc, C, 1]
            }
            dw = ggml_reshape_2d(ctx, dw, Tc, D);                 // [Tc, C]
            ggml_tensor* dwb = clone_weight_opt(ctx, ml, pre + "conv.depthwise_conv.bias");
            ggml_tensor* dwt = ggml_cont(ctx, ggml_transpose(ctx, dw)); // [C, Tc]
            if (dwb) dwt = ggml_add(ctx, dwt, dwb);

            ggml_tensor* normed;
            if (cnt == "layer_norm") {
                ggml_tensor* g = clone_weight(ctx, ml, pre + "conv.batch_norm.weight");
                ggml_tensor* bb = clone_weight(ctx, ml, pre + "conv.batch_norm.bias");
                normed = ggml_norm(ctx, dwt, ln_eps);
                normed = ggml_mul(ctx, normed, g);
                normed = ggml_add(ctx, normed, bb);
            } else {
                sc_host.assign(D, 0.0f);
                sh_host.assign(D, 0.0f);
                const float* g = ggml_get_data_f32(ml.tensor(pre + "conv.batch_norm.weight"));
                const float* bb = ggml_get_data_f32(ml.tensor(pre + "conv.batch_norm.bias"));
                const float* mm = ggml_get_data_f32(ml.tensor(pre + "conv.batch_norm.running_mean"));
                const float* var = ggml_get_data_f32(ml.tensor(pre + "conv.batch_norm.running_var"));
                for (int cc2 = 0; cc2 < D; ++cc2) {
                    sc_host[cc2] = g[cc2] / std::sqrt(var[cc2] + 1e-5f);
                    sh_host[cc2] = bb[cc2] - mm[cc2] * sc_host[cc2];
                }
                int64_t d_ne[1] = {D};
                ggml_tensor* sc_t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1,
                                        d_ne, sc_host.data(), sc_host.size() * sizeof(float));
                ggml_tensor* sh_t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1,
                                        d_ne, sh_host.data(), sh_host.size() * sizeof(float));
                normed = ggml_add(ctx, ggml_mul(ctx, dwt, sc_t), sh_t);
            }
            normed = ggml_silu(ctx, normed);
            ggml_tensor* pw2w = clone_weight(ctx, ml, pre + "conv.pointwise_conv2.weight");
            pw2w = ggml_reshape_2d(ctx, pw2w, D, D);
            ggml_tensor* pw2b = clone_weight_opt(ctx, ml, pre + "conv.pointwise_conv2.bias");
            ggml_tensor* cout = ggml_mul_mat(ctx, pw2w, normed); // [D, Tc]
            if (pw2b) cout = ggml_add(ctx, cout, pw2b);

            // pack [conv_out(Tc) ; glu(Tc)] along time -> [D, 2Tc].
            return ggml_concat(ctx, cout, glu, /*dim=*/1);
        }, packed);
        assert(ok && "stream conv graph failed"); (void)ok;
    }
    // packed is row-major [2Tc, D]: first Tc rows = conv_out, next Tc rows = glu.
    // r = r + conv_out.
    for (int t = 0; t < Tc; ++t)
        for (int c = 0; c < D; ++c)
            r[(size_t)t * D + c] += packed[(size_t)t * D + c];

    // Update conv cache: last LP columns of [cache_time(LP) ; glu(Tc)] (NeMo
    // new_x[:, :, -LP:]). glu is rows [Tc, 2Tc) of `packed` (row-major [Tc, D]).
    {
        std::vector<float> nc((size_t)LP * D, 0.0f);
        // logical sequence of LP+Tc frames: old_cache[0..LP) then glu[0..Tc).
        // take the last LP of these.
        const float* old_cache = cache_time_[layer_idx].data();
        const float* glu = packed.data() + (size_t)Tc * D; // [Tc, D]
        for (int j = 0; j < LP; ++j) {
            const int src = (LP + Tc - LP) + j; // index into the LP+Tc sequence
            if (src < LP) {
                std::memcpy(nc.data() + (size_t)j * D, old_cache + (size_t)src * D,
                            (size_t)D * sizeof(float));
            } else {
                std::memcpy(nc.data() + (size_t)j * D, glu + (size_t)(src - LP) * D,
                            (size_t)D * sizeof(float));
            }
        }
        cache_time_[layer_idx].swap(nc);
    }

    // === Stage D: r = r + 0.5*FFN2(norm_ff2(r)); out = norm_out(r). ===
    std::vector<float> out;
    {
        bool ok = pk::run_graph(mem_bytes, 4, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t rt_ne[2] = {D, Tc};
            ggml_tensor* rt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, rt_ne,
                                  r.data(), (size_t)Tc * D * sizeof(float));
            ggml_tensor* h = layer_norm(ctx, rt, "norm_feed_forward2");
            h = feed_forward(ctx, h, "feed_forward2");
            h = ggml_scale(ctx, h, 0.5f);
            ggml_tensor* res = ggml_add(ctx, rt, h);
            return layer_norm(ctx, res, "norm_out");
        }, out);
        assert(ok && "stream FFN2/out graph failed"); (void)ok;
    }
    return out;
}

std::vector<float> StreamingEncoder::step(const std::vector<float>& mel_chunk_frames,
                                          int n_mel_frames, bool is_last,
                                          int& n_valid_out) {
    const int D = d_model_;
    const int n_mels = (int)ml_.config().n_mels;
    assert((int)mel_chunk_frames.size() == n_mels * n_mel_frames);

    // ---- 1. Subsampling on the mel chunk window (already includes pre-encode
    //         cache overlap). The WHOLE window is real audio (no preprocessor
    //         center-pad), so the entry valid length is n_mel_frames (NOT T-1):
    //         pass in_valid_frames = n_mel_frames. Output [Tsub, d_model] + valid.
    Subsampling sub(ml_);
    std::vector<float> sx;
    int Tsub = 0, dm = 0, sub_valid = 0;
    sub.forward(mel_chunk_frames, n_mels, n_mel_frames, sx, Tsub, dm, sub_valid,
                /*in_valid_frames*/n_mel_frames);
    assert(dm == D);

    // ---- 2. Drop drop_extra_pre_encoded leading subsampled frames (NeMo
    //         forward_internal 636-638; only when cache is present == every step
    //         after the first; step 0 uses drop=0). ----
    const int drop = (step_ != 0) ? drop_extra_ : 0;
    int Tc = Tsub - drop;
    if (Tc < 0) Tc = 0;
    std::vector<float> x;
    if (drop > 0 && Tc > 0) {
        x.assign(sx.begin() + (size_t)drop * D, sx.end());
    } else if (drop == 0) {
        x = std::move(sx);
    } else {
        x.clear();
    }
    assert((int)x.size() == Tc * D);

    if (Tc == 0) {     // no output this step (e.g. a too-short tail chunk)
        n_valid_out = 0;
        step_ += 1;
        return {};
    }

    // ---- 3. xscaling (gated; off for this model). ----
    if (xscaling_) {
        const float xs = std::sqrt((float)D);
        for (float& v : x) v *= xs;
    }

    // ---- 4. Positional encoding. NeMo pos_enc(cache_len): input_len = Tc +
    //         cache_len; pos_emb spans 2*input_len-1 positions, == rel_pos_encoding
    //         for length input_len. cache_len = last_channel_cache (70). ----
    const int cache_len = last_channel_cache_;
    const int Pn = Tc + cache_len;
    std::vector<float> pos_emb;
    rel_pos_encoding(Pn, D, pos_emb);
    const int pos_len = 2 * Pn - 1;

    // ---- 5. Conformer layer stack (with conv + attention caches). ----
    for (int i = 0; i < n_layers_; ++i) {
        std::vector<float> next = layer_step(i, x, Tc, pos_emb, pos_len, cache_len);
        x.swap(next);
    }
    // After the layer stack the attention cache_last_channel_len has grown by Tc.
    clc_len_ = std::min(cache_len, clc_len_ + Tc);
    step_ += 1;

    // ---- 6. Slice valid output. keep_all_outputs == is_last: mid-stream keep
    //         only valid_out_len frames; final chunk keep all Tc frames. ----
    int valid = is_last ? Tc : std::min(valid_out_len_, Tc);
    n_valid_out = valid;
    std::vector<float> out((size_t)valid * D);
    std::memcpy(out.data(), x.data(), (size_t)valid * D * sizeof(float));
    return out;
}

} // namespace pk
