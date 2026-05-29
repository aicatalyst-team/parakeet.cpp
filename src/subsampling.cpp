#include "subsampling.hpp"
#include "ggml_graph.hpp"
#include "backend.hpp"
#include "ggml.h"
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>

namespace pk {

// Weights from the GGUF (loader context) are brought into the graph as inputs
// via the shared pk::clone_weight (backend.cpp): a same-type/same-ne input
// tensor whose loader bytes are copied in AFTER the gallocr allocates it. The
// conv kernels stay F32 (the converter never quantizes them); only out.weight
// is allowlisted and may be f16/q8_0, fed into ggml_mul_mat which dequantizes
// src0. GGUF ne is reverse of the torch shape == ggml's [KW,KH,IC,OC] layout.

Subsampling::Subsampling(const ModelLoader& ml)
    : ml_(ml) {
    conv_channels_ = (int)ml.config().subsampling_conv_channels;
    d_model_       = (int)ml.config().d_model;
    causal_        = ml.config().causal_downsampling;
}

int Subsampling::valid_out_len(int T, int in_valid_frames) const {
    // The mel has T spatial frames, but the OFFLINE preprocessor reports a valid
    // length of T-1 (center-padding adds one extra trailing frame). Each of the
    // three stride-2, k=3 conv stages reduces the valid length via NeMo's
    // calc_length: out = floor((in + all_paddings - k)/s) + 1, all_paddings =
    // left+right.
    //
    // Non-causal (offline): symmetric pad (k-1)/2 each side -> all_paddings = 2.
    //   out = floor((in + 2 - 3)/2) + 1 = (in - 1)/2 + 1  (matches existing path).
    // Causal (causal_downsampling=True): left = k-1 = 2, right = stride-1 = 1 ->
    //   all_paddings = 3.  out = floor((in + 3 - 3)/2) + 1 = floor(in/2) + 1.
    // NeMo's calc_length runs in float; for these integer inputs floor matches
    // integer division, so we use integer arithmetic directly.
    //
    // Streaming (in_valid_frames >= 0): the chunk window is fully real audio, so
    // the entry valid length is the supplied count (typically T), NOT T-1.
    const int all_paddings = causal_ ? 3 : 2;
    int valid = (in_valid_frames >= 0) ? in_valid_frames : (T - 1);
    for (int st = 0; st < 3; ++st)            // conv0, conv2, conv5
        valid = (valid + all_paddings - 3) / 2 + 1;
    return valid;
}

void Subsampling::forward(const std::vector<float>& mel, int n_mels, int T,
                          std::vector<float>& out, int& Tout, int& d_model) const {
    int valid_len_unused = 0;
    forward(mel, n_mels, T, out, Tout, d_model, valid_len_unused, -1);
}

void Subsampling::forward(const std::vector<float>& mel, int n_mels, int T,
                          std::vector<float>& out, int& Tout, int& d_model,
                          int& valid_len) const {
    forward(mel, n_mels, T, out, Tout, d_model, valid_len, -1);
}

void Subsampling::forward(const std::vector<float>& mel, int n_mels, int T,
                          std::vector<float>& out, int& Tout, int& d_model,
                          int& valid_len, int in_valid_frames) const {
    const int C = conv_channels_;
    const int F = n_mels;            // feature dim (80)

    // Generous compute budget: a few large intermediate tensors (im2col of the
    // first conv dominates). Scales with T and C.
    const size_t mem_bytes =
        (size_t)256 * 1024 * 1024 +
        (size_t)C * (size_t)T * (size_t)F * 64 * sizeof(float);

    const ModelLoader& ml = ml_;

    // --- Input (host-side): ggml conv data layout is [W=feat, H=T, IC=1, N=1].
    // NeMo conv input is [B,1,T,feat] (H=T, W=feat). We must feed
    // x[t*F + f] = mel(feat=f, time=t). mel is feat-major [F,T] (mel[m*T + t])
    // so transpose into time-major here, then feed it as a graph input.
    std::vector<float> x_host((size_t)F * T);
    for (int t = 0; t < T; ++t)
        for (int f = 0; f < F; ++f)
            x_host[(size_t)t * F + f] = mel[(size_t)f * T + t];

    // Host-built masks (filled inside the build lambda if needed; kept alive
    // for the whole call so the deferred input copies stay valid). Use stable
    // storage (pointers) since several time masks may be registered at once.
    std::vector<std::unique_ptr<std::vector<float>>> mask_hosts;
    std::vector<float> outmask_host;

    bool ok = pk::run_graph(mem_bytes, /*n_threads*/4,
        [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t x_ne[4] = {F, T, 1, 1};
            ggml_tensor* x = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 4, x_ne,
                                                    x_host.data(),
                                                    x_host.size() * sizeof(float));

            // Subsampling conv padding. NeMo dw_striding uses k=3, s=2 on each
            // stage; the padding differs by model:
            //   non-causal (offline): symmetric (k-1)/2 = 1 on every side, applied
            //     directly via the conv's p0/p1 (byte-identical to the old path).
            //   causal (causal_downsampling=True, e.g. parakeet_realtime_eou_120m):
            //     NeMo CausalConv2D pads BOTH spatial axes (time H and feature W)
            //     with left = k-1 = 2, right = stride-1 = 1 (F.pad order
            //     (W_l, W_r, H_l, H_r)). ggml conv takes a single symmetric p per
            //     axis, so for the causal case we pad explicitly with ggml_pad_ext
            //     (lp0/rp0 = W=feature, lp1/rp1 = H=time) and run the conv with p=0.
            const bool causal = causal_;
            auto pad_causal = [&](ggml_tensor* t) -> ggml_tensor* {
                // ne0 = W (feature), ne1 = H (time); left k-1=2, right s-1=1 on both.
                return ggml_pad_ext(ctx, t, /*lp0*/2, /*rp0*/1, /*lp1*/2, /*rp1*/1,
                                    0, 0, 0, 0);
            };

            // NeMo's MaskedConvSequential zeros the trailing (pad) time frames of
            // the conv input BEFORE every stage, tracking the valid length via
            // calc_length. For the SYMMETRIC (offline) path a valid output frame
            // never reaches a masked input frame (the kernel is centred), so the
            // old code masks only the flattened output and stays byte-identical —
            // keep that. For the CAUSAL path the right pad is +1, so the last valid
            // output frame DOES read the trailing pad input frame; we must replicate
            // the per-stage input masking. Mask the time axis (ne1=H) to `valid_t`
            // frames via a [1, H, 1, 1] mask broadcast over W (feat) and C.
            auto mask_time = [&](ggml_tensor* t, int valid_t) -> ggml_tensor* {
                const int H = (int)t->ne[1];
                if (valid_t >= H) return t;
                mask_hosts.emplace_back(new std::vector<float>(H));
                std::vector<float>& md = *mask_hosts.back();
                for (int h = 0; h < H; ++h) md[h] = (h < valid_t) ? 1.0f : 0.0f;
                int64_t m_ne[4] = {1, H, 1, 1};
                ggml_tensor* tm = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 4,
                                      m_ne, md.data(), md.size() * sizeof(float));
                return ggml_mul(ctx, t, tm); // broadcast over ne0(W), ne2(C), ne3
            };
            // Per-stage valid time lengths (NeMo calc_length on the time axis).
            // Stage entry length: offline = T-1 (preprocessor center-pad valid mel
            // len); streaming = in_valid_frames (whole chunk window is real audio).
            // all_paddings = 3 (causal).
            int valid_t0 = (in_valid_frames >= 0) ? in_valid_frames : (T - 1); // before stage 0
            int valid_t1 = (valid_t0 + 3 - 3) / 2 + 1;  // before stage 2 (after stage 0)
            int valid_t2 = (valid_t1 + 3 - 3) / 2 + 1;  // before stage 5 (after stage 2)

            // ---- Stage 1: full Conv2d(1 -> C, k=3, s=2) + ReLU ----
            // kernel conv.0.weight: torch [C,1,3,3] -> ggml ne [3,3,1,C] = [KW,KH,IC,OC].
            ggml_tensor* w0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.weight");
            ggml_tensor* b0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.bias");
            if (causal) {
                x = mask_time(x, valid_t0);   // zero trailing pad mel frames
                x = pad_causal(x);
                x = ggml_conv_2d(ctx, w0, x, /*s0*/2, /*s1*/2, /*p0*/0, /*p1*/0, /*d0*/1, /*d1*/1);
            } else {
                x = ggml_conv_2d(ctx, w0, x, /*s0*/2, /*s1*/2, /*p0*/1, /*p1*/1, /*d0*/1, /*d1*/1);
            }
            // x: ne [OW=F/2, OH=T/2, OC=C, 1]. Add bias broadcast over channels:
            // reshape bias to [1,1,C,1] so it broadcasts across W,H.
            x = ggml_add(ctx, x, ggml_reshape_4d(ctx, b0, 1, 1, C, 1));
            x = ggml_relu(ctx, x);

            // ---- Stages 2 & 3: depthwise(k=3,s=2,p=1,groups=C) + pointwise(k=1) + ReLU ----
            struct StageW { const char* dw_w; const char* dw_b; const char* pw_w; const char* pw_b; };
            const StageW stages[2] = {
                { "encoder.pre_encode.conv.2.weight", "encoder.pre_encode.conv.2.bias",
                  "encoder.pre_encode.conv.3.weight", "encoder.pre_encode.conv.3.bias" },
                { "encoder.pre_encode.conv.5.weight", "encoder.pre_encode.conv.5.bias",
                  "encoder.pre_encode.conv.6.weight", "encoder.pre_encode.conv.6.bias" },
            };
            int stage_valid_t[2] = {valid_t1, valid_t2};
            for (int si = 0; si < 2; ++si) {
                const StageW& s = stages[si];
                // Depthwise: weight torch [C,1,3,3] -> ggml ne [3,3,1,C] = [KW,KH,1,C].
                // ggml_conv_2d_dw_direct expects a:[KW,KH,1,C], b:[W,H,C,N].
                ggml_tensor* dww = clone_weight(ctx, ml, s.dw_w);
                ggml_tensor* dwb = clone_weight(ctx, ml, s.dw_b);
                if (causal) {
                    x = mask_time(x, stage_valid_t[si]); // zero trailing pad time frames
                    x = pad_causal(x);
                    x = ggml_conv_2d_dw_direct(ctx, dww, x, /*s0*/2, /*s1*/2, /*p0*/0, /*p1*/0, /*d0*/1, /*d1*/1);
                } else {
                    x = ggml_conv_2d_dw_direct(ctx, dww, x, /*s0*/2, /*s1*/2, /*p0*/1, /*p1*/1, /*d0*/1, /*d1*/1);
                }
                // x: ne [OW, OH, C, 1]. dw_direct keeps WHCN; make it contiguous so
                // the bias add and following ops see a standard layout.
                x = ggml_cont(ctx, x);
                x = ggml_add(ctx, x, ggml_reshape_4d(ctx, dwb, 1, 1, C, 1));

                // Pointwise: weight torch [C,C,1,1] -> ggml ne [1,1,C,C] = [KW,KH,IC,OC].
                ggml_tensor* pww = clone_weight(ctx, ml, s.pw_w);
                ggml_tensor* pwb = clone_weight(ctx, ml, s.pw_b);
                x = ggml_conv_2d(ctx, pww, x, /*s0*/1, /*s1*/1, /*p0*/0, /*p1*/0, /*d0*/1, /*d1*/1);
                x = ggml_add(ctx, x, ggml_reshape_4d(ctx, pwb, 1, 1, C, 1));
                x = ggml_relu(ctx, x);
            }

            // x: ne [F'=OW, T'=OH, C, 1]. NeMo flatten:
            //   [B,C,T',F'].transpose(1,2).reshape(B,T',C*F')
            // -> per time t, vector is channel-major: idx = c*F' + f.
            const int Fp = (int)x->ne[0]; // F'
            const int Tp = (int)x->ne[1]; // T'
            // Want contiguous [F', C, T', 1] so flat = t*(C*F') + c*F' + f.
            // current dims (0,1,2,3) = (F', T', C, 1); permute to (F', C, T', 1):
            // place src dim0->0, src dim2(C)->1, src dim1(T')->2, src dim3->3.
            ggml_tensor* xp = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
            // xp ne = [F', C, T', 1]; flatten inner two dims -> [C*F', T'].
            ggml_tensor* flat = ggml_reshape_2d(ctx, xp, (int64_t)C * Fp, Tp);

            // --- Length masking (faithful to NeMo MaskedConvSequential) ---
            // The mel has T spatial frames, but the preprocessor reports a valid
            // length of T-1 (center-padding adds one extra trailing frame). The
            // conv stack masks trailing frames beyond the valid length to zero, so
            // those output rows reduce to the Linear's bias. Valid output frames
            // never read masked input frames (kernel reach stays inside the valid
            // region), so we can run the conv stack spatially and zero the
            // flattened conv output at frames >= valid_out_len before the Linear.
            const int valid_out = valid_out_len(T, in_valid_frames);
            if (valid_out < Tp) {
                // Multiply by a time mask [1, T'] (1.0 valid, 0.0 masked), which
                // broadcasts over the C*F' feature dim. Must be applied in-graph
                // (flat is computed at execute time, not build time).
                outmask_host.assign(Tp, 0.0f);
                for (int t = 0; t < Tp; ++t)
                    outmask_host[t] = (t < valid_out) ? 1.0f : 0.0f;
                int64_t mk_ne[2] = {1, Tp};
                ggml_tensor* mask = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2,
                                        mk_ne, outmask_host.data(),
                                        outmask_host.size() * sizeof(float));
                flat = ggml_mul(ctx, flat, mask);
            }

            // ---- Linear out: torch [d_model, C*F'] -> ggml ne [C*F', d_model]. ----
            ggml_tensor* ow = clone_weight(ctx, ml, "encoder.pre_encode.out.weight");
            ggml_tensor* ob = clone_weight(ctx, ml, "encoder.pre_encode.out.bias");
            // mul_mat(W[C*F', d_model], flat[C*F', T']) -> [d_model, T'].
            ggml_tensor* y = ggml_mul_mat(ctx, ow, flat);
            y = ggml_add(ctx, y, ob); // broadcast bias [d_model] over T'
            // y: ne [d_model, T'] contiguous -> row-major [T', d_model]. Matches baseline.
            return y;
        },
        out);

    assert(ok && "subsampling graph failed");
    (void)ok;

    // Output geometry: T' from conv reductions, d_model from config.
    Tout = (int)out.size() / d_model_;
    d_model = d_model_;
    valid_len = valid_out_len(T, in_valid_frames);
    if (valid_len > Tout) valid_len = Tout;
}

} // namespace pk
