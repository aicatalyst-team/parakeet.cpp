#include "subsampling.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"
#include <cassert>
#include <cstring>
#include <vector>

namespace pk {

// Copy a tensor loaded from the GGUF (lives in the loader context) into a new
// tensor inside the compute context `ctx`, preserving its ne[] layout, type, and
// raw data. Weights in the GGUF have ggml ne = reverse of the torch shape, which
// is exactly the layout ggml's conv kernels expect ([KW,KH,IC,OC]). The conv
// kernels here stay F32 (the converter never quantizes them, and ggml_conv_2d
// has no quantized path); only the final out.weight projection is allowlisted
// and may be f16/q8_0, fed straight into ggml_mul_mat which dequantizes src0.
static ggml_tensor* clone_weight(ggml_context* ctx, const ModelLoader& ml,
                                 const char* name) {
    ggml_tensor* src = ml.tensor(name);
    assert(src && "missing tensor");
    const int nd = ggml_n_dims(src);
    int64_t ne[4] = {1,1,1,1};
    for (int i = 0; i < nd; ++i) ne[i] = src->ne[i];
    ggml_tensor* dst = ggml_new_tensor(ctx, src->type, nd, ne);
    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    return dst;
}

Subsampling::Subsampling(const ModelLoader& ml)
    : ml_(ml) {
    conv_channels_ = (int)ml.config().subsampling_conv_channels;
    d_model_       = (int)ml.config().d_model;
}

int Subsampling::valid_out_len(int T) const {
    // The mel has T spatial frames, but the preprocessor reports a valid length
    // of T-1 (center-padding adds one extra trailing frame). Each of the three
    // stride-2, k=3, p=1 conv stages reduces the valid length via NeMo's
    // calc_length: out = (in + 2*p - k)/s + 1.
    int valid = T - 1;
    for (int st = 0; st < 3; ++st)            // conv0, conv2, conv5
        valid = (valid + 2 - 3) / 2 + 1;
    return valid;
}

void Subsampling::forward(const std::vector<float>& mel, int n_mels, int T,
                          std::vector<float>& out, int& Tout, int& d_model) const {
    int valid_len_unused = 0;
    forward(mel, n_mels, T, out, Tout, d_model, valid_len_unused);
}

void Subsampling::forward(const std::vector<float>& mel, int n_mels, int T,
                          std::vector<float>& out, int& Tout, int& d_model,
                          int& valid_len) const {
    const int C = conv_channels_;
    const int F = n_mels;            // feature dim (80)

    // Generous compute budget: a few large intermediate tensors (im2col of the
    // first conv dominates). Scales with T and C.
    const size_t mem_bytes =
        (size_t)256 * 1024 * 1024 +
        (size_t)C * (size_t)T * (size_t)F * 64 * sizeof(float);

    const ModelLoader& ml = ml_;

    bool ok = pk::run_graph(mem_bytes, /*n_threads*/4,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // --- Input: ggml conv data layout is b: [W=feat, H=T, IC=1, N=1].
            // NeMo conv input is [B,1,T,feat] (H=T, W=feat). We must feed
            // x[t*F + f] = mel(feat=f, time=t). mel is feat-major [F,T]
            // (mel[m*T + t]) so transpose into time-major here.
            ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, F, T, 1, 1);
            {
                float* xd = (float*)x->data;
                for (int t = 0; t < T; ++t)
                    for (int f = 0; f < F; ++f)
                        xd[(size_t)t * F + f] = mel[(size_t)f * T + t];
            }

            // ---- Stage 1: full Conv2d(1 -> C, k=3, s=2, p=1) + ReLU ----
            // kernel conv.0.weight: torch [C,1,3,3] -> ggml ne [3,3,1,C] = [KW,KH,IC,OC].
            ggml_tensor* w0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.weight");
            ggml_tensor* b0 = clone_weight(ctx, ml, "encoder.pre_encode.conv.0.bias");
            x = ggml_conv_2d(ctx, w0, x, /*s0*/2, /*s1*/2, /*p0*/1, /*p1*/1, /*d0*/1, /*d1*/1);
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
            for (const StageW& s : stages) {
                // Depthwise: weight torch [C,1,3,3] -> ggml ne [3,3,1,C] = [KW,KH,1,C].
                // ggml_conv_2d_dw_direct expects a:[KW,KH,1,C], b:[W,H,C,N].
                ggml_tensor* dww = clone_weight(ctx, ml, s.dw_w);
                ggml_tensor* dwb = clone_weight(ctx, ml, s.dw_b);
                x = ggml_conv_2d_dw_direct(ctx, dww, x, /*s0*/2, /*s1*/2, /*p0*/1, /*p1*/1, /*d0*/1, /*d1*/1);
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
            const int valid_out = valid_out_len(T);
            if (valid_out < Tp) {
                // Multiply by a time mask [1, T'] (1.0 valid, 0.0 masked), which
                // broadcasts over the C*F' feature dim. Must be applied in-graph
                // (flat is computed at execute time, not build time).
                ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, Tp);
                float* md = (float*)mask->data;
                for (int t = 0; t < Tp; ++t) md[t] = (t < valid_out) ? 1.0f : 0.0f;
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
    valid_len = valid_out_len(T);
    if (valid_len > Tout) valid_len = Tout;
}

} // namespace pk
