#include "joint.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"
#include <cassert>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace pk {

namespace {

// Copy a weight tensor from the model loader into the compute context,
// preserving its type and raw bytes. Supports 1-D and 2-D weights; returns a
// contiguous ggml tensor. The joint enc/pred PROJECTION weights are on the
// converter's quantization allowlist and may be stored f16/q8_0 -- they are fed
// straight into ggml_mul_mat below, which dequantizes src0 on the fly. (The
// joint OUTPUT projection joint.joint_net.2.weight is read as a raw float* in
// forward() and is intentionally kept F32 by the converter.)
static ggml_tensor* clone_w(ggml_context* ctx, const ModelLoader& ml, const char* name) {
    ggml_tensor* src = ml.tensor(name);
    assert(src && "missing joint tensor");
    const int nd = ggml_n_dims(src);
    int64_t ne[4] = {1, 1, 1, 1};
    for (int i = 0; i < nd; ++i) ne[i] = src->ne[i];
    ggml_tensor* dst = ggml_new_tensor(ctx, src->type, nd, ne);
    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    return dst;
}

} // namespace

Joint::Joint(const ModelLoader& ml) : ml_(ml) {
    // Read joint_hidden from the enc weight shape: ne[1] (ggml) = joint_hidden.
    ggml_tensor* ew = ml.tensor("joint.enc.weight");
    assert(ew && "missing joint.enc.weight");
    joint_hidden_ = (int)ew->ne[1];

    vocab_size_    = (int)ml.config().vocab_size;
    num_durations_ = (int)ml.config().tdt_durations.size();
    V_plus_        = vocab_size_ + 1 + num_durations_;
}

void Joint::forward(const std::vector<float>& enc,  int T, int enc_hidden,
                    const std::vector<float>& pred, int U, int pred_hidden,
                    std::vector<float>& logits, int& V_plus_out) const {
    assert((int)enc.size()  == T * enc_hidden);
    assert((int)pred.size() == U * pred_hidden);

    const int H = joint_hidden_;
    const int V = V_plus_;
    V_plus_out = V;

    // ---- Step 1: enc_proj[T, H] = enc[T, E] · enc_weight[E, H]^T + enc_bias ----
    // Computed via ggml: enc input ne[0]=enc_hidden, ne[1]=T (row-major [T, E]),
    // weight has ne[0]=enc_hidden, ne[1]=H  →  ggml_mul_mat gives [H, T].
    std::vector<float> enc_proj;
    {
        const size_t mem = (size_t)64 * 1024 * 1024 +
                           (size_t)(T * enc_hidden + enc_hidden * H + T * H + H) * sizeof(float) * 4;
        bool ok = pk::run_graph(mem, 4,
            [&](ggml_context* ctx) -> ggml_tensor* {
                // Input: row-major [T, E], ggml ne[0]=enc_hidden (fastest), ne[1]=T
                ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, enc_hidden, T);
                std::memcpy(x->data, enc.data(), (size_t)T * enc_hidden * sizeof(float));

                // Weight: ggml ne[0]=enc_hidden, ne[1]=H
                ggml_tensor* W = clone_w(ctx, ml_, "joint.enc.weight");
                // W ne[0]=enc_hidden, ne[1]=H; x ne[0]=enc_hidden → mul_mat gives [H, T]
                ggml_tensor* y = ggml_mul_mat(ctx, W, x);

                // Bias: [H]
                ggml_tensor* b = clone_w(ctx, ml_, "joint.enc.bias");
                y = ggml_add(ctx, y, b);
                return y;
            }, enc_proj);
        assert(ok && "enc_proj graph failed");
        // enc_proj is now [H, T] ggml → memory enc_proj[t*H + h]  (ne[0]=H fastest)
    }

    // ---- Step 2: pred_proj[U, H] = pred[U, P] · pred_weight[P, H]^T + pred_bias ----
    std::vector<float> pred_proj;
    {
        const size_t mem = (size_t)64 * 1024 * 1024 +
                           (size_t)(U * pred_hidden + pred_hidden * H + U * H + H) * sizeof(float) * 4;
        bool ok = pk::run_graph(mem, 4,
            [&](ggml_context* ctx) -> ggml_tensor* {
                // Input: row-major [U, P], ggml ne[0]=pred_hidden, ne[1]=U
                ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pred_hidden, U);
                std::memcpy(x->data, pred.data(), (size_t)U * pred_hidden * sizeof(float));

                // Weight: ggml ne[0]=pred_hidden, ne[1]=H
                ggml_tensor* W = clone_w(ctx, ml_, "joint.pred.weight");
                ggml_tensor* y = ggml_mul_mat(ctx, W, x);

                // Bias: [H]
                ggml_tensor* b = clone_w(ctx, ml_, "joint.pred.bias");
                y = ggml_add(ctx, y, b);
                return y;
            }, pred_proj);
        assert(ok && "pred_proj graph failed");
        // pred_proj is now [H, U] ggml → memory pred_proj[u*H + h]
    }

    // ---- Step 3: f[t,u] = ReLU(enc_proj[t] + pred_proj[u]) ----
    // Then logits[t,u] = out_weight · f[t,u] + out_bias
    // Allocate f[T, U, H] and logits[T, U, V].
    // Both enc_proj and pred_proj have memory layout [rows, H] so:
    //   enc_proj[t*H + h]   for t in [0..T)
    //   pred_proj[u*H + h]  for u in [0..U)

    const float* ep = enc_proj.data();
    const float* pp = pred_proj.data();

    // Load output weight and bias directly (no ggml graph needed for the final matmul
    // since we loop over T*U anyway).
    ggml_tensor* wout_src = ml_.tensor("joint.joint_net.2.weight");
    ggml_tensor* bout_src = ml_.tensor("joint.joint_net.2.bias");
    assert(wout_src && bout_src && "missing joint_net.2 weight/bias");
    assert(wout_src->type == GGML_TYPE_F32 && bout_src->type == GGML_TYPE_F32);
    // wout: ggml ne[0]=H (joint_hidden), ne[1]=V → memory W[v*H + h]
    const float* Wout = (const float*)wout_src->data;
    const float* bout = (const float*)bout_src->data;

    logits.resize((size_t)T * U * V);
    std::vector<float> f(H);

    for (int t = 0; t < T; ++t) {
        const float* ep_t = ep + (size_t)t * H;   // enc_proj for frame t: [H]
        for (int u = 0; u < U; ++u) {
            const float* pp_u = pp + (size_t)u * H;  // pred_proj for step u: [H]

            // f[h] = ReLU(enc_proj[t][h] + pred_proj[u][h])
            for (int h = 0; h < H; ++h) {
                float v = ep_t[h] + pp_u[h];
                f[h] = v > 0.0f ? v : 0.0f;
            }

            // logits[t,u,v] = sum_h Wout[v,h] * f[h] + bout[v]
            // Wout memory: W[v*H + h]
            float* out_tuv = logits.data() + ((size_t)t * U + u) * V;
            for (int v = 0; v < V; ++v) {
                const float* row = Wout + (size_t)v * H;
                float acc = bout[v];
                for (int h = 0; h < H; ++h) acc += row[h] * f[h];
                out_tuv[v] = acc;
            }
        }
    }
}

} // namespace pk
