#include "ctc_decoder.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"
#include <cassert>
#include <cstring>
#include <string>

namespace pk {

CTCDecoder::CTCDecoder(const ModelLoader& ml) : ml_(ml) {
    d_model_     = (int)ml.config().d_model;
    vocab_plus_1_ = (int)ml.config().vocab_size + 1;
}

void CTCDecoder::forward(const std::vector<float>& enc, int d_model, int T,
                         std::vector<float>& logits, int& vocab_plus_1) const {
    assert(d_model == d_model_);
    assert((int)enc.size() == d_model * T);

    const int V = vocab_plus_1_;
    vocab_plus_1 = V;

    // Memory budget: weight [d_model, V], input [d_model, T], output [V, T].
    const size_t mem_bytes =
        (size_t)64 * 1024 * 1024 +
        (size_t)(d_model * V + d_model * T + V * T + V) * sizeof(float) * 4;

    const ModelLoader& ml = ml_;

    bool ok = pk::run_graph(mem_bytes, /*n_threads=*/4,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // ---- Input: enc [d_model, T] row-major, enc[c*T + t] ----
            // ggml ne[0]=d_model (fastest), ne[1]=T — so data layout is enc[t*d_model + c]
            // BUT our enc is row-major [d_model, T] = enc[c*T + t], i.e. T is fastest.
            // To match ggml layout where ne[0] is fastest, we store enc with ne[0]=T, ne[1]=d_model:
            //   ggml tensor: ne[0]=T, ne[1]=d_model → memory enc[t + T*d] = enc[d*T + t] ✓
            ggml_tensor* xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, d_model);
            std::memcpy(xt->data, enc.data(), (size_t)d_model * T * sizeof(float));
            // xt: ne[0]=T (fastest), ne[1]=d_model → memory enc[d*T + t] ✓

            // ---- Weight: GGUF stores ne[0]=1, ne[1]=d_model, ne[2]=V ----
            // Squeeze k=1 dim → reshape to ne[0]=d_model, ne[1]=V.
            // Memory layout of weight: w[k + 1*c + d_model*v] = w[c + d_model*v]
            // So d_model (ne[0]) is fastest → correct for ggml_mul_mat contraction on ne[0].
            ggml_tensor* w3 = ml.tensor("ctc_decoder.decoder_layers.0.weight");
            assert(w3 && "missing ctc_decoder weight");
            assert(w3->type == GGML_TYPE_F32);
            // Clone weight into compute context preserving ne[] layout.
            const int nd_w = ggml_n_dims(w3);
            int64_t ne_w[4] = {1, 1, 1, 1};
            for (int i = 0; i < nd_w; ++i) ne_w[i] = w3->ne[i];
            ggml_tensor* wclone = ggml_new_tensor(ctx, GGML_TYPE_F32, nd_w, ne_w);
            std::memcpy(wclone->data, w3->data, ggml_nbytes(w3));
            // Reshape from [ne0=1, ne1=d_model, ne2=V] to [ne0=d_model, ne1=V].
            // ggml_reshape_2d requires the tensor to be contiguous (it creates a view).
            ggml_tensor* W = ggml_reshape_2d(ctx, wclone, d_model, V);
            // W: ne[0]=d_model (fastest), ne[1]=V ✓

            // ---- Bias: [V] ----
            ggml_tensor* bsrc = ml.tensor("ctc_decoder.decoder_layers.0.bias");
            assert(bsrc && "missing ctc_decoder bias");
            assert(bsrc->type == GGML_TYPE_F32);
            ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, V);
            std::memcpy(b->data, bsrc->data, (size_t)V * sizeof(float));

            // ---- Linear: ggml_mul_mat(W, xt) ----
            // ggml_mul_mat(A, B): A[ne0(A), ne1(A)], B[ne0(B), ne1(B)]
            // requires ne0(A) == ne0(B), contracts on ne0, output [ne1(A), ne1(B)].
            // W: ne0=d_model, ne1=V  (but we have xt: ne0=T, ne1=d_model)
            // W ne0=d_model ≠ xt ne0=T → need to transpose xt so ne0=d_model.
            //
            // Transpose xt: ggml_transpose gives a view with ne[0] and ne[1] swapped.
            // xt transposed: ne[0]=d_model, ne[1]=T  (now matches W ne0=d_model).
            ggml_tensor* xt_t = ggml_cont(ctx, ggml_transpose(ctx, xt));
            // xt_t: ne[0]=d_model (fastest after cont), ne[1]=T
            // ggml_mul_mat(W, xt_t): W[d_model, V], xt_t[d_model, T] → output [V, T]
            ggml_tensor* y = ggml_mul_mat(ctx, W, xt_t);
            // y: ne[0]=V, ne[1]=T → memory y[t*V + v] ✓

            // ---- Bias add: broadcast b[V] over T ----
            y = ggml_add(ctx, y, b);

            // ---- Log-softmax over vocab axis (ne[0]=V) ----
            // ggml_soft_max operates over ne[0] (the innermost dimension).
            // Our y has ne[0]=V, so softmax is over vocab ✓.
            ggml_tensor* sm  = ggml_soft_max(ctx, y);
            ggml_tensor* lsm = ggml_log(ctx, sm);
            // lsm: ne[0]=V, ne[1]=T → memory lsm[t*V + v] ✓
            return lsm;
        },
        logits);

    assert(ok && "ctc_decoder graph failed");
    (void)ok;
}

} // namespace pk
