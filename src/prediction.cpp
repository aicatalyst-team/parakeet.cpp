#include "prediction.hpp"
#include "ggml.h"
#include <cassert>
#include <cstring>
#include <cmath>
#include <string>

namespace pk {

namespace {
inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Fetch an f32 tensor from the loader; assert it exists and is f32.
const float* tensor_data(const ModelLoader& ml, const char* name) {
    ggml_tensor* t = ml.tensor(name);
    assert(t && "missing prediction tensor");
    assert(t->type == GGML_TYPE_F32 && "prediction tensor not f32");
    return (const float*)t->data;
}
} // namespace

PredictionNet::PredictionNet(const ModelLoader& ml) : ml_(ml) {
    H_        = (int)ml.config().pred_hidden;
    vocab_p1_ = (int)ml.config().vocab_size + 1;
    assert(H_ > 0 && "pred_hidden not set");
}

void PredictionNet::forward(const std::vector<int32_t>& ids, bool add_sos,
                            std::vector<float>& out, int& U_out, int& hidden) const {
    const int H  = H_;
    const int H4 = 4 * H;

    // ---- Weights (read directly; layout is row-major in the 4H/output dim) ----
    // embed.weight  ggml ne=[H, vocab+1]   → memory embed[id*H + h]
    // weight_ih_l0  ggml ne=[H, 4H]        → memory W_ih[out*H + in]  (out in 4H)
    // weight_hh_l0  ggml ne=[H, 4H]        → memory W_hh[out*H + in]
    // bias_ih_l0    ggml ne=[4H]
    // bias_hh_l0    ggml ne=[4H]
    const float* embed = tensor_data(ml_, "decoder.prediction.embed.weight");
    const float* W_ih  = tensor_data(ml_, "decoder.prediction.dec_rnn.lstm.weight_ih_l0");
    const float* W_hh  = tensor_data(ml_, "decoder.prediction.dec_rnn.lstm.weight_hh_l0");
    const float* b_ih  = tensor_data(ml_, "decoder.prediction.dec_rnn.lstm.bias_ih_l0");
    const float* b_hh  = tensor_data(ml_, "decoder.prediction.dec_rnn.lstm.bias_hh_l0");

    // ---- Build the embedded input sequence ----
    // With add_sos, prepend a literal zero [H] vector (NeMo's SOS), then the
    // embeddings of each id. embed[id] is H contiguous floats at offset id*H.
    const int U   = (int)ids.size();
    const int seq = add_sos ? U + 1 : U;
    U_out  = seq;
    hidden = H;

    // X: row-major [seq, H], X[step*H + h]
    std::vector<float> X((size_t)seq * H, 0.0f);
    int base = 0;
    if (add_sos) {
        // step 0 is the zero SOS vector → already zeros.
        base = 1;
    }
    for (int u = 0; u < U; ++u) {
        const int32_t id = ids[u];
        assert(id >= 0 && id < vocab_p1_ && "embedding id out of range");
        std::memcpy(&X[(size_t)(base + u) * H], &embed[(size_t)id * H],
                    (size_t)H * sizeof(float));
    }

    // ---- LSTM recurrence (PyTorch LSTMCell, gate order [i, f, g, o]) ----
    // Precompute per step the input contribution Wx[step] = W_ih · x_t + b_ih,
    // then add the recurrent term W_hh · h + b_hh each step. h0 = c0 = 0.
    out.assign((size_t)seq * H, 0.0f);
    std::vector<float> h(H, 0.0f), c(H, 0.0f);
    std::vector<float> z(H4, 0.0f);

    for (int t = 0; t < seq; ++t) {
        const float* x = &X[(size_t)t * H];

        // z[out] = b_ih[out] + b_hh[out] + sum_in W_ih[out,in]*x[in]
        //                                + sum_in W_hh[out,in]*h[in]
        for (int o = 0; o < H4; ++o) {
            const float* wi = &W_ih[(size_t)o * H];
            const float* wh = &W_hh[(size_t)o * H];
            float acc_i = 0.0f, acc_h = 0.0f;
            for (int k = 0; k < H; ++k) {
                acc_i += wi[k] * x[k];
                acc_h += wh[k] * h[k];
            }
            z[o] = b_ih[o] + b_hh[o] + acc_i + acc_h;
        }

        // Gates: i = z[0:H], f = z[H:2H], g = z[2H:3H], o = z[3H:4H]
        // c' = f*c + i*g ;  h' = o*tanh(c')
        float* hout = &out[(size_t)t * H];
        for (int k = 0; k < H; ++k) {
            const float ig = sigmoidf(z[k]);          // input gate
            const float fg = sigmoidf(z[H + k]);      // forget gate
            const float gg = std::tanh(z[2 * H + k]); // cell candidate
            const float og = sigmoidf(z[3 * H + k]);  // output gate
            const float cn = fg * c[k] + ig * gg;
            const float hn = og * std::tanh(cn);
            c[k] = cn;
            h[k] = hn;
            hout[k] = hn;
        }
    }
}

} // namespace pk
