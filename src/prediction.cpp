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
    n_layers_ = (int)ml.config().pred_rnn_layers;
    if (n_layers_ <= 0) n_layers_ = 1; // default to a single LSTM layer
    assert(H_ > 0 && "pred_hidden not set");
}

// ---------------------------------------------------------------------------
// Single LSTM cell step — shared by forward() and step().
// x[H]: current input embedding.
// h_in[H], c_in[H]: state coming in.
// h_out[H], c_out[H]: state written out.
// ---------------------------------------------------------------------------
void PredictionNet::lstm_cell(int layer, const float* x,
                               const float* h_in, const float* c_in,
                               float* h_out, float* c_out) const {
    const int H  = H_;
    const int H4 = 4 * H;

    const std::string sfx = "_l" + std::to_string(layer);
    const float* W_ih = tensor_data(ml_, ("decoder.prediction.dec_rnn.lstm.weight_ih" + sfx).c_str());
    const float* W_hh = tensor_data(ml_, ("decoder.prediction.dec_rnn.lstm.weight_hh" + sfx).c_str());
    const float* b_ih = tensor_data(ml_, ("decoder.prediction.dec_rnn.lstm.bias_ih" + sfx).c_str());
    const float* b_hh = tensor_data(ml_, ("decoder.prediction.dec_rnn.lstm.bias_hh" + sfx).c_str());

    // z[out] = b_ih[out] + b_hh[out]
    //        + sum_in W_ih[out, in] * x[in]
    //        + sum_in W_hh[out, in] * h_in[in]
    std::vector<float> z(H4);
    for (int o = 0; o < H4; ++o) {
        const float* wi = &W_ih[(size_t)o * H];
        const float* wh = &W_hh[(size_t)o * H];
        float acc_i = 0.0f, acc_h = 0.0f;
        for (int k = 0; k < H; ++k) {
            acc_i += wi[k] * x[k];
            acc_h += wh[k] * h_in[k];
        }
        z[o] = b_ih[o] + b_hh[o] + acc_i + acc_h;
    }

    // Gates: i = z[0:H], f = z[H:2H], g = z[2H:3H], o = z[3H:4H]
    // c' = f*c + i*g ;  h' = o*tanh(c')
    for (int k = 0; k < H; ++k) {
        const float ig = sigmoidf(z[k]);           // input gate
        const float fg = sigmoidf(z[H + k]);       // forget gate
        const float gg = std::tanh(z[2 * H + k]);  // cell candidate
        const float og = sigmoidf(z[3 * H + k]);   // output gate
        const float cn = fg * c_in[k] + ig * gg;
        const float hn = og * std::tanh(cn);
        c_out[k] = cn;
        h_out[k] = hn;
    }
}

// ---------------------------------------------------------------------------
// Full-sequence forward pass (unchanged API; now delegates to lstm_cell).
// ---------------------------------------------------------------------------
void PredictionNet::forward(const std::vector<int32_t>& ids, bool add_sos,
                            std::vector<float>& out, int& U_out, int& hidden) const {
    const int H = H_;

    const float* embed = tensor_data(ml_, "decoder.prediction.embed.weight");

    // Build the embedded input sequence.
    const int U   = (int)ids.size();
    const int seq = add_sos ? U + 1 : U;
    U_out  = seq;
    hidden = H;

    // X: row-major [seq, H]
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

    // Stacked-LSTM recurrence using the shared lstm_cell helper. Per-layer state
    // (h, c) is carried across timesteps; within a timestep layer l>0 consumes
    // the previous layer's hidden output. The sequence output is the TOP layer's
    // h' at each step (matching PyTorch nn.LSTM output semantics).
    const int L = n_layers_;
    out.assign((size_t)seq * H, 0.0f);
    std::vector<std::vector<float>> h(L, std::vector<float>(H, 0.0f));
    std::vector<std::vector<float>> c(L, std::vector<float>(H, 0.0f));
    std::vector<float> h_new(H), c_new(H);

    for (int t = 0; t < seq; ++t) {
        const float* layer_in = &X[(size_t)t * H];
        for (int l = 0; l < L; ++l) {
            lstm_cell(l, layer_in, h[l].data(), c[l].data(),
                      h_new.data(), c_new.data());
            h[l].swap(h_new);
            c[l].swap(c_new);
            layer_in = h[l].data(); // next layer consumes this layer's output
        }
        std::memcpy(&out[(size_t)t * H], h[L - 1].data(), (size_t)H * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Stateful helpers.
// ---------------------------------------------------------------------------
PredState PredictionNet::zero_state() const {
    PredState s;
    s.h.assign((size_t)n_layers_, std::vector<float>((size_t)H_, 0.0f));
    s.c.assign((size_t)n_layers_, std::vector<float>((size_t)H_, 0.0f));
    return s;
}

void PredictionNet::step(int32_t token_id, bool is_sos,
                          const PredState& in,
                          std::vector<float>& g,
                          PredState& out_state) const {
    const int H = H_;
    const int L = n_layers_;

    // Build the input embedding for this one step (layer 0's input).
    std::vector<float> x(H, 0.0f); // zero → SOS by default
    if (!is_sos) {
        assert(token_id >= 0 && token_id < vocab_p1_ && "embedding id out of range");
        const float* embed = tensor_data(ml_, "decoder.prediction.embed.weight");
        std::memcpy(x.data(), &embed[(size_t)token_id * H], (size_t)H * sizeof(float));
    }

    // Run one step through the stacked LSTM, layer l>0 consuming layer l-1's h'.
    out_state.h.assign((size_t)L, std::vector<float>((size_t)H));
    out_state.c.assign((size_t)L, std::vector<float>((size_t)H));
    const float* layer_in = x.data();
    for (int l = 0; l < L; ++l) {
        lstm_cell(l, layer_in, in.h[l].data(), in.c[l].data(),
                  out_state.h[l].data(), out_state.c[l].data());
        layer_in = out_state.h[l].data(); // feed this layer's output to the next
    }
    // g = top layer's h'
    g.resize(H);
    std::memcpy(g.data(), out_state.h[L - 1].data(), (size_t)H * sizeof(float));
}

} // namespace pk
