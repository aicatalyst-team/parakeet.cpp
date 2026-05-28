#pragma once
#include "model_loader.hpp"
#include <vector>
#include <cstdint>

namespace pk {

// RNN-Transducer prediction network — NeMo RNNTDecoder prediction net.
//
// Architecture:
//   Embedding(vocab+1, pred_hidden, padding_idx=blank) lookup
//   Single-layer LSTM (PyTorch convention)
//
// LSTM math (per step, input x_t [H], prev h,c [H]; h0=c0=0):
//   z = W_ih · x_t + b_ih + W_hh · h + b_hh        # [4H]
//   i = sigmoid(z[0:H]); f = sigmoid(z[H:2H]); g = tanh(z[2H:3H]); o = sigmoid(z[3H:4H])
//   c' = f * c + i * g
//   h' = o * tanh(c')
// Gate order is PyTorch [input, forget, cell, output] stacked in the 4H dim.
//
// add_sos=true prepends a literal zero [H] "start of sequence" vector to the
// embedded sequence (matching NeMo predict(add_sos=True)), so the output has
// U+1 hidden states. The output is the sequence of h' states.
//
// Tensors (verbatim NeMo names):
//   decoder.prediction.embed.weight              ggml ne=[H, vocab+1]
//   decoder.prediction.dec_rnn.lstm.weight_ih_l0 ggml ne=[H, 4H]
//   decoder.prediction.dec_rnn.lstm.weight_hh_l0 ggml ne=[H, 4H]
//   decoder.prediction.dec_rnn.lstm.bias_ih_l0   ggml ne=[4H]
//   decoder.prediction.dec_rnn.lstm.bias_hh_l0   ggml ne=[4H]
class PredictionNet {
public:
    explicit PredictionNet(const ModelLoader& ml);

    // ids:    input label ids (length U). Each id indexes embed.weight.
    // add_sos: prepend a zero SOS step (output length becomes U+1).
    // out:    row-major [U_out, hidden] — out[u*hidden + h]
    // U_out:  number of output steps (U, or U+1 if add_sos)
    // hidden: pred_hidden (H)
    void forward(const std::vector<int32_t>& ids, bool add_sos,
                 std::vector<float>& out, int& U_out, int& hidden) const;

    int hidden_size() const { return H_; }

private:
    const ModelLoader& ml_;
    int H_;       // pred_hidden
    int vocab_p1_; // vocab + 1 (embedding rows)
};

} // namespace pk
