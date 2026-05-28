#include "encoder.hpp"
#include "subsampling.hpp"
#include "conformer.hpp"
#include "pos_enc.hpp"
#include <cassert>
#include <cmath>
#include <vector>

namespace pk {

Encoder::Encoder(const ModelLoader& ml)
    : ml_(ml) {
    d_model_  = (int)ml.config().d_model;
    n_layers_ = (int)ml.config().n_layers;
    xscaling_ = ml.config().xscaling;
    assert(n_layers_ > 0 && d_model_ > 0);
}

void Encoder::forward(const std::vector<float>& mel, int n_mels, int T,
                      std::vector<float>& enc_out, int& d_model, int& Tout) const {
    std::vector<int> none;
    std::vector<std::vector<float>> ignored;
    forward_capture(mel, n_mels, T, enc_out, d_model, Tout, none, ignored);
}

void Encoder::forward_capture(const std::vector<float>& mel, int n_mels, int T,
                              std::vector<float>& enc_out, int& d_model, int& Tout,
                              const std::vector<int>& capture_layers,
                              std::vector<std::vector<float>>& layer_outs) const {
    // ---- 1. Subsampling: mel [n_mels,T] -> x [T', d_model], plus valid_len. ----
    Subsampling sub(ml_);
    std::vector<float> x;        // row-major [T', d_model]
    int Tp = 0, dm = 0, valid_len = 0;
    sub.forward(mel, n_mels, T, x, Tp, dm, valid_len);
    assert(dm == d_model_);

    // ---- 2. xscaling (gated; off for this model). NeMo scales x, not pos_emb. ----
    if (xscaling_) {
        const float xscale = std::sqrt((float)d_model_);
        for (float& v : x) v *= xscale;
    }

    // ---- 3. Relative positional encoding pos_emb [2T'-1, d_model]. ----
    std::vector<float> pos_emb;
    rel_pos_encoding(Tp, d_model_, pos_emb);
    const int pos_len = 2 * Tp - 1;

    // ---- 4. Conformer layer stack. Each layer: [T',d_model] -> [T',d_model]. ----
    layer_outs.assign(capture_layers.size(), {});
    std::vector<float> next;
    for (int i = 0; i < n_layers_; ++i) {
        ConformerLayer layer(ml_, i);
        layer.forward(x, Tp, pos_emb, pos_len, valid_len, next);
        x.swap(next);
        for (size_t c = 0; c < capture_layers.size(); ++c) {
            if (capture_layers[c] == i) layer_outs[c] = x;  // copy [T', d_model]
        }
    }

    // ---- 5. Final transpose [T', d_model] -> [d_model, T'] (channels-first). ----
    // x is row-major [T', d_model] (x[t*d_model + c]); produce enc_out[c*T' + t].
    enc_out.assign((size_t)d_model_ * Tp, 0.0f);
    for (int t = 0; t < Tp; ++t)
        for (int c = 0; c < d_model_; ++c)
            enc_out[(size_t)c * Tp + t] = x[(size_t)t * d_model_ + c];

    d_model = d_model_;
    Tout = Tp;
}

} // namespace pk
