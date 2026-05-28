#pragma once
#include "model_loader.hpp"
#include <vector>
namespace pk {

// Full FastConformer encoder (NeMo ConformerEncoder.forward / forward_internal):
//
//   mel [n_mels, T] -> Subsampling (dw_striding ÷8) -> [T', d_model]
//                   -> (if xscaling) x *= sqrt(d_model)
//                   -> RelPositionalEncoding produces pos_emb [2T'-1, d_model]
//                   -> for i in 0..n_layers-1: ConformerLayer(i).forward(...)
//                   -> transpose to [d_model, T']  (NeMo returns [B, d_model, T'])
//
// The `valid_len` (number of non-pad output frames) is derived from Subsampling
// and threaded into every ConformerLayer (attention + conv pad masking).
class Encoder {
public:
    explicit Encoder(const ModelLoader& ml);

    // mel: row-major [n_mels, T] (feat-major inner = T), i.e. mel[m*T + t].
    // enc_out: row-major [d_model, Tout] (channels-first, time fastest), matching
    //          the baseline `encoder_out` orientation: enc_out[c*Tout + t].
    void forward(const std::vector<float>& mel, int n_mels, int T,
                 std::vector<float>& enc_out, int& d_model, int& Tout) const;

    // Same as forward(), but also captures the per-layer outputs at indices
    // `capture_layers` (each row-major [T', d_model]) into `layer_outs` (parallel
    // to capture_layers). Used by the parity test to localize divergence.
    void forward_capture(const std::vector<float>& mel, int n_mels, int T,
                         std::vector<float>& enc_out, int& d_model, int& Tout,
                         const std::vector<int>& capture_layers,
                         std::vector<std::vector<float>>& layer_outs) const;

private:
    const ModelLoader& ml_;
    int d_model_;
    int n_layers_;
    bool xscaling_;
};

} // namespace pk
