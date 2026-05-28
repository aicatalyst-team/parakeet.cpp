#pragma once
#include "model_loader.hpp"
#include <vector>
namespace pk {

// dw_striding (÷8) convolutional subsampling front end of the FastConformer
// encoder (NeMo ConvSubsampling, dw_striding branch). Builds one ggml graph.
class Subsampling {
public:
    explicit Subsampling(const ModelLoader& ml);
    // mel: row-major [n_mels, T] (feat-major inner = T) — i.e. mel[m*T + t].
    // out: row-major [Tout, d_model] (time-major) matching baseline subsampling_out.
    void forward(const std::vector<float>& mel, int n_mels, int T,
                 std::vector<float>& out, int& Tout, int& d_model) const;

    // Same as forward(), but also returns `valid_len`: the number of non-padding
    // output frames after the conv stride reductions (NeMo's masked conv length).
    // Frames in [valid_len, Tout) are center-pad and must be masked downstream
    // (attention key/query masking, depthwise-conv pad masking). This is the
    // value that should be threaded into ConformerLayer::forward.
    void forward(const std::vector<float>& mel, int n_mels, int T,
                 std::vector<float>& out, int& Tout, int& d_model,
                 int& valid_len) const;

    // Number of valid (non-pad) output frames for an input of T mel frames,
    // applying the same per-stage `calc_length` reductions NeMo uses. Pure
    // arithmetic, no graph; exposed so the encoder can derive valid_len.
    int valid_out_len(int T) const;
private:
    const ModelLoader& ml_;
    int conv_channels_;   // C
    int d_model_;         // out features
};

} // namespace pk
