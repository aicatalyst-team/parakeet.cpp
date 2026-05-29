#pragma once
#include "model_loader.hpp"
#include "graph_builder.hpp"
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace pk {

// dw_striding (÷8) convolutional subsampling front end of the FastConformer
// encoder (NeMo ConvSubsampling, dw_striding branch). Builds one ggml graph.
class Subsampling {
public:
    explicit Subsampling(const ModelLoader& ml);

    // GRAPH-BUILDER: append the subsampling ops to a SHARED graph `ctx` and
    // return the output tensor [d_model, T'] (ggml ne0=d_model fastest, i.e.
    // row-major [T', d_model]). `mel` is the host mel [n_mels, T] (feat-major
    // inner = T, mel[m*T+t]); it is transposed host-side into `pool`-owned
    // storage and fed as a graph input, so it must outlive the enclosing
    // Backend::compute. `out_Tp`/`out_valid` receive the subsampled time length
    // and valid (non-pad) frame count. Reused by both the fused encoder and the
    // unit test. Uses pk::clone_weight (zero-copy weights) + pk::graph_input_*
    // for host-built masks (registered into `pool` for lifetime).
    ggml_tensor* build_graph(ggml_context* ctx, const std::vector<float>& mel,
                             int n_mels, int T, GraphInputPool& pool,
                             int& out_Tp, int& out_valid,
                             int in_valid_frames = -1) const;
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

    // Streaming variant: `in_valid_frames` is the number of VALID (real, non-pad)
    // input mel frames for this window. The offline forward assumes the
    // preprocessor's center-pad convention (valid = T-1); for cache-aware
    // streaming the whole chunk window is real audio (valid = T, or the
    // clamped chunk length), so the entry valid length must be supplied
    // explicitly instead of using T-1. When in_valid_frames < 0 the offline
    // (T-1) behaviour is used (identical to the overload above).
    void forward(const std::vector<float>& mel, int n_mels, int T,
                 std::vector<float>& out, int& Tout, int& d_model,
                 int& valid_len, int in_valid_frames) const;

    // Number of valid (non-pad) output frames for an input of T mel frames,
    // applying the same per-stage `calc_length` reductions NeMo uses. Pure
    // arithmetic, no graph; exposed so the encoder can derive valid_len.
    // `in_valid_frames` (>=0) overrides the offline T-1 entry valid length
    // with an explicit count (streaming); <0 keeps the offline convention.
    int valid_out_len(int T, int in_valid_frames = -1) const;
private:
    const ModelLoader& ml_;
    int conv_channels_;   // C
    int d_model_;         // out features
    bool causal_ = false; // causal_downsampling: left-heavy conv padding (streaming)
};

} // namespace pk
