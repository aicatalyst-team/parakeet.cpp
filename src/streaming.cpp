#include "streaming.hpp"
#include <cassert>

namespace pk {

StreamingSession::StreamingSession(const ModelLoader& ml)
    : ml_(ml), enc_(ml), pred_(ml), joint_(ml) {
    const ParakeetConfig& cfg = ml.config();
    d_model_  = (int)cfg.d_model;
    blank_id_ = (int)cfg.blank_id;
    // NeMo's greedy default of 10 symbols per frame (the converter does not emit
    // this value; matches the offline pk::transcribe path in model.cpp).
    max_symbols_ = 10;
    assert(joint_.num_durations() == 0 && "StreamingSession is RNN-T only (no TDT durations)");
    reset();
}

void StreamingSession::reset() {
    enc_.reset();
    state_ = rnnt_decode_init(pred_);
}

std::vector<int32_t> StreamingSession::feed_mel_chunk(const std::vector<float>& mel_chunk,
                                                      int n_frames, bool is_last) {
    // 1. Encoder step: the chunk's valid encoder frames, row-major [valid, d_model]
    //    (d_model fastest) — exactly the orientation rnnt_decode_frames expects.
    int n_valid = 0;
    std::vector<float> enc_frames = enc_.step(mel_chunk, n_frames, is_last, n_valid);

    if (n_valid <= 0) {
        return {};
    }

    // 2. RNN-T greedy over the new encoder frames, carrying the decoder state
    //    across chunks (do NOT reset). Appends to state_.hyp and returns the ids
    //    emitted in this chunk.
    return rnnt_decode_frames(pred_, joint_, enc_frames, n_valid, d_model_,
                              state_, blank_id_, max_symbols_);
}

} // namespace pk
