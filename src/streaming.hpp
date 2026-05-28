#pragma once
#include "model_loader.hpp"
#include "streaming_encoder.hpp"
#include "prediction.hpp"
#include "joint.hpp"
#include "rnnt.hpp"
#include <memory>
#include <vector>
#include <cstdint>

namespace pk {

// Cache-aware streaming RNN-T session for the pure-RNNT streaming model
// nvidia/parakeet_realtime_eou_120m-v1.
//
// Owns a StreamingEncoder (carries per-layer conv + attention caches across
// chunks) AND the RNN-T greedy decoder state (prediction-net LSTM state, last
// emitted token, accumulated hypothesis). Per chunk:
//   1. StreamingEncoder::step(mel_chunk) -> the chunk's VALID encoder frames
//      [valid, d_model] row-major (the cache-aware-equivalent leading frames).
//   2. rnnt_decode_frames over those new frames, carrying the decoder state
//      (NOT reset between chunks). Newly emitted token ids are appended to the
//      session hypothesis and returned.
//
// Mirrors NeMo's rnnt_decoder_predictions_tensor(..., partial_hypotheses=...):
// the decoder state persists, so feeding the encoder frames in chunks produces
// the identical token sequence as decoding the whole offline encoder output at
// once.
//
// This is the decoder-facing surface only — PCM buffering / the C-API / EOU
// events are Task 7. The caller drives the encoder chunk schedule (mel windows
// already including pre-encode-cache overlap, matching test_streaming_encoder).
class StreamingSession {
public:
    explicit StreamingSession(const ModelLoader& ml);

    // Reset the encoder caches AND the decoder state to a fresh stream.
    void reset();

    // Feed one mel chunk window (row-major [n_mels, n_frames], feat-major
    // inner=time — the same orientation StreamingEncoder::step expects, already
    // including any pre-encode-cache overlap). Runs the encoder step then the
    // RNN-T decode over the new encoder frames. Returns the token ids emitted in
    // THIS chunk (and accumulates them into the session hypothesis tokens()).
    //
    // is_last selects the encoder keep_all_outputs path (final chunk keeps the
    // streaming tail). Defaults to false for mid-stream chunks.
    std::vector<int32_t> feed_mel_chunk(const std::vector<float>& mel_chunk,
                                        int n_frames, bool is_last = false);

    // The full accumulated emitted token-id sequence across all fed chunks.
    const std::vector<int32_t>& tokens() const { return state_.hyp; }

    // Streaming chunk schedule (delegated from the encoder), so the caller can
    // window the mel exactly like test_streaming_encoder.
    int chunk_size_first() const { return enc_.chunk_size_first(); }
    int chunk_size() const { return enc_.chunk_size(); }
    int pre_encode_cache_size() const { return enc_.pre_encode_cache_size(); }
    int valid_out_len() const { return enc_.valid_out_len(); }

private:
    const ModelLoader& ml_;
    StreamingEncoder enc_;
    PredictionNet pred_;
    Joint joint_;
    int d_model_;
    int blank_id_;
    int max_symbols_;
    RnntDecodeState state_;
};

} // namespace pk
