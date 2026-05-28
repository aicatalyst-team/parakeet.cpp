#include "streaming.hpp"
#include "tokenizer.hpp"
#include "mel.hpp"
#include <algorithm>
#include <cassert>

namespace pk {

StreamingSession::StreamingSession(const ModelLoader& ml)
    : ml_(ml), enc_(ml), pred_(ml), joint_(ml) {
    const ParakeetConfig& cfg = ml.config();
    d_model_  = (int)cfg.d_model;
    blank_id_ = (int)cfg.blank_id;
    // Greedy max symbols per frame, from model metadata (NeMo default 10);
    // matches the offline pk::transcribe path in model.cpp.
    max_symbols_ = (int)cfg.max_symbols;
    assert(joint_.num_durations() == 0 && "StreamingSession is RNN-T only (no TDT durations)");

    // Resolve the <EOU>/<EOB> special token ids from the tokenizer pieces (do
    // NOT hardcode 1024/1025; read them from the loaded vocab).
    const auto& pieces = cfg.tokenizer_pieces;
    for (int i = 0; i < (int)pieces.size(); ++i) {
        if (pieces[i] == "<EOU>") eou_id_ = i;
        else if (pieces[i] == "<EOB>") eob_id_ = i;
    }

    // Per-encoder-frame stride in seconds: each encoder output frame spans
    // hop_length * subsampling_factor input samples (the subsampling downsamples
    // the mel-frame time axis by subsampling_factor). time = frame * stride.
    const double hop = (double)cfg.hop_length;
    const double sub = (double)(cfg.subsampling_factor ? cfg.subsampling_factor : 1);
    const double sr  = (double)(cfg.sample_rate ? cfg.sample_rate : 16000);
    frame_sec_ = (hop * sub) / sr;

    reset();
}

void StreamingSession::reset() {
    enc_.reset();
    state_ = rnnt_decode_init(pred_);
    // eou_id_/eob_id_/frame_sec_ are resolved once in the constructor.
    enc_frame_ = 0;
    non_special_.clear();
    text_.clear();
    text_taken_ = 0;
    last_chunk_had_eou_ = false;
    events_.clear();
}

void StreamingSession::process_emitted(const std::vector<int32_t>& emitted) {
    last_chunk_had_eou_ = false;
    bool text_changed = false;
    for (int32_t tok : emitted) {
        if (tok == eou_id_ || tok == eob_id_) {
            // EOU/EOB: surface as an event, do NOT add to the text.
            EouEvent ev;
            ev.token = tok;
            ev.is_eob = (tok == eob_id_);
            // The frame index is recorded into events_ by feed/finalize (which
            // knows the per-token frame); we patch it there. Here we only know
            // the token, so push a placeholder the caller will overwrite.
            ev.encoder_frame = enc_frame_;  // refined by caller via per-token frame
            ev.time_sec = ev.encoder_frame * frame_sec_;
            events_.push_back(ev);
            last_chunk_had_eou_ = true;
        } else {
            non_special_.push_back(tok);
            text_changed = true;
        }
    }
    if (text_changed) {
        text_ = detokenize(ml_.config().tokenizer_pieces, non_special_);
    }
}

std::vector<int32_t> StreamingSession::feed_mel_chunk(const std::vector<float>& mel_chunk,
                                                      int n_frames, bool is_last) {
    // 1. Encoder step: the chunk's valid encoder frames, row-major [valid, d_model]
    //    (d_model fastest) — exactly the orientation rnnt_decode_frames expects.
    int n_valid = 0;
    std::vector<float> enc_frames = enc_.step(mel_chunk, n_frames, is_last, n_valid);

    if (n_valid <= 0) {
        last_chunk_had_eou_ = false;
        return {};
    }

    // 2. RNN-T greedy over the new encoder frames, carrying the decoder state
    //    across chunks (do NOT reset). Appends to state_.hyp and returns the ids
    //    emitted in this chunk, with their LOCAL frame index in [0, n_valid).
    const int base_frame = enc_frame_;
    std::vector<int32_t> local_frames;
    std::vector<int32_t> emitted =
        rnnt_decode_frames(pred_, joint_, enc_frames, n_valid, d_model_,
                           state_, blank_id_, max_symbols_, &local_frames);
    enc_frame_ += n_valid;

    // 3. Update text + EOU events; refine each new event's absolute frame index
    //    from the per-token local frame the decoder reported.
    const size_t prev_events = events_.size();
    process_emitted(emitted);
    // Re-walk emitted to assign the correct absolute frame to each new event.
    size_t evi = prev_events;
    for (size_t i = 0; i < emitted.size(); ++i) {
        if (emitted[i] == eou_id_ || emitted[i] == eob_id_) {
            const int abs_frame = base_frame + (int)local_frames[i];
            events_[evi].encoder_frame = abs_frame;
            events_[evi].time_sec = abs_frame * frame_sec_;
            ++evi;
        }
    }
    return emitted;
}

std::string StreamingSession::finalize() {
    // The end-of-stream tail is flushed by the caller feeding the final buffered
    // mel chunk with is_last=true (which keeps the streaming tail frames). At the
    // session level there is no further audio to process here, so finalize just
    // returns whatever newly-finalized text remains since the last take. NeMo's
    // cache-aware streaming behaves identically: the final chunk's incomplete
    // right context means a trailing <EOU> is NOT recovered, so we never
    // fabricate one — finalize emits only the carried-state tokens already
    // decoded.
    return take_new_text();
}

std::string StreamingSession::take_new_text() {
    if (text_taken_ >= text_.size()) return std::string();
    std::string delta = text_.substr(text_taken_);
    text_taken_ = text_.size();
    return delta;
}

std::vector<EouEvent> StreamingSession::drain_events() {
    std::vector<EouEvent> out;
    out.swap(events_);
    return out;
}

void run_stream_over_pcm(
    StreamingSession& sess, const ModelLoader& ml,
    const std::vector<float>& pcm16k,
    const std::function<void(const std::string&,
                             const std::vector<EouEvent>&)>& on_chunk) {
    // 1. Full-clip mel [n_mels, T] (feat-major inner=T), matching the offline /
    //    NeMo online_normalization=False reference (normalization over the whole
    //    clip). The streaming numerics come from the carried encoder/decoder
    //    caches, not from chunking the front end.
    MelFrontend mel_fe(ml);
    std::vector<float> mel;
    int n_mels = 0, T = 0;
    mel_fe.compute(pcm16k, mel, n_mels, T);
    if (T <= 0) return;

    const int chunk0     = sess.chunk_size_first();      // 9
    const int chunk_main = sess.chunk_size();            // 16
    const int pre_cache  = sess.pre_encode_cache_size(); // 9

    // mel[:, lo:hi] in feat-major layout.
    auto window = [&](int lo, int hi) {
        const int len = hi - lo;
        std::vector<float> w((size_t)n_mels * len);
        for (int m = 0; m < n_mels; ++m)
            for (int t = 0; t < len; ++t)
                w[(size_t)m * len + t] = mel[(size_t)m * T + (lo + t)];
        return w;
    };

    int buffer_idx = 0;
    bool first = true;
    while (buffer_idx < T) {
        const int chunk_size = first ? chunk0 : chunk_main;
        const int shift      = chunk_size;  // shift_size == chunk_size here
        const int chunk_hi   = std::min(buffer_idx + chunk_size, T);
        if (chunk_hi - buffer_idx <= 0) break;
        const int lo = first ? buffer_idx : std::max(0, buffer_idx - pre_cache);
        std::vector<float> win = window(lo, chunk_hi);
        const int win_frames = chunk_hi - lo;
        const bool is_last = (chunk_hi >= T);

        sess.feed_mel_chunk(win, win_frames, is_last);

        if (on_chunk) {
            std::string nt = sess.take_new_text();
            std::vector<EouEvent> ev = sess.drain_events();
            if (!nt.empty() || !ev.empty()) on_chunk(nt, ev);
        }

        buffer_idx += shift;
        first = false;
    }
}

} // namespace pk
