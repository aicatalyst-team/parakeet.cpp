#include "model.hpp"

#include "audio_io.hpp"
#include "mel.hpp"
#include "mel_gpu.hpp"
#include "encoder.hpp"
#include "ctc_decoder.hpp"
#include "search.hpp"
#include "tokenizer.hpp"
#include "prediction.hpp"
#include "joint.hpp"
#include "tdt.hpp"
#include "rnnt.hpp"
#include "transcription.hpp"
#include "decode_types.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace pk {

namespace {
// Returns true when the arch string indicates a TDT/RNNT transducer head should
// be used by default (NeMo's cur_decoder='rnnt' for hybrid models).
bool arch_prefers_tdt(const std::string& arch) {
    return arch == "tdt"
        || arch == "hybrid_tdt_ctc"
        || arch == "rnnt"
        || arch == "hybrid_rnnt_ctc";
}
} // namespace

std::unique_ptr<Model> Model::load(const std::string& gguf_path) {
    // unique_ptr<Model> via private ctor: construct then load. We avoid
    // std::make_unique (private ctor) and never throw out of here.
    std::unique_ptr<Model> m(new (std::nothrow) Model());
    if (!m) return nullptr;
    if (!m->loader_.load(gguf_path)) {
        return nullptr;
    }
    // Give the weights a CPU backend buffer ONCE so graphs reference them
    // directly as leaves (zero per-call copy). Done at load (vs. lazily on first
    // clone_weight) so the cost is paid up front, not per utterance.
    ensure_weights_realized(m->loader_);
    return m;
}

// Decode one item's encoder output (row-major [d_model, Tout], channels-first)
// into a transcript. Mirrors the tail of transcribe_16k exactly.
static std::string decode_enc_out(const ModelLoader& loader,
                                  const std::vector<float>& enc_out,
                                  int d_model, int Tout, bool use_tdt) {
    const ParakeetConfig& cfg = loader.config();
    if (use_tdt) {
        std::vector<float> enc_row((size_t)Tout * d_model);
        for (int t = 0; t < Tout; ++t)
            for (int c = 0; c < d_model; ++c)
                enc_row[(size_t)t * d_model + c] = enc_out[(size_t)c * Tout + t];
        PredictionNet pred(loader);
        Joint        joint(loader);
        const int max_symbols = static_cast<int>(cfg.max_symbols);
        std::vector<int32_t> ids;
        if (!cfg.tdt_durations.empty())
            ids = tdt_greedy(pred, joint, enc_row, Tout, d_model,
                             cfg.tdt_durations, (int)cfg.blank_id, max_symbols);
        else
            ids = rnnt_greedy(pred, joint, enc_row, Tout, d_model,
                              (int)cfg.blank_id, max_symbols);
        return detokenize(loader.tokenizer_pieces(), ids);
    } else {
        CTCDecoder ctc(loader);
        std::vector<float> logits; int vocab_plus_1 = 0;
        ctc.forward(enc_out, d_model, Tout, logits, vocab_plus_1);
        std::vector<int32_t> ids = ctc_greedy(logits, Tout, vocab_plus_1,
                                              (int)cfg.blank_id);
        return detokenize(loader.tokenizer_pieces(), ids);
    }
}

std::string Model::transcribe_16k(const std::vector<float>& pcm16k,
                                  Decoder decoder) const {
    const ParakeetConfig& cfg = loader_.config();

    // 1. Log-mel front end -> feats [n_mels, T]. On a non-CPU backend run the
    //    heavy STFT/power/filterbank/log on the backend (GPU) via GpuMel; on CPU
    //    keep the byte-identical FFT-based MelFrontend.
    std::vector<float> feats;
    int n_mels = 0, T = 0;
    if (std::string(pk::global_backend().device_name()) != "cpu") {
        GpuMel gmel(loader_);
        gmel.compute(pcm16k, feats, n_mels, T);
    } else {
        MelFrontend mel(loader_);
        mel.compute(pcm16k, feats, n_mels, T);
    }

    // 2. FastConformer encoder -> enc_out [d_model, Tout] (channels-first).
    Encoder encoder(loader_);
    std::vector<float> enc_out;
    int d_model = 0, Tout = 0;
    encoder.forward(feats, n_mels, T, enc_out, d_model, Tout);

    // Decide which head to use.
    const bool use_tdt = (decoder == Decoder::kTDT)
        || (decoder == Decoder::kDefault && arch_prefers_tdt(cfg.arch));

    return decode_enc_out(loader_, enc_out, d_model, Tout, use_tdt);
}

std::vector<std::string> Model::transcribe_16k_batch(
    const std::vector<std::vector<float>>& pcms16k, Decoder decoder) const {
    const ParakeetConfig& cfg = loader_.config();
    const bool use_tdt = (decoder == Decoder::kTDT)
        || (decoder == Decoder::kDefault && arch_prefers_tdt(cfg.arch));

    // 1. Per-clip mel, then stack to T_max.
    const bool gpu = std::string(pk::global_backend().device_name()) != "cpu";
    MelBatch mb;
    mb.B = (int)pcms16k.size();
    std::vector<std::vector<float>> feats(mb.B);
    std::vector<int> Ts(mb.B, 0);
    int n_mels = 0;
    for (int b = 0; b < mb.B; ++b) {
        int nm = 0, T = 0;
        if (gpu) { GpuMel g(loader_); g.compute(pcms16k[b], feats[b], nm, T); }
        else     { MelFrontend m(loader_); m.compute(pcms16k[b], feats[b], nm, T); }
        n_mels = nm; Ts[b] = T;
    }
    mb.n_mels = n_mels;
    mb.T_max = 0;
    for (int b = 0; b < mb.B; ++b) mb.T_max = std::max(mb.T_max, Ts[b]);
    mb.valid_T = Ts;
    mb.data.assign((size_t)mb.B * n_mels * mb.T_max, 0.0f);
    for (int b = 0; b < mb.B; ++b)
        for (int m = 0; m < n_mels; ++m)
            for (int t = 0; t < Ts[b]; ++t)
                mb.data[((size_t)b * n_mels + m) * mb.T_max + t] =
                    feats[b][(size_t)m * Ts[b] + t];

    // 2. Batched encoder.
    Encoder encoder(loader_);
    std::vector<std::vector<float>> enc_outs; int d_model = 0, Tout = 0;
    std::vector<int> valid_Tout;
    encoder.forward_batch(mb, enc_outs, d_model, Tout, valid_Tout);

    // 3. Per-item decode (each enc_out is [d_model, valid_Tout[b]]).
    std::vector<std::string> outs(mb.B);
    for (int b = 0; b < mb.B; ++b)
        outs[b] = decode_enc_out(loader_, enc_outs[b], d_model, valid_Tout[b], use_tdt);
    return outs;
}

std::vector<std::string> Model::transcribe_pcm_batch(
    const std::vector<std::vector<float>>& pcms, int sample_rate,
    Decoder decoder) const {
    std::vector<std::vector<float>> r(pcms.size());
    for (size_t i = 0; i < pcms.size(); ++i)
        r[i] = (sample_rate == 16000) ? pcms[i]
                                      : resample_linear(pcms[i], sample_rate, 16000);
    return transcribe_16k_batch(r, decoder);
}

Transcription Model::transcribe_16k_with_timestamps(
    const std::vector<float>& pcm16k, Decoder decoder) const {
    const ParakeetConfig& cfg = loader_.config();

    // frame_sec = hop_length * subsampling_factor / sample_rate (= 0.08 s here).
    // This is NeMo's window_stride * subsampling_factor (window_stride =
    // hop_length / sample_rate).
    const float frame_sec =
        (float)cfg.hop_length * (float)cfg.subsampling_factor / (float)cfg.sample_rate;

    // 1. Log-mel front end -> feats [n_mels, T]. On a non-CPU backend run the
    //    heavy STFT/power/filterbank/log on the backend (GPU) via GpuMel; on CPU
    //    keep the byte-identical FFT-based MelFrontend.
    std::vector<float> feats;
    int n_mels = 0, T = 0;
    if (std::string(pk::global_backend().device_name()) != "cpu") {
        GpuMel gmel(loader_);
        gmel.compute(pcm16k, feats, n_mels, T);
    } else {
        MelFrontend mel(loader_);
        mel.compute(pcm16k, feats, n_mels, T);
    }

    // 2. FastConformer encoder -> enc_out [d_model, Tout] (channels-first).
    Encoder encoder(loader_);
    std::vector<float> enc_out;
    int d_model = 0, Tout = 0;
    encoder.forward(feats, n_mels, T, enc_out, d_model, Tout);

    const bool use_tdt = (decoder == Decoder::kTDT)
        || (decoder == Decoder::kDefault && arch_prefers_tdt(cfg.arch));

    Transcription result;
    std::vector<TokenInfo> toks;

    if (use_tdt) {
        // Transpose channels-first [d_model, Tout] -> row-major [Tout, d_model].
        std::vector<float> enc_row(static_cast<size_t>(Tout) * d_model);
        for (int t = 0; t < Tout; ++t)
            for (int c = 0; c < d_model; ++c)
                enc_row[static_cast<size_t>(t) * d_model + c] = enc_out[static_cast<size_t>(c) * Tout + t];

        PredictionNet pred(loader_);
        Joint        joint(loader_);
        const int max_symbols = static_cast<int>(cfg.max_symbols);

        if (!cfg.tdt_durations.empty()) {
            tdt_greedy(pred, joint, enc_row, Tout, d_model,
                       cfg.tdt_durations, static_cast<int>(cfg.blank_id),
                       max_symbols, &toks);
        } else {
            rnnt_greedy(pred, joint, enc_row, Tout, d_model,
                        static_cast<int>(cfg.blank_id), max_symbols, &toks);
        }
        // TDT/RNNT TokenInfo.span is already the per-token end-offset extent
        // (duration for TDT, 1 for RNNT) -> group_words' frame+span rule is
        // correct as-is.
    } else {
        // CTC path: head -> log-probs [Tout, vocab+1].
        CTCDecoder ctc(loader_);
        std::vector<float> logits;
        int vocab_plus_1 = 0;
        ctc.forward(enc_out, d_model, Tout, logits, vocab_plus_1);

        const int blank_id = static_cast<int>(cfg.blank_id);
        ctc_greedy(logits, Tout, vocab_plus_1, blank_id, &toks);

        // NeMo CTC word end_offset = the NEXT collapsed token's start frame
        // (cumulative run lengths), not start+1. ctc_greedy documents span == 1;
        // rewrite each token's span to (next_frame - frame) so group_words'
        // `frame + span` rule reproduces NeMo's end_offset exactly. The final
        // token keeps span == 1 (its true run-length is unknown to the collapse,
        // and within the 1-frame word-end tolerance).
        for (size_t i = 0; i + 1 < toks.size(); ++i) {
            toks[i].span = toks[i + 1].frame - toks[i].frame;
        }
    }

    // Detokenize the flat text from the emitted ids.
    std::vector<int32_t> ids;
    ids.reserve(toks.size());
    for (const TokenInfo& ti : toks) ids.push_back(ti.id);
    result.text   = detokenize(loader_.tokenizer_pieces(), ids);
    result.words  = group_words(toks, loader_.tokenizer_pieces(), frame_sec);
    result.tokens = std::move(toks);
    return result;
}

std::string Model::transcribe_pcm(const std::vector<float>& pcm, int sample_rate,
                                  Decoder decoder) const {
    if (sample_rate <= 0) {
        throw std::runtime_error("parakeet: invalid sample_rate");
    }
    if (sample_rate == 16000) {
        return transcribe_16k(pcm, decoder);
    }
    std::vector<float> pcm16k = resample_linear(pcm, sample_rate, 16000);
    return transcribe_16k(pcm16k, decoder);
}

std::string Model::transcribe_path(const std::string& wav_path,
                                   Decoder decoder) const {
    Audio audio;
    if (!load_audio_16k_mono(wav_path, audio)) {
        throw std::runtime_error("parakeet: failed to load audio: " + wav_path);
    }
    // load_audio_16k_mono already resamples to 16 kHz mono.
    return transcribe_16k(audio.samples, decoder);
}

Transcription Model::transcribe_with_timestamps(
    const std::vector<float>& pcm, int sample_rate, Decoder decoder) const {
    if (sample_rate <= 0) {
        throw std::runtime_error("parakeet: invalid sample_rate");
    }
    if (sample_rate == 16000) {
        return transcribe_16k_with_timestamps(pcm, decoder);
    }
    std::vector<float> pcm16k = resample_linear(pcm, sample_rate, 16000);
    return transcribe_16k_with_timestamps(pcm16k, decoder);
}

Transcription Model::transcribe_path_with_timestamps(
    const std::string& wav_path, Decoder decoder) const {
    Audio audio;
    if (!load_audio_16k_mono(wav_path, audio)) {
        throw std::runtime_error("parakeet: failed to load audio: " + wav_path);
    }
    return transcribe_16k_with_timestamps(audio.samples, decoder);
}

} // namespace pk
