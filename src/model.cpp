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

    if (use_tdt) {
        // 3a. TDT path: transpose encoder output to row-major [Tout, d_model].
        //     enc_out from Encoder is [d_model, Tout] (channels-first).
        std::vector<float> enc_row(static_cast<size_t>(Tout) * d_model);
        for (int t = 0; t < Tout; ++t)
            for (int c = 0; c < d_model; ++c)
                enc_row[t * d_model + c] = enc_out[c * Tout + t];

        // 3b. Prediction net + Joint.
        PredictionNet pred(loader_);
        Joint        joint(loader_);

        // max_symbols: greedy max symbols emitted per frame, read from the model
        // metadata (parakeet.decoding.max_symbols; NeMo default 10).
        const int max_symbols = static_cast<int>(cfg.max_symbols);

        // Branch on the duration table: TDT (durations present) uses the
        // duration-aware greedy loop; a pure RNNT transducer (no durations, e.g.
        // arch ∈ {rnnt, hybrid_rnnt_ctc}) uses the standard RNNT greedy loop.
        std::vector<int32_t> ids;
        if (!cfg.tdt_durations.empty()) {
            ids = tdt_greedy(
                pred, joint, enc_row, Tout, d_model,
                cfg.tdt_durations, static_cast<int>(cfg.blank_id), max_symbols);
        } else {
            ids = rnnt_greedy(
                pred, joint, enc_row, Tout, d_model,
                static_cast<int>(cfg.blank_id), max_symbols);
        }

        // 4a. Detokenize.
        return detokenize(loader_.tokenizer_pieces(), ids);

    } else {
        // 3b. CTC path: head -> log-probs [Tout, vocab+1].
        CTCDecoder ctc(loader_);
        std::vector<float> logits;
        int vocab_plus_1 = 0;
        ctc.forward(enc_out, d_model, Tout, logits, vocab_plus_1);

        // 4b. CTC greedy collapse -> token ids. Blank is the last column.
        const int blank_id = static_cast<int>(cfg.blank_id);
        std::vector<int32_t> ids = ctc_greedy(logits, Tout, vocab_plus_1, blank_id);

        // 5b. Detokenize.
        return detokenize(loader_.tokenizer_pieces(), ids);
    }
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
