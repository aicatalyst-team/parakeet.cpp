#include "model.hpp"

#include "audio_io.hpp"
#include "mel.hpp"
#include "encoder.hpp"
#include "ctc_decoder.hpp"
#include "search.hpp"
#include "tokenizer.hpp"
#include "prediction.hpp"
#include "joint.hpp"
#include "tdt.hpp"
#include "rnnt.hpp"

#include <stdexcept>

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
    return m;
}

std::string Model::transcribe_16k(const std::vector<float>& pcm16k,
                                  Decoder decoder) const {
    const ParakeetConfig& cfg = loader_.config();

    // 1. Log-mel front end -> feats [n_mels, T].
    MelFrontend mel(loader_);
    std::vector<float> feats;
    int n_mels = 0, T = 0;
    mel.compute(pcm16k, feats, n_mels, T);

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

} // namespace pk
