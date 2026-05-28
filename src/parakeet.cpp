#include "parakeet.h"
#include "model_loader.hpp"
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

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define PARAKEET_VERSION "0.0.1"

extern "C" const char* parakeet_version(void) { return PARAKEET_VERSION; }

namespace pk {

// Returns true when the arch string indicates a TDT/RNNT transducer head should
// be used by default (NeMo's cur_decoder='rnnt' for hybrid models).
static bool arch_prefers_tdt(const std::string& arch) {
    return arch == "tdt"
        || arch == "hybrid_tdt_ctc"
        || arch == "rnnt"
        || arch == "hybrid_rnnt_ctc";
}

std::string transcribe(const std::string& model_path, const std::string& wav_path,
                       Decoder decoder) {
    // 1. Load the model (weights + config + tokenizer pieces).
    ModelLoader loader;
    if (!loader.load(model_path)) {
        throw std::runtime_error("parakeet: failed to load model: " + model_path);
    }
    const ParakeetConfig& cfg = loader.config();

    // 2. Load + resample audio to 16k mono.
    Audio audio;
    if (!load_audio_16k_mono(wav_path, audio)) {
        throw std::runtime_error("parakeet: failed to load audio: " + wav_path);
    }

    // 3. Log-mel front end -> feats [n_mels, T].
    MelFrontend mel(loader);
    std::vector<float> feats;
    int n_mels = 0, T = 0;
    mel.compute(audio.samples, feats, n_mels, T);

    // 4. FastConformer encoder -> enc_out [d_model, Tout] (channels-first).
    Encoder encoder(loader);
    std::vector<float> enc_out;
    int d_model = 0, Tout = 0;
    encoder.forward(feats, n_mels, T, enc_out, d_model, Tout);

    // Decide which head to use.
    const bool use_tdt = (decoder == Decoder::kTDT)
        || (decoder == Decoder::kDefault && arch_prefers_tdt(cfg.arch));

    if (use_tdt) {
        // 5a. TDT path: transpose encoder output to row-major [Tout, d_model].
        //     enc_out from Encoder is [d_model, Tout] (channels-first).
        std::vector<float> enc_row(Tout * d_model);
        for (int t = 0; t < Tout; ++t)
            for (int c = 0; c < d_model; ++c)
                enc_row[t * d_model + c] = enc_out[c * Tout + t];

        // 5b. Prediction net + Joint.
        PredictionNet pred(loader);
        Joint        joint(loader);

        // max_symbols: use NeMo's default of 10 (the converter does not currently
        // emit this value; if/when it does the config could be read here).
        const int max_symbols = 10;

        // Branch on the duration table: TDT (durations present) uses the
        // duration-aware greedy loop; a pure RNNT transducer (no durations, e.g.
        // arch ∈ {rnnt, hybrid_rnnt_ctc}) uses the standard RNNT greedy loop.
        std::vector<int32_t> ids;
        if (!cfg.tdt_durations.empty()) {
            ids = tdt_greedy(
                pred, joint, enc_row, Tout, d_model,
                cfg.tdt_durations, (int)cfg.blank_id, max_symbols);
        } else {
            ids = rnnt_greedy(
                pred, joint, enc_row, Tout, d_model,
                (int)cfg.blank_id, max_symbols);
        }

        // 6a. Detokenize.
        return detokenize(loader.tokenizer_pieces(), ids);

    } else {
        // 5b. CTC path: head -> log-probs [Tout, vocab+1].
        CTCDecoder ctc(loader);
        std::vector<float> logits;
        int vocab_plus_1 = 0;
        ctc.forward(enc_out, d_model, Tout, logits, vocab_plus_1);

        // 6b. CTC greedy collapse -> token ids. Blank is the last column.
        const int blank_id = (int)cfg.blank_id;
        std::vector<int32_t> ids = ctc_greedy(logits, Tout, vocab_plus_1, blank_id);

        // 7b. Detokenize.
        return detokenize(loader.tokenizer_pieces(), ids);
    }
}

} // namespace pk

extern "C" int parakeet_transcribe_file(const char* model_path,
                                        const char* wav_path, char** out) {
    if (!model_path || !wav_path || !out) return 1;
    try {
        std::string text = pk::transcribe(model_path, wav_path);
        char* buf = (char*)std::malloc(text.size() + 1);
        if (!buf) return 2;
        std::memcpy(buf, text.c_str(), text.size() + 1);
        *out = buf;
        return 0;
    } catch (const std::exception&) {
        return 3;
    } catch (...) {
        return 4;
    }
}

extern "C" void parakeet_free_string(char* s) { std::free(s); }
