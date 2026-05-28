#include "parakeet.h"
#include "model_loader.hpp"
#include "audio_io.hpp"
#include "mel.hpp"
#include "encoder.hpp"
#include "ctc_decoder.hpp"
#include "search.hpp"
#include "tokenizer.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define PARAKEET_VERSION "0.0.1"

extern "C" const char* parakeet_version(void) { return PARAKEET_VERSION; }

namespace pk {

std::string transcribe(const std::string& model_path, const std::string& wav_path) {
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

    // 4. FastConformer encoder -> enc_out [d_model, Tout].
    Encoder encoder(loader);
    std::vector<float> enc_out;
    int d_model = 0, Tout = 0;
    encoder.forward(feats, n_mels, T, enc_out, d_model, Tout);

    // 5. CTC head -> log-probs [Tout, vocab+1].
    CTCDecoder ctc(loader);
    std::vector<float> logits;
    int vocab_plus_1 = 0;
    ctc.forward(enc_out, d_model, Tout, logits, vocab_plus_1);

    // 6. CTC greedy collapse -> token ids. Blank is the last column
    //    (blank_id == vocab_size for the hybrid CTC head).
    const int blank_id = (int)cfg.blank_id;
    std::vector<int32_t> ids = ctc_greedy(logits, Tout, vocab_plus_1, blank_id);

    // 7. Detokenize.
    return detokenize(loader.tokenizer_pieces(), ids);
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
