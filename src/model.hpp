#pragma once
#include "parakeet.h"          // pk::Decoder
#include "model_loader.hpp"

#include <memory>
#include <string>
#include <vector>

namespace pk {

// Load-once transcription context.
//
// Loads a GGUF model ONCE (owns the ModelLoader) and reuses it across many
// transcribe calls — in contrast to the free function pk::transcribe(), which
// reloads the model on every call. This is what the flat C-API holds.
//
// The component objects (MelFrontend, Encoder, PredictionNet, Joint, ...) are
// lightweight views over the ModelLoader (they hold `const ModelLoader&`), so
// they are constructed per call; the expensive part — parsing the GGUF and
// mapping every weight tensor — happens exactly once, in load().
class Model {
public:
    // Loads the GGUF at `gguf_path`. Returns nullptr on failure (no throw).
    static std::unique_ptr<Model> load(const std::string& gguf_path);

    // Transcribe raw mono float PCM. If `sample_rate != 16000` the audio is
    // linearly resampled to 16 kHz (via pk::resample_linear) before inference.
    // Throws std::runtime_error on failure (e.g. unsupported arch).
    std::string transcribe_pcm(const std::vector<float>& pcm, int sample_rate,
                               Decoder decoder = Decoder::kDefault) const;

    // Transcribe a WAV file (loaded + resampled to 16 kHz mono via
    // pk::load_audio_16k_mono). Throws std::runtime_error on failure.
    std::string transcribe_path(const std::string& wav_path,
                                Decoder decoder = Decoder::kDefault) const;

    const ParakeetConfig& config() const { return loader_.config(); }

    // Non-copyable (owns the GGUF mapping).
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

private:
    Model() = default;

    // Core orchestration: 16 kHz mono PCM -> transcript. Shared by the two
    // public entry points (transcribe_pcm resamples first; transcribe_path
    // loads the WAV first).
    std::string transcribe_16k(const std::vector<float>& pcm16k,
                               Decoder decoder) const;

    ModelLoader loader_;
};

} // namespace pk
