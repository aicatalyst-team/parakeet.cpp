#pragma once
#include "parakeet.h"          // pk::Decoder
#include "model_loader.hpp"
#include "transcription.hpp"   // pk::Transcription

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

    // Transcribe a batch of mono float PCM clips. Each is resampled to 16 kHz if
    // needed, then all run through the batched encoder; decode is per item.
    // Returns one transcript per input, in order.
    std::vector<std::string> transcribe_pcm_batch(
        const std::vector<std::vector<float>>& pcms, int sample_rate,
        Decoder decoder = Decoder::kDefault) const;

    // Transcribe raw mono float PCM, returning the flat text plus per-word and
    // per-token timestamps + confidence (matching NeMo timestamps=True +
    // 'max_prob' confidence). If `sample_rate != 16000` the audio is linearly
    // resampled to 16 kHz first. Throws std::runtime_error on failure.
    Transcription transcribe_with_timestamps(
        const std::vector<float>& pcm, int sample_rate,
        Decoder decoder = Decoder::kDefault) const;

    // Convenience: transcribe a WAV file with timestamps + confidence.
    Transcription transcribe_path_with_timestamps(
        const std::string& wav_path,
        Decoder decoder = Decoder::kDefault) const;

    const ParakeetConfig& config() const { return loader_.config(); }

    // The underlying loaded GGUF. Exposed so the streaming C-API can build a
    // pk::StreamingSession (and a MelFrontend) over the same load-once model.
    const ModelLoader& loader() const { return loader_; }

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

    // Core batched orchestration: N 16 kHz clips -> N transcripts. Stacks mels,
    // runs forward_batch, decodes each item with the existing greedy decoders.
    std::vector<std::string> transcribe_16k_batch(
        const std::vector<std::vector<float>>& pcms16k, Decoder decoder) const;

    // Core orchestration for the timestamps path: 16 kHz mono PCM -> full
    // Transcription (text + per-token TokenInfo + grouped words). Shared by the
    // two timestamp entry points.
    Transcription transcribe_16k_with_timestamps(
        const std::vector<float>& pcm16k, Decoder decoder) const;

    ModelLoader loader_;
};

} // namespace pk
