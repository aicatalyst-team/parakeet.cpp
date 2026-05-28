#include "model_loader.hpp"
#include "audio_io.hpp"
#include "mel.hpp"
#include "encoder.hpp"
#include "prediction.hpp"
#include "joint.hpp"
#include "tdt.hpp"
#include "ggml.h"
#include "gguf.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdint>
#include <exception>

// TDT duration-aware greedy decoding — token-id parity vs NeMo.
//
// Runs the full C++ path on the committed real-speech clip:
//   speech.wav -> load_audio_16k_mono -> MelFrontend -> Encoder
//              -> transpose encoder_out [d_model, T] to [T, d_model]
//              -> pk::tdt_greedy(prediction, joint, ...)
// and asserts the emitted token-id sequence EQUALS NeMo's TDT greedy reference
// `tdt_token_ids` (captured by scripts/gen_nemo_baseline.py on the same clip),
// exactly — every id in order.
//
// Env:
//   PARAKEET_TEST_GGUF            : model weights (required; skip 77 if unset)
//   PARAKEET_TEST_BASELINE_SPEECH : speech baseline gguf carrying tdt_token_ids
//                                   (required; skip 77 if unset)

// Read tdt_token_ids (int32) from a baseline gguf. Returns true on success;
// `present` is false (with empty `out`) when the model emitted no tokens
// (baseline.tdt_token_count == 0, so the tensor is absent by design).
static bool read_tdt_token_ids(const char* path, std::vector<int32_t>& out, bool& present) {
    out.clear();
    present = false;
    ggml_context* ctx = nullptr;
    gguf_init_params p{ /*no_alloc=*/false, &ctx };
    gguf_context* g = gguf_init_from_file(path, p);
    if (!g) { std::fprintf(stderr, "[tdt] open baseline failed: %s\n", path); return false; }

    // Length KV (always written) lets us tell "empty" from "missing".
    int64_t key = gguf_find_key(g, "baseline.tdt_token_count");
    uint32_t count = 0;
    if (key >= 0) count = gguf_get_val_u32(g, key);

    ggml_tensor* t = ggml_get_tensor(ctx, "tdt_token_ids");
    if (!t) {
        gguf_free(g); ggml_free(ctx);
        present = (count > 0);   // present-but-missing-tensor would be a bug
        return true;             // count==0 -> legitimately empty
    }
    const int n = (int)ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_I32) {
        std::memcpy(out.data(), t->data, (size_t)n * sizeof(int32_t));
    } else if (t->type == GGML_TYPE_F32) {
        const float* d = (const float*)t->data;
        for (int i = 0; i < n; ++i) out[i] = (int32_t)d[i];
    } else {
        std::fprintf(stderr, "[tdt] tdt_token_ids unexpected type %d\n", (int)t->type);
        gguf_free(g); ggml_free(ctx);
        return false;
    }
    present = true;
    gguf_free(g); ggml_free(ctx);
    return true;
}

int main() {
    const char* gguf_path = std::getenv("PARAKEET_TEST_GGUF");
    const char* base_path = std::getenv("PARAKEET_TEST_BASELINE_SPEECH");
    if (!gguf_path || !base_path) {
        std::fprintf(stderr, "test_tdt_greedy: PARAKEET_TEST_GGUF / "
                             "PARAKEET_TEST_BASELINE_SPEECH not set; skip\n");
        return 77;
    }

    try {
        pk::ModelLoader ml;
        if (!ml.load(gguf_path)) { std::fprintf(stderr, "[tdt] model load failed\n"); return 1; }
        const pk::ParakeetConfig& cfg = ml.config();

        const std::vector<int32_t>& durations = cfg.tdt_durations;
        const int blank_id = (int)cfg.blank_id;
        const int max_symbols = 10;  // NeMo's effective default for this model.
        if (durations.empty()) {
            std::fprintf(stderr, "[tdt] config has no tdt_durations\n");
            return 1;
        }
        std::fprintf(stderr, "[tdt] durations=");
        for (int32_t d : durations) std::fprintf(stderr, "%d ", d);
        std::fprintf(stderr, "| blank_id=%d | max_symbols=%d\n", blank_id, max_symbols);

        // ---- Audio -> mel -> encoder ----
        pk::Audio audio;
        if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio)) {
            std::fprintf(stderr, "[tdt] failed to load speech.wav\n");
            return 1;
        }
        pk::MelFrontend mel(ml);
        std::vector<float> feats; int n_mels = 0, T = 0;
        mel.compute(audio.samples, feats, n_mels, T);

        pk::Encoder encoder(ml);
        std::vector<float> enc_out; int d_model = 0, Tout = 0;
        encoder.forward(feats, n_mels, T, enc_out, d_model, Tout);
        std::fprintf(stderr, "[tdt] encoder_out: d_model=%d Tout=%d\n", d_model, Tout);

        // Encoder outputs channels-first [d_model, Tout] (enc_out[c*Tout + t]);
        // tdt_greedy wants row-major [Tout, d_model] (enc[t*d_model + c]).
        std::vector<float> enc((size_t)Tout * d_model);
        for (int t = 0; t < Tout; ++t)
            for (int c = 0; c < d_model; ++c)
                enc[(size_t)t * d_model + c] = enc_out[(size_t)c * Tout + t];

        // ---- TDT greedy ----
        pk::PredictionNet prednet(ml);
        pk::Joint joint(ml);
        std::vector<int32_t> got = pk::tdt_greedy(prednet, joint, enc, Tout, d_model,
                                                  durations, blank_id, max_symbols);

        // ---- Reference ----
        std::vector<int32_t> ref; bool ref_present = false;
        if (!read_tdt_token_ids(base_path, ref, ref_present)) return 1;

        std::fprintf(stderr, "[tdt] got (%zu):", got.size());
        for (int32_t v : got) std::fprintf(stderr, " %d", v);
        std::fprintf(stderr, "\n[tdt] ref (%zu):", ref.size());
        for (int32_t v : ref) std::fprintf(stderr, " %d", v);
        std::fprintf(stderr, "\n");

        if (got.size() != ref.size()) {
            std::fprintf(stderr, "[tdt] LENGTH MISMATCH got=%zu ref=%zu\n",
                         got.size(), ref.size());
            return 1;
        }
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] != ref[i]) {
                std::fprintf(stderr, "[tdt] FIRST DIVERGENCE at index %zu: got=%d ref=%d\n",
                             i, got[i], ref[i]);
                return 1;
            }
        }

        std::fprintf(stderr, "[tdt] PASS — %zu token ids match NeMo exactly\n", got.size());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tdt] threw: %s\n", e.what());
        return 1;
    }
}
