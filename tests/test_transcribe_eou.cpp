#include "parakeet.h"
#include "model_loader.hpp"
#include "audio_io.hpp"
#include "mel.hpp"
#include "encoder.hpp"
#include "prediction.hpp"
#include "joint.hpp"
#include "rnnt.hpp"
#include "tokenizer.hpp"
#include "parity.hpp"
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>
#include <cstdint>

// Offline end-to-end transcription regression test for the streaming EOU model
// nvidia/parakeet_realtime_eou_120m-v1 (arch=rnnt, vocab=1026, blank=1026).
//
// This is the "5a milestone" test: the full offline pipeline (mel → encoder with
// layer_norm conv + causal conv + chunked-limited attention → rnnt_greedy →
// detokenize) must produce the NeMo reference transcript EXACTLY (WER 0), including
// the literal "<EOU>" token the transducer emits at the end of the utterance.
//
// Two assertions:
//   1. Text: pk::transcribe(eou.gguf, speech.wav, kTDT) == baseline.rnnt_text
//   2. Token ids: the raw rnnt_greedy token id sequence == rnnt_token_ids tensor
//      from the baseline GGUF (int32, 45 tokens, last = 1024 = <EOU>).
//
// Model facts:
//   vocab_size = 1026  (<EOU>=1024, <EOB>=1025, blank not in vocab)
//   blank_id   = 1026  (one past the last vocab index; V_plus = 1027)
//   tdt_durations = [] (pure RNNT, no duration table → routes to rnnt_greedy)
//
// The test skips (exit 77) unless BOTH env vars are set, so the default CI suite
// stays green without the ~480MB streaming model GGUF.
//
// Env:
//   PARAKEET_TEST_GGUF_EOU      path to the converted eou GGUF (skip 77 if unset)
//   PARAKEET_TEST_BASELINE_EOU  path to the baseline GGUF with rnnt_token_ids +
//                               baseline.rnnt_text KV (skip 77 if unset)
//
// LABEL model
// WORKING_DIRECTORY (run from project root; wav path is relative)

// NeMo RNNT reference for nvidia/parakeet_realtime_eou_120m-v1 on speech.wav.
// Pure-RNNT EOU model emits lowercase, unpunctuated text, with the literal
// "<EOU>" token appended (the transducer emits token id 1024 = <EOU>).
// Validated against NeMo offline transcription in Phase 5 Task 1.
static const char* kRefEOU =
    "well i don't wish to see it any more observed phoebe turning away her "
    "eyes it is certainly very like the old portrait<EOU>";

// NeMo reference token id sequence (45 tokens, last = 1024 = <EOU>).
// Read from baseline GGUF if available; this hardcoded array is the fallback
// for verifying the baseline content is as expected.
static const int32_t kRefTokenIds[] = {
    51, 33, 4, 180, 1019, 998, 8, 292, 24, 214, 47, 244, 113, 7, 436, 880,
    415, 389, 999, 997, 352, 1, 85, 372, 3, 313, 302, 819, 29, 47, 56, 627,
    435, 78, 996, 159, 86, 71, 5, 611, 23, 174, 138, 36, 1024
};
static const int kRefTokenCount = 45;

int main() {
    const char* gguf_path = std::getenv("PARAKEET_TEST_GGUF_EOU");
    const char* base_path = std::getenv("PARAKEET_TEST_BASELINE_EOU");
    if (!gguf_path || !base_path) {
        std::fprintf(stderr,
            "test_transcribe_eou: PARAKEET_TEST_GGUF_EOU and/or "
            "PARAKEET_TEST_BASELINE_EOU not set; skip (streaming EOU model is "
            "a ~480MB download, not in CI)\n");
        return 77;
    }

    // --- Load baseline reference text + token ids ---
    std::string ref_text;
    if (!pktest::load_kv_str(base_path, "baseline.rnnt_text", ref_text)) {
        // Fall back to the hardcoded reference
        ref_text = std::string(kRefEOU);
        std::fprintf(stderr,
            "test_transcribe_eou: baseline.rnnt_text not found; using hardcoded ref\n");
    }

    std::vector<int32_t> ref_ids;
    if (!pktest::load_baseline_i32(base_path, "rnnt_token_ids", ref_ids)) {
        // Fall back to hardcoded
        ref_ids.assign(kRefTokenIds, kRefTokenIds + kRefTokenCount);
        std::fprintf(stderr,
            "test_transcribe_eou: rnnt_token_ids not found; using hardcoded ref\n");
    }

    // Verify the hardcoded reference matches the baseline (cross-check)
    {
        const std::string hardcoded_text(kRefEOU);
        if (ref_text != hardcoded_text) {
            std::fprintf(stderr,
                "test_transcribe_eou: WARNING: baseline.rnnt_text differs from "
                "hardcoded reference\n  baseline: %s\n  hardcoded: %s\n",
                ref_text.c_str(), hardcoded_text.c_str());
        }
    }

    std::fprintf(stderr, "test_transcribe_eou: ref_text = %s\n", ref_text.c_str());
    std::fprintf(stderr, "test_transcribe_eou: ref_ids  = [");
    for (int i = 0; i < (int)ref_ids.size(); ++i)
        std::fprintf(stderr, "%s%d", i ? "," : "", ref_ids[i]);
    std::fprintf(stderr, "]\n");

    // --- Part 1: End-to-end text via pk::transcribe ---
    std::string got_text;
    try {
        // kTDT selects the transducer head; routes to rnnt_greedy because
        // parakeet_realtime_eou_120m-v1 has no TDT duration table.
        got_text = pk::transcribe(gguf_path, "tests/fixtures/speech.wav",
                                  pk::Decoder::kTDT);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_transcribe_eou: pk::transcribe threw: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "test_transcribe_eou: got_text = %s\n", got_text.c_str());

    bool text_ok = (got_text == ref_text);
    if (!text_ok) {
        std::fprintf(stderr,
            "test_transcribe_eou: TEXT MISMATCH vs NeMo reference\n"
            "  got:      %s\n"
            "  expected: %s\n",
            got_text.c_str(), ref_text.c_str());
    }

    // --- Part 2: Token id sequence via the low-level pipeline ---
    // Run the pipeline manually to capture the raw token id sequence, since
    // pk::transcribe only returns the text string.
    std::vector<int32_t> got_ids;
    try {
        pk::ModelLoader ml;
        if (!ml.load(gguf_path)) {
            std::fprintf(stderr, "test_transcribe_eou: failed to load model %s\n", gguf_path);
            return 1;
        }
        const pk::ParakeetConfig& cfg = ml.config();

        // Load and process audio
        pk::Audio audio;
        if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio)) {
            std::fprintf(stderr, "test_transcribe_eou: failed to load audio\n");
            return 1;
        }

        // Mel front end
        pk::MelFrontend mel_fe(ml);
        std::vector<float> feats;
        int n_mels = 0, T = 0;
        mel_fe.compute(audio.samples, feats, n_mels, T);

        // Encoder
        pk::Encoder encoder(ml);
        std::vector<float> enc_out;
        int d_model = 0, Tout = 0;
        encoder.forward(feats, n_mels, T, enc_out, d_model, Tout);

        // Transpose encoder output from [d_model, Tout] to [Tout, d_model]
        std::vector<float> enc_row(static_cast<size_t>(Tout) * d_model);
        for (int t = 0; t < Tout; ++t)
            for (int c = 0; c < d_model; ++c)
                enc_row[t * d_model + c] = enc_out[c * Tout + t];

        // RNNT greedy decode (no duration table → rnnt_greedy)
        pk::PredictionNet pred(ml);
        pk::Joint        joint(ml);
        const int max_symbols = 10;
        got_ids = pk::rnnt_greedy(
            pred, joint, enc_row, Tout, d_model,
            static_cast<int>(cfg.blank_id), max_symbols);

    } catch (const std::exception& e) {
        std::fprintf(stderr,
            "test_transcribe_eou: pipeline threw: %s\n", e.what());
        return 1;
    }

    std::fprintf(stderr, "test_transcribe_eou: got_ids  = [");
    for (int i = 0; i < (int)got_ids.size(); ++i)
        std::fprintf(stderr, "%s%d", i ? "," : "", got_ids[i]);
    std::fprintf(stderr, "]\n");

    bool ids_ok = (got_ids == ref_ids);
    if (!ids_ok) {
        std::fprintf(stderr,
            "test_transcribe_eou: TOKEN ID MISMATCH vs NeMo reference\n"
            "  got_count=%zu ref_count=%zu\n",
            got_ids.size(), ref_ids.size());
        // Show first divergence
        size_t minlen = std::min(got_ids.size(), ref_ids.size());
        for (size_t i = 0; i < minlen; ++i) {
            if (got_ids[i] != ref_ids[i]) {
                std::fprintf(stderr, "  first diff at index %zu: got=%d ref=%d\n",
                             i, got_ids[i], ref_ids[i]);
                break;
            }
        }
    }

    if (text_ok && ids_ok) {
        std::fprintf(stderr,
            "test_transcribe_eou: PASS — text + token ids match NeMo reference "
            "(WER 0, 5a milestone)\n"
            "  blank_id=%d, V_plus=1027, vocab=1026, <EOU>=1024, <EOB>=1025\n",
            1026);
        return 0;
    }
    return 1;
}
