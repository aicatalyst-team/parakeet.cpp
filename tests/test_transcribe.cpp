#include "parakeet.h"
#include "search.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include "gguf.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// End-to-end CTC transcription test.
//
//   1. pk::transcribe(model, clip.wav) == baseline.ctc_text (NeMo's authoritative
//      CTC greedy transcript of the clip). For the synthetic-tone clip both are
//      expected to be the empty string "" — all frames decode to blank — but the
//      assertion validates the full wiring runs and agrees with NeMo.
//
//   2. Deterministic collapse check: feed the baseline `ctc_argmax_ids` (the raw
//      per-frame argmax ids from NeMo) through pk::ctc_greedy as one-hot logits
//      and assert the collapsed/blank-stripped id sequence matches what we expect
//      (empty for the all-blank tone clip). This exercises the collapse rule
//      independent of the encoder/CTC numerics.
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE");
    if (!gguf || !base) {
        std::fprintf(stderr, "test_transcribe: env not set; skip\n");
        return 77;
    }

    // ---- Load model config (for blank_id / vocab) ----
    pk::ModelLoader ml;
    if (!ml.load(gguf)) {
        std::fprintf(stderr, "test_transcribe: failed to load model %s\n", gguf);
        return 1;
    }
    const int blank_id     = (int)ml.config().blank_id;
    const int vocab_plus_1 = (int)ml.config().vocab_size + 1;
    std::fprintf(stderr, "test_transcribe: blank_id=%d vocab+1=%d\n", blank_id, vocab_plus_1);

    // ---- Read baseline.ctc_text KV + ctc_argmax_ids tensor ----
    ggml_context* bctx = nullptr;
    gguf_init_params bp{ false, &bctx };
    gguf_context* bg = gguf_init_from_file(base, bp);
    if (!bg) {
        std::fprintf(stderr, "test_transcribe: failed to open baseline %s\n", base);
        return 1;
    }
    // The gguf writer silently drops empty-string KVs, so a missing
    // baseline.ctc_text means NeMo's CTC transcript of the clip was "" (the
    // expected case for the synthetic-tone fixture). Treat absent as empty.
    int64_t key = gguf_find_key(bg, "baseline.ctc_text");
    std::string expected_text = (key < 0) ? std::string() : gguf_get_val_str(bg, key);

    ggml_tensor* id_t = ggml_get_tensor(bctx, "ctc_argmax_ids");
    if (!id_t) {
        std::fprintf(stderr, "test_transcribe: 'ctc_argmax_ids' tensor not found\n");
        gguf_free(bg); ggml_free(bctx);
        return 1;
    }
    size_t T = (size_t)ggml_nelements(id_t);
    std::vector<int32_t> argmax_ids(T);
    std::memcpy(argmax_ids.data(), id_t->data, T * sizeof(int32_t));
    gguf_free(bg); ggml_free(bctx);

    std::fprintf(stderr, "test_transcribe: baseline.ctc_text=%s\n", expected_text.c_str());
    std::fprintf(stderr, "test_transcribe: ctc_argmax_ids T=%zu\n", T);

    // ---- (1) Full pipeline ----
    std::string got;
    try {
        got = pk::transcribe(gguf, "tests/fixtures/clip.wav");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_transcribe: pk::transcribe threw: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "test_transcribe: pk::transcribe=%s\n", got.c_str());
    if (got != expected_text) {
        std::fprintf(stderr,
            "test_transcribe: MISMATCH (full pipeline)\n  got:      %s\n  expected: %s\n",
            got.c_str(), expected_text.c_str());
        return 1;
    }

    // ---- (2) Deterministic collapse of baseline ctc_argmax_ids ----
    // Build one-hot logits [T, vocab+1] with a large value at each frame's id,
    // then collapse. The blank id (== vocab_size) sits at the last column.
    std::vector<float> onehot((size_t)T * vocab_plus_1, 0.0f);
    for (size_t t = 0; t < T; ++t) {
        int32_t id = argmax_ids[t];
        if (id < 0 || id >= vocab_plus_1) {
            std::fprintf(stderr, "test_transcribe: argmax id %d out of range\n", id);
            return 1;
        }
        onehot[t * vocab_plus_1 + id] = 10.0f;
    }
    std::vector<int32_t> collapsed =
        pk::ctc_greedy(onehot, (int)T, vocab_plus_1, blank_id);

    // Expected = NeMo's fold_consecutive collapse: drop blanks + consecutive dups.
    std::vector<int32_t> expected_collapsed;
    int32_t prev = blank_id;
    for (size_t t = 0; t < T; ++t) {
        int32_t p = argmax_ids[t];
        if ((p != prev || prev == blank_id) && p != blank_id)
            expected_collapsed.push_back(p);
        prev = p;
    }

    if (collapsed != expected_collapsed) {
        std::fprintf(stderr, "test_transcribe: collapse MISMATCH (n got=%zu exp=%zu)\n",
                     collapsed.size(), expected_collapsed.size());
        return 1;
    }
    std::fprintf(stderr, "test_transcribe: collapse OK (n=%zu, %s)\n",
                 collapsed.size(), collapsed.empty() ? "empty" : "non-empty");

    std::fprintf(stderr, "test_transcribe: PASS (text=%s)\n",
                 got.empty() ? "<empty>" : got.c_str());
    return 0;
}
