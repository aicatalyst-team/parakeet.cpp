#include "ctc_decoder.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>

// Parity test for the CTC head vs NeMo ConvASRDecoder.
//
// Input:  baseline `encoder_out` [d_model=512, T'=26] (channels-first row-major)
// Output: baseline `ctc_logits`  [T'=26, vocab+1=1025] (post-log_softmax)
//
// The NeMo ConvASRDecoder applies:
//   Conv1d(d_model, vocab+1, kernel=1)  i.e. a linear map per time step
//   followed by log_softmax(dim=-1)     over the vocab axis
//
// We verify exp(row).sum() ≈ 1 for each row of the baseline (confirming it is
// post-log_softmax), then compare our output to it with atol/rtol 5e-2.
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE");
    if (!gguf || !base) {
        std::fprintf(stderr, "env not set; skip\n");
        return 77;
    }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) return 1;

    // ---- Load encoder_out from baseline ----
    // baseline encoder_out: ggml ne[0]=T (fastest), ne[1]=d_model
    // → parity.hpp shape = [ne[1], ne[0]] = [d_model, T]
    // → memory enc[c*T + t] (row-major [d_model, T]) ✓
    std::vector<float> enc; std::vector<int64_t> eshape;
    if (!pktest::load_baseline(base, "encoder_out", enc, eshape)) return 1;
    if (eshape.size() != 2) {
        std::fprintf(stderr, "encoder_out rank=%zu\n", eshape.size());
        return 1;
    }
    const int d_model = (int)eshape[0];  // ne[1] in ggml, outermost = d_model
    const int T       = (int)eshape[1];  // ne[0] in ggml, innermost = T

    std::fprintf(stderr, "[ctc] encoder_out: d_model=%d T=%d (n=%zu)\n",
                 d_model, T, enc.size());

    // ---- Run CTC decoder ----
    pk::CTCDecoder ctc(ml);
    std::vector<float> logits; int vocab_plus_1 = 0;
    ctc.forward(enc, d_model, T, logits, vocab_plus_1);

    std::fprintf(stderr, "[ctc] output: T=%d vocab+1=%d (n=%zu)\n",
                 T, vocab_plus_1, logits.size());

    // ---- Sanity check: exp(row).sum() ≈ 1 for our output (confirms log_softmax) ----
    {
        double min_sum = 1e9, max_sum = -1e9;
        for (int t = 0; t < T; ++t) {
            double s = 0.0;
            for (int v = 0; v < vocab_plus_1; ++v)
                s += std::exp((double)logits[t * vocab_plus_1 + v]);
            if (s < min_sum) min_sum = s;
            if (s > max_sum) max_sum = s;
        }
        std::fprintf(stderr, "[ctc] exp-row-sum: min=%.6f max=%.6f (expect ≈1)\n",
                     min_sum, max_sum);
    }

    // ---- Load reference ctc_logits ----
    // baseline ctc_logits: ggml ne[0]=V (fastest), ne[1]=T
    // → parity.hpp shape = [ne[1], ne[0]] = [T, V]
    // → memory ref[t*V + v] (row-major [T, V]) ✓ same layout as our logits
    std::vector<float> ref; std::vector<int64_t> rshape;
    if (!pktest::load_baseline(base, "ctc_logits", ref, rshape)) return 1;
    if (rshape.size() != 2) {
        std::fprintf(stderr, "ctc_logits rank=%zu\n", rshape.size());
        return 1;
    }
    const int ref_T = (int)rshape[0];
    const int ref_V = (int)rshape[1];
    std::fprintf(stderr, "[ctc] ref ctc_logits: T=%d V=%d (n=%zu)\n",
                 ref_T, ref_V, ref.size());

    if (ref_T != T || ref_V != vocab_plus_1) {
        std::fprintf(stderr, "[ctc] shape mismatch: got [%d,%d] ref [%d,%d]\n",
                     T, vocab_plus_1, ref_T, ref_V);
        return 1;
    }

    bool ok = pktest::compare(logits, ref, "ctc", /*atol=*/5e-2f, /*rtol=*/5e-2f);
    return ok ? 0 : 1;
}
