#include "prediction.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstring>

// Parity test for the stateful single-step prediction net API.
//
// We step through the same sequence that test_prediction exercises via forward():
//   [SOS, ids[0], ids[1], ids[2], ids[3]]  (5 steps, first with is_sos=true)
//
// Each step produces a g=[hidden] output vector; stacked they form [5, hidden].
// This must equal baseline `pred_out` (atol/rtol 2e-3), proving the stateful
// step API reproduces the full-sequence output exactly.
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE");
    if (!gguf || !base) {
        std::fprintf(stderr, "env not set; skip\n");
        return 77;
    }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) return 1;

    // ---- Load pred_input_ids (int32) from baseline ----
    std::vector<int32_t> ids;
    {
        ggml_context* ctx = nullptr;
        gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx };
        gguf_context* g = gguf_init_from_file(base, p);
        if (!g) { std::fprintf(stderr, "[prediction_step] open baseline failed\n"); return 1; }
        ggml_tensor* t = ggml_get_tensor(ctx, "pred_input_ids");
        if (!t) { std::fprintf(stderr, "[prediction_step] pred_input_ids missing\n"); gguf_free(g); ggml_free(ctx); return 1; }
        const int n = (int)ggml_nelements(t);
        ids.resize(n);
        if (t->type == GGML_TYPE_I32) {
            std::memcpy(ids.data(), t->data, (size_t)n * sizeof(int32_t));
        } else if (t->type == GGML_TYPE_F32) {
            const float* d = (const float*)t->data;
            for (int i = 0; i < n; ++i) ids[i] = (int32_t)d[i];
        } else {
            std::fprintf(stderr, "[prediction_step] pred_input_ids unexpected type\n");
            gguf_free(g); ggml_free(ctx); return 1;
        }
        gguf_free(g); ggml_free(ctx);
    }
    std::fprintf(stderr, "[prediction_step] ids (U=%zu):", ids.size());
    for (int32_t v : ids) std::fprintf(stderr, " %d", v);
    std::fprintf(stderr, "\n");

    // ---- Step through [SOS, ids[0], ..., ids[U-1]] using the stateful API ----
    pk::PredictionNet pred(ml);
    const int H = pred.hidden_size();
    const int U = (int)ids.size();
    const int steps = U + 1; // SOS + U tokens

    std::vector<float> out((size_t)steps * H);
    pk::PredState state = pred.zero_state();

    for (int s = 0; s < steps; ++s) {
        bool is_sos = (s == 0);
        int32_t token_id = is_sos ? 0 : ids[s - 1];

        std::vector<float> g;
        pk::PredState next_state;
        pred.step(token_id, is_sos, state, g, next_state);

        std::memcpy(&out[(size_t)s * H], g.data(), (size_t)H * sizeof(float));
        state = std::move(next_state);
    }

    std::fprintf(stderr, "[prediction_step] output: steps=%d hidden=%d (n=%zu)\n",
                 steps, H, out.size());

    // ---- Load reference pred_out ----
    std::vector<float> ref; std::vector<int64_t> rshape;
    if (!pktest::load_baseline(base, "pred_out", ref, rshape)) return 1;
    if (rshape.size() != 2) {
        std::fprintf(stderr, "[prediction_step] pred_out rank=%zu\n", rshape.size());
        return 1;
    }
    const int ref_U = (int)rshape[0];
    const int ref_H = (int)rshape[1];
    std::fprintf(stderr, "[prediction_step] ref pred_out: U=%d H=%d (n=%zu)\n",
                 ref_U, ref_H, ref.size());

    if (ref_U != steps || ref_H != H) {
        std::fprintf(stderr, "[prediction_step] shape mismatch: got [%d,%d] ref [%d,%d]\n",
                     steps, H, ref_U, ref_H);
        return 1;
    }

    bool ok = pktest::compare(out, ref, "prediction_step", /*atol=*/2e-3f, /*rtol=*/2e-3f);
    return ok ? 0 : 1;
}
