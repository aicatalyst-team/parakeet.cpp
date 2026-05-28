#include "prediction.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>

// Parity test for the RNN-Transducer prediction net vs NeMo RNNTDecoder.predict.
//
// Input:  baseline `pred_input_ids` int32 [U=4] = [120, 7, 300, 42]
// Output: baseline `pred_out`       f32  [U+1=5, pred_hidden=640]
//
// NeMo's predict(y, add_sos=True) prepends a zero "start of sequence" vector
// (a literal zero [H] vector, independent of the embedding padding_idx) so the
// output sequence has U+1 hidden states: [h(SOS), h(id0), h(id1), h(id2), h(id3)].
//
// The prediction net is: embedding lookup + single-layer LSTM (PyTorch gate
// order [input, forget, cell, output]). We run forward with add_sos=true and
// compare to the baseline within atol/rtol 2e-3.
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
    // parity.hpp loads f32; the ids are stored as int32 so we read raw bytes
    // via the gguf API directly here.
    std::vector<int32_t> ids;
    {
        ggml_context* ctx = nullptr;
        gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx };
        gguf_context* g = gguf_init_from_file(base, p);
        if (!g) { std::fprintf(stderr, "[prediction] open baseline failed\n"); return 1; }
        ggml_tensor* t = ggml_get_tensor(ctx, "pred_input_ids");
        if (!t) { std::fprintf(stderr, "[prediction] pred_input_ids missing\n"); gguf_free(g); ggml_free(ctx); return 1; }
        const int n = (int)ggml_nelements(t);
        ids.resize(n);
        if (t->type == GGML_TYPE_I32) {
            std::memcpy(ids.data(), t->data, (size_t)n * sizeof(int32_t));
        } else if (t->type == GGML_TYPE_F32) {
            const float* d = (const float*)t->data;
            for (int i = 0; i < n; ++i) ids[i] = (int32_t)d[i];
        } else {
            std::fprintf(stderr, "[prediction] pred_input_ids unexpected type\n");
            gguf_free(g); ggml_free(ctx); return 1;
        }
        gguf_free(g); ggml_free(ctx);
    }
    std::fprintf(stderr, "[prediction] ids (U=%zu):", ids.size());
    for (int32_t v : ids) std::fprintf(stderr, " %d", v);
    std::fprintf(stderr, "\n");

    // ---- Run prediction net with add_sos=true ----
    pk::PredictionNet pred(ml);
    std::vector<float> out; int U_out = 0, hidden = 0;
    pred.forward(ids, /*add_sos=*/true, out, U_out, hidden);
    std::fprintf(stderr, "[prediction] output: U_out=%d hidden=%d (n=%zu)\n",
                 U_out, hidden, out.size());

    // ---- Load reference pred_out ----
    // baseline pred_out: ggml ne[0]=H (fastest), ne[1]=U+1
    // → parity.hpp shape = [ne[1], ne[0]] = [U+1, H], memory ref[u*H + h] (row-major)
    std::vector<float> ref; std::vector<int64_t> rshape;
    if (!pktest::load_baseline(base, "pred_out", ref, rshape)) return 1;
    if (rshape.size() != 2) { std::fprintf(stderr, "[prediction] pred_out rank=%zu\n", rshape.size()); return 1; }
    const int ref_U = (int)rshape[0];
    const int ref_H = (int)rshape[1];
    std::fprintf(stderr, "[prediction] ref pred_out: U=%d H=%d (n=%zu)\n", ref_U, ref_H, ref.size());

    if (ref_U != U_out || ref_H != hidden) {
        std::fprintf(stderr, "[prediction] shape mismatch: got [%d,%d] ref [%d,%d]\n",
                     U_out, hidden, ref_U, ref_H);
        return 1;
    }

    bool ok = pktest::compare(out, ref, "prediction", /*atol=*/2e-3f, /*rtol=*/2e-3f);
    return ok ? 0 : 1;
}
