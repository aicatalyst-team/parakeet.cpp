#include "relpos_attention.hpp"
#include "model_loader.hpp"
#include "graph_builder.hpp"
#include "ggml_graph.hpp"
#include "backend.hpp"
#include "ggml.h"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstddef>

// Batched-attention isolation test for RelPosAttention::build_graph_batched
// (the 4D rel-shift added in T4.1). It empirically validates the riskiest code
// path in the batched encoder by comparing per-item batched output slices
// against the scalar RelPosAttention::forward reference.
//
// Input source: the SAME real baseline tensors the scalar test uses
// ("l0_attn_in" = [T, d_model] row-major, "pos_emb" = [2T-1, d_model] row-major),
// so the batched path is exercised on the exact distribution NeMo produces.
//
// Layout mapping (load-bearing):
//   scalar x   : row-major [T, D]      -> x[t*D + c]
//   batched xt : ggml [D, T, B]        -> data index (b*T + t)*D + c
//   pe         : ggml [D, pos_len]     -> pe[p*D + c]  (shared across batch)
//   out        : ggml [D, T, B]        -> out[(b*T + t)*D + c]
//
// Case A (equal-length B=2, identical clips): item0 == item1 == scalar ref.
//   Proves the 4D rel-shift is correct under batching.
// Case B (ragged + padding invariance): item0 full (valid=T), item1 short
//   (valid=T/2). item1's valid rows match the scalar short-clip ref; item0 is
//   UNCHANGED vs the scalar full ref (no leakage from the shorter neighbor).

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE");
    if (!gguf || !base) { std::fprintf(stderr, "env not set; skip\n"); return 77; }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) return 1;

    // Attention input: baseline "l0_attn_in" is [T, d_model] row-major.
    std::vector<float> x; std::vector<int64_t> xshape;
    if (!pktest::load_baseline(base, "l0_attn_in", x, xshape)) return 1;
    if (xshape.size() != 2) { std::fprintf(stderr, "l0_attn_in rank=%zu\n", xshape.size()); return 1; }
    const int T = (int)xshape[0];
    const int D = (int)xshape[1];

    // Relative positional encoding: baseline "pos_emb" is [2T-1, d_model] row-major.
    std::vector<float> pe; std::vector<int64_t> pshape;
    if (!pktest::load_baseline(base, "pos_emb", pe, pshape)) return 1;
    if (pshape.size() != 2) { std::fprintf(stderr, "pos_emb rank=%zu\n", pshape.size()); return 1; }
    const int pos_len = (int)pshape[0];
    if (pos_len != 2 * T - 1 || (int)pshape[1] != D) {
        std::fprintf(stderr, "pos_emb shape=[%d,%lld] expected [%d,%d]\n",
                     pos_len, (long long)pshape[1], 2 * T - 1, D);
        return 1;
    }

    pk::RelPosAttention attn(ml, /*layer_idx*/0);

    // Helper: run build_graph_batched for B items, all sharing the same input x
    // (row-major [T,D]) and pe, with per-item valid_len. Returns [D,T,B] row-major.
    auto run_batched = [&](int B, const std::vector<int>& valid_len,
                           std::vector<float>& out) -> bool {
        pk::GraphInputPool pool;
        // Build the [D,T,B] host buffer: xb[(b*T+t)*D + c] = x[t*D + c].
        std::vector<float> xb((size_t)B * T * D);
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                for (int c = 0; c < D; ++c)
                    xb[((size_t)(b * T + t)) * D + c] = x[(size_t)t * D + c];
        return pk::run_graph(/*mem_bytes*/0, /*n_threads*/4,
            [&](ggml_context* ctx) -> ggml_tensor* {
                int64_t xt_ne[3] = {D, T, B};
                ggml_tensor* xt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, xt_ne,
                                      xb.data(), (size_t)B * T * D * sizeof(float));
                int64_t pe_ne[2] = {D, pos_len};
                ggml_tensor* pet = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, pe_ne,
                                      pe.data(), (size_t)pos_len * D * sizeof(float));
                return attn.build_graph_batched(ctx, xt, T, B, pet, pos_len,
                                                valid_len, pool);
            }, out);
    };

    // Extract item b's [T,D] (row-major) from a [D,T,B] row-major buffer.
    auto slice_item = [&](const std::vector<float>& out, int b) {
        std::vector<float> item((size_t)T * D);
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < D; ++c)
                item[(size_t)t * D + c] = out[((size_t)(b * T + t)) * D + c];
        return item;
    };

    bool ok = true;

    // ---- Case A: equal-length B=2, identical clips ----
    {
        std::vector<float> ref;
        attn.forward(x, T, pe, pos_len, /*valid_len*/T, ref);

        std::vector<float> out;
        if (!run_batched(2, {T, T}, out)) {
            std::fprintf(stderr, "[caseA] run_graph failed\n");
            return 1;
        }
        std::vector<float> item0 = slice_item(out, 0);
        std::vector<float> item1 = slice_item(out, 1);
        bool a0 = pktest::compare(item0, ref, "caseA_item0", 1e-3f, 1e-3f);
        bool a1 = pktest::compare(item1, ref, "caseA_item1", 1e-3f, 1e-3f);
        ok = ok && a0 && a1;
    }

    // ---- Case B: ragged + padding invariance ----
    {
        const int v0 = T;
        const int v1 = T / 2;

        std::vector<float> ref0;
        attn.forward(x, T, pe, pos_len, /*valid_len*/v0, ref0);
        std::vector<float> ref1;
        attn.forward(x, T, pe, pos_len, /*valid_len*/v1, ref1);

        std::vector<float> out;
        if (!run_batched(2, {v0, v1}, out)) {
            std::fprintf(stderr, "[caseB] run_graph failed\n");
            return 1;
        }
        std::vector<float> item0 = slice_item(out, 0);
        std::vector<float> item1 = slice_item(out, 1);

        // item0 (full) must be UNCHANGED vs the full scalar ref (padding invariance).
        bool b0 = pktest::compare(item0, ref0, "caseB_item0", 1e-3f, 1e-3f);

        // item1's first v1 query rows must match ref1's first v1 rows.
        std::vector<float> item1_valid((size_t)v1 * D);
        std::vector<float> ref1_valid((size_t)v1 * D);
        for (int t = 0; t < v1; ++t)
            for (int c = 0; c < D; ++c) {
                item1_valid[(size_t)t * D + c] = item1[(size_t)t * D + c];
                ref1_valid[(size_t)t * D + c]  = ref1[(size_t)t * D + c];
            }
        bool b1 = pktest::compare(item1_valid, ref1_valid, "caseB_item1", 1e-3f, 1e-3f);
        ok = ok && b0 && b1;
    }

    return ok ? 0 : 1;
}
