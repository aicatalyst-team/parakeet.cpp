#include "tdt.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>

namespace pk {

namespace {
// argmax over a[0..n) returning the first index of the maximum value.
// torch.max(dim) returns the FIRST max index on ties; match that.
int argmax(const float* a, int n) {
    int best = 0;
    float bv = a[0];
    for (int i = 1; i < n; ++i) {
        if (a[i] > bv) { bv = a[i]; best = i; }
    }
    return best;
}
} // namespace

std::vector<int32_t> tdt_greedy(const PredictionNet& pred, const Joint& joint,
                                const std::vector<float>& enc, int T, int enc_hidden,
                                const std::vector<int32_t>& durations,
                                int blank_id, int max_symbols) {
    assert((int)enc.size() == (size_t)T * enc_hidden);
    assert(!durations.empty());

    const int V_plus       = joint.V_plus();
    const int num_dur      = (int)durations.size();
    const int token_count  = V_plus - num_dur;   // vocab + 1 (incl. blank) = 1025
    assert(token_count == joint.vocab_size() + 1);
    assert(num_dur == joint.num_durations());

    std::vector<int32_t> hyp;

    // Committed (non-blank) decoding state and last emitted token.
    PredState committed = pred.zero_state();
    int32_t last_token = -1;      // -1 sentinel: nothing emitted yet -> SOS.
    bool emitted_any = false;

    // Scratch reused across inner steps.
    std::vector<float> g;
    PredState out_state;
    std::vector<float> logits;
    int v_out = 0;
    std::vector<float> enc_frame((size_t)enc_hidden);

    int t = 0;
    while (t < T) {
        int symbols_added = 0;
        bool need_loop = true;
        int skip = 0;

        while (need_loop && symbols_added < max_symbols) {
            // First step (no token emitted, no committed state) uses SOS;
            // otherwise feed the last EMITTED token.
            const bool is_sos = !emitted_any;
            const int32_t last_label = emitted_any ? last_token : blank_id;

            // Prediction net single step from the committed state.
            pred.step(last_label, is_sos, committed, g, out_state);

            // Joint: enc[t] (T=1) x g (U=1) -> raw logits [V_plus].
            std::memcpy(enc_frame.data(), &enc[(size_t)t * enc_hidden],
                        (size_t)enc_hidden * sizeof(float));
            joint.forward(enc_frame, /*T=*/1, enc_hidden,
                          g, /*U=*/1, (int)g.size(), logits, v_out);
            assert(v_out == V_plus);

            // Split: token logits [0, token_count), duration logits [token_count, V_plus).
            const int k   = argmax(logits.data(), token_count);
            const int d_k = argmax(logits.data() + token_count, num_dur);
            skip = durations[d_k];

            // Commit state + last_token ONLY when k != blank.
            if (k != blank_id) {
                hyp.push_back((int32_t)k);
                last_token = (int32_t)k;
                committed = out_state;   // carry the step's new (h', c')
                emitted_any = true;
            }
            // else: discard out_state; committed/last_token unchanged.

            symbols_added += 1;
            t += skip;
            need_loop = (skip == 0);
        }

        // Infinite-loop guard: if we exited with duration 0 (blank + dur 0), step
        // forward by one frame anyway.
        if (skip == 0) skip = 1;

        // If we stopped because max_symbols was hit (not because of a positive
        // duration), advance the frame by one to make progress.
        if (symbols_added == max_symbols) t += 1;
    }

    return hyp;
}

} // namespace pk
