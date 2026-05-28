#include "rnnt.hpp"
#include <cassert>
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

std::vector<int32_t> rnnt_greedy(const PredictionNet& pred, const Joint& joint,
                                 const std::vector<float>& enc, int T, int enc_hidden,
                                 int blank_id, int max_symbols) {
    assert((int)enc.size() == (size_t)T * enc_hidden);
    assert(joint.num_durations() == 0);

    const int V_plus      = joint.V_plus();   // vocab + 1 (incl. blank), no durations
    const int token_count = V_plus;           // argmax over the full output vector
    assert(token_count == joint.vocab_size() + 1);

    std::vector<int32_t> hyp;

    // Committed (non-blank) decoding state and last emitted token.
    PredState committed = pred.zero_state();
    int32_t last_token = -1;     // -1 sentinel: nothing emitted yet -> SOS.
    bool have_token = false;

    // Scratch reused across inner steps.
    std::vector<float> g;
    PredState out_state;
    std::vector<float> logits;
    int v_out = 0;
    std::vector<float> enc_frame((size_t)enc_hidden);

    int t = 0;
    while (t < T) {
        int emitted = 0;
        while (emitted < max_symbols) {
            // First step (no token committed) uses SOS; otherwise feed the last
            // EMITTED token.
            const bool is_sos = !have_token;
            const int32_t last_label = have_token ? last_token : blank_id;

            // Prediction net single step from the committed state.
            pred.step(last_label, is_sos, committed, g, out_state);

            // Joint: enc[t] (T=1) x g (U=1) -> raw logits [V_plus=vocab+1].
            std::memcpy(enc_frame.data(), &enc[(size_t)t * enc_hidden],
                        (size_t)enc_hidden * sizeof(float));
            joint.forward(enc_frame, /*T=*/1, enc_hidden,
                          g, /*U=*/1, (int)g.size(), logits, v_out);
            assert(v_out == V_plus);

            const int k = argmax(logits.data(), token_count);

            // Blank -> stop emitting at this frame and advance time.
            if (k == blank_id) break;

            // Non-blank -> emit, commit state + last token, STAY at this frame.
            hyp.push_back((int32_t)k);
            last_token = (int32_t)k;
            committed = out_state;
            have_token = true;
            emitted += 1;
        }

        // Advance exactly one frame (blank, or max_symbols exhausted).
        t += 1;
    }

    return hyp;
}

} // namespace pk
