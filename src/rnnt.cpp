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

RnntDecodeState rnnt_decode_init(const PredictionNet& pred) {
    RnntDecodeState st;
    st.state      = pred.zero_state();
    st.last_token = -1;     // -1 sentinel: nothing emitted yet -> SOS.
    st.have_token = false;
    st.hyp.clear();
    return st;
}

std::vector<int32_t> rnnt_decode_frames(const PredictionNet& pred, const Joint& joint,
                                        const std::vector<float>& enc_frames,
                                        int Tnew, int enc_hidden,
                                        RnntDecodeState& st,
                                        int blank_id, int max_symbols) {
    assert((int)enc_frames.size() == (size_t)Tnew * enc_hidden);
    assert(joint.num_durations() == 0);

    const int V_plus      = joint.V_plus();   // vocab + 1 (incl. blank), no durations
    const int token_count = V_plus;           // argmax over the full output vector
    assert(token_count == joint.vocab_size() + 1);

    // Tokens emitted in THIS call only (st.hyp accumulates across all calls).
    std::vector<int32_t> emitted_this_call;

    // Scratch reused across inner steps.
    std::vector<float> g;
    PredState out_state;
    std::vector<float> logits;
    int v_out = 0;
    std::vector<float> enc_frame((size_t)enc_hidden);

    int t = 0;
    while (t < Tnew) {
        int emitted = 0;
        while (emitted < max_symbols) {
            // First step (no token committed yet, EVER across the whole stream)
            // uses SOS; otherwise feed the last EMITTED token. The committed
            // state / last_token / have_token are carried in `st` across chunks.
            const bool is_sos = !st.have_token;
            const int32_t last_label = st.have_token ? st.last_token : blank_id;

            // Prediction net single step from the committed state.
            pred.step(last_label, is_sos, st.state, g, out_state);

            // Joint: enc[t] (T=1) x g (U=1) -> raw logits [V_plus=vocab+1].
            std::memcpy(enc_frame.data(), &enc_frames[(size_t)t * enc_hidden],
                        (size_t)enc_hidden * sizeof(float));
            joint.forward(enc_frame, /*T=*/1, enc_hidden,
                          g, /*U=*/1, (int)g.size(), logits, v_out);
            assert(v_out == V_plus);

            const int k = argmax(logits.data(), token_count);

            // Blank -> stop emitting at this frame and advance time.
            if (k == blank_id) break;

            // Non-blank -> emit, commit state + last token, STAY at this frame.
            st.hyp.push_back((int32_t)k);
            emitted_this_call.push_back((int32_t)k);
            st.last_token = (int32_t)k;
            st.state = out_state;
            st.have_token = true;
            emitted += 1;
        }

        // Advance exactly one frame (blank, or max_symbols exhausted).
        t += 1;
    }

    return emitted_this_call;
}

std::vector<int32_t> rnnt_greedy(const PredictionNet& pred, const Joint& joint,
                                 const std::vector<float>& enc, int T, int enc_hidden,
                                 int blank_id, int max_symbols) {
    // The whole-encoder greedy decode is exactly the stateful stepper driven
    // once over all T frames from a fresh state (the loop carries nothing but
    // RnntDecodeState across frames, so chunking is irrelevant to the result).
    RnntDecodeState st = rnnt_decode_init(pred);
    rnnt_decode_frames(pred, joint, enc, T, enc_hidden, st, blank_id, max_symbols);
    return st.hyp;
}

} // namespace pk
