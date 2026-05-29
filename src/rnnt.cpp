#include "rnnt.hpp"
#include <cassert>
#include <cmath>

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

// NeMo's rescaled `max_prob` confidence (method 'max_prob', alpha==1.0):
//   conf = (N * p_max - 1) / (N - 1),  p_max = softmax(logits)[k].
// For RNN-T the confidence slice is the FULL joint output vector (V_plus =
// vocab + 1, blank included; no durations) — NeMo log_softmaxes the whole
// joint output. N == n (the slice size). Stable softmax (subtract the max).
float max_prob_conf_logits(const float* a, int n, int k) {
    float mx = a[0];
    for (int i = 1; i < n; ++i) if (a[i] > mx) mx = a[i];
    double denom = 0.0;
    for (int i = 0; i < n; ++i) denom += std::exp((double)a[i] - (double)mx);
    const double p_max = std::exp((double)a[k] - (double)mx) / denom;
    const double N = (double)n;
    return (float)((N * p_max - 1.0) / (N - 1.0));
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
                                        int blank_id, int max_symbols,
                                        std::vector<int32_t>* emit_frames,
                                        std::vector<TokenInfo>* tokens) {
    assert((int)enc_frames.size() == (size_t)Tnew * enc_hidden);
    assert(joint.num_durations() == 0);

    const int V_plus      = joint.V_plus();   // vocab + 1 (incl. blank), no durations
    const int token_count = V_plus;           // argmax over the full output vector
    assert(token_count == joint.vocab_size() + 1);

    // Tokens emitted in THIS call only (st.hyp accumulates across all calls).
    std::vector<int32_t> emitted_this_call;

    // Precompute the encoder projection over ALL frames ONCE (one matmul on the
    // persistent backend), reused for every step. The per-step joint below is a
    // tight churn-free graph on the same backend. The old code rebuilt the full
    // joint per (t,u) (a fresh graph per step, the bulk of the per-utterance
    // graph dispatches).
    std::vector<float> enc_proj;   // row-major [Tnew, joint_hidden]
    joint.precompute_enc_proj(enc_frames, Tnew, enc_hidden, enc_proj);
    const int H = joint.joint_hidden();

    // Scratch reused across inner steps.
    std::vector<float> g;
    PredState out_state;
    std::vector<float> logits;

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

            // Joint for (t,u): precomputed enc_proj[t] x g -> raw logits [V_plus].
            joint.step_logits(enc_proj.data() + (size_t)t * H,
                              g.data(), (int)g.size(), logits);

            const int k = argmax(logits.data(), token_count);

            // Blank -> stop emitting at this frame and advance time.
            if (k == blank_id) break;

            // Non-blank -> emit, commit state + last token, STAY at this frame.
            st.hyp.push_back((int32_t)k);
            emitted_this_call.push_back((int32_t)k);
            if (emit_frames) emit_frames->push_back((int32_t)t);
            if (tokens) {
                // NeMo per-token metadata (GreedyRNNTInfer._greedy_decode +
                // max_prob confidence): frame = the (local) encoder frame t at
                // emission, conf = max_prob over the full joint output vector
                // (N = V_plus = vocab+1), span = 1 (RNN-T advances one frame).
                const float conf = max_prob_conf_logits(logits.data(), token_count, k);
                tokens->push_back(TokenInfo{ (int32_t)k, (int32_t)t, conf, 1 });
            }
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
                                 int blank_id, int max_symbols,
                                 std::vector<TokenInfo>* tokens) {
    // The whole-encoder greedy decode is exactly the stateful stepper driven
    // once over all T frames from a fresh state (the loop carries nothing but
    // RnntDecodeState across frames, so chunking is irrelevant to the result).
    RnntDecodeState st = rnnt_decode_init(pred);
    if (tokens) tokens->clear();
    rnnt_decode_frames(pred, joint, enc, T, enc_hidden, st, blank_id, max_symbols,
                       /*emit_frames=*/nullptr, tokens);
    return st.hyp;
}

} // namespace pk
