#pragma once
#include "prediction.hpp"
#include "joint.hpp"
#include <vector>
#include <cstdint>

namespace pk {

// Standard RNN-Transducer greedy decoding (no duration head).
//
// Ports NeMo GreedyRNNTInfer._greedy_decode (rnnt_greedy_decoding.py). Drives the
// prediction net + joint frame-by-frame. Unlike TDT there are NO duration logits:
// the joint output is exactly vocab+1 (num_durations()==0 -> V_plus=vocab+1). On a
// blank the time index advances by exactly one frame; on a non-blank a symbol is
// emitted and the loop STAYS at the same frame (capped by max_symbols).
//
// Algorithm (NeMo):
//   t = 0; hyp = []; committed_state = zeros; last_token = SOS; have_token = false
//   while t < T:
//     emitted = 0
//     while emitted < max_symbols:
//       is_sos = !have_token
//       g, out_state = pred.step(last_label, is_sos, committed_state)
//       logits = joint(enc[t], g)             # raw [vocab+1]
//       k = argmax(logits)                    # token (incl. blank)
//       if k == blank: break                  # blank -> stop emitting, advance time
//       hyp.append(k); last_token = k; committed_state = out_state; have_token = true
//       emitted += 1                          # commit on non-blank, STAY at t
//     t += 1                                  # advance exactly one frame
//
// Argmax is taken over the RAW joint logits — argmax is invariant under the
// monotonic log_softmax NeMo applies for confidence, so greedy needs no softmax.
//
// pred:       stateful prediction net (carries LSTM h,c across steps).
// joint:      RNN-T joint network (called per (t, u) with T=U=1), V_plus=vocab+1.
// enc:        encoder output, row-major [T, enc_hidden] — enc[t*enc_hidden + c].
// T:          number of encoder time frames.
// enc_hidden: encoder feature dimension (= d_model).
// blank_id:   blank token id (= vocab_size); argmax range is [0, vocab+1).
// max_symbols: max symbols emitted per time frame (NeMo default 10).
//
// Returns the emitted token-id sequence (hyp). All emitted ids are < blank_id.
std::vector<int32_t> rnnt_greedy(const PredictionNet& pred, const Joint& joint,
                                 const std::vector<float>& enc, int T, int enc_hidden,
                                 int blank_id, int max_symbols);

} // namespace pk
