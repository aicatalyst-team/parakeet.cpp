#include "search.hpp"

namespace pk {

std::vector<int32_t> ctc_greedy(const std::vector<float>& logits,
                                int T, int vocab_plus_1, int blank_id) {
    std::vector<int32_t> out;
    if (T <= 0 || vocab_plus_1 <= 0) return out;

    out.reserve(T);
    // Mirror NeMo's AbstractCTCDecoding.decode_hypothesis fold_consecutive loop:
    //   previous = blank_id
    //   emit p iff (p != previous or previous == blank_id) and p != blank_id
    //   previous = p   (updated every frame, including blank frames)
    int32_t previous = blank_id;
    for (int t = 0; t < T; ++t) {
        const float* row = logits.data() + (size_t)t * vocab_plus_1;
        // argmax over the vocab axis (ties resolve to the lowest index, as in
        // PyTorch/NumPy argmax — matches NeMo's torch.argmax / prediction.max).
        int32_t p = 0;
        float best_val = row[0];
        for (int v = 1; v < vocab_plus_1; ++v) {
            if (row[v] > best_val) { best_val = row[v]; p = v; }
        }
        if ((p != previous || previous == blank_id) && p != blank_id) {
            out.push_back(p);
        }
        previous = p;
    }
    return out;
}

} // namespace pk
