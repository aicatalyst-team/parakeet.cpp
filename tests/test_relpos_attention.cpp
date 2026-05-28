#include "relpos_attention.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>

// Parity test for relative-position multi-head self-attention vs NeMo's
// RelPositionMultiHeadAttention (encoder.layers[0].self_attn).
//
// The baseline tensor `l0_attn_in` is the EXACT input NeMo feeds to self_attn,
// i.e. norm_self_att(residual) where residual already includes FFN1 — so it is
// NOT norm_self_att(enc_pre_layers). Feeding it directly keeps this test scoped
// to attention only (FFN1 is the next task). We compare against `l0_attn_out`.
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE");
    if (!gguf || !base) { std::fprintf(stderr, "env not set; skip\n"); return 77; }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) return 1;

    // Attention input: baseline "l0_attn_in" is [T, d_model] row-major.
    std::vector<float> xin; std::vector<int64_t> xshape;
    if (!pktest::load_baseline(base, "l0_attn_in", xin, xshape)) return 1;
    if (xshape.size() != 2) { std::fprintf(stderr, "l0_attn_in rank=%zu\n", xshape.size()); return 1; }
    const int T       = (int)xshape[0];
    const int d_model = (int)xshape[1];

    // Relative positional encoding: baseline "pos_emb" is [2T-1, d_model].
    std::vector<float> pos; std::vector<int64_t> pshape;
    if (!pktest::load_baseline(base, "pos_emb", pos, pshape)) return 1;
    if (pshape.size() != 2) { std::fprintf(stderr, "pos_emb rank=%zu\n", pshape.size()); return 1; }
    const int pos_len = (int)pshape[0];
    if (pos_len != 2 * T - 1 || (int)pshape[1] != d_model) {
        std::fprintf(stderr, "pos_emb shape=[%d,%lld] expected [%d,%d]\n",
                     pos_len, (long long)pshape[1], 2 * T - 1, d_model);
        return 1;
    }

    // Valid frames: 0..24 (frame 25 is a center-pad/padding frame). NeMo masks
    // key column 25 and query row 25 in layer-0 self-attention.
    const int valid_len = T - 1;

    pk::RelPosAttention attn(ml, /*layer_idx*/0);
    std::vector<float> out;
    attn.forward(xin, T, pos, pos_len, valid_len, out);

    // Reference: l0_attn_out is [T, d_model] row-major.
    std::vector<float> ref; std::vector<int64_t> rshape;
    if (!pktest::load_baseline(base, "l0_attn_out", ref, rshape)) return 1;
    if (rshape.size() != 2 || (int)rshape[0] != T || (int)rshape[1] != d_model) {
        std::fprintf(stderr, "l0_attn_out shape=[%lld,%lld] expected [%d,%d]\n",
                     (long long)rshape[0], (long long)rshape[1], T, d_model);
        return 1;
    }

    bool ok = pktest::compare(out, ref, "relpos_attention", /*atol*/2e-2f, /*rtol*/2e-2f);
    return ok ? 0 : 1;
}
