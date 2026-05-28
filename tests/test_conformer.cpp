#include "conformer.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>

// Parity test for a full FastConformer layer vs NeMo's ConformerLayer
// (encoder.layers[0]). Input is the baseline `enc_pre_layers` (the tensor fed
// into layers[0]) plus the relative positional encoding `pos_emb`; output is
// compared against `enc_layer_0` (the layer's final output after norm_out).
//
// For localization we also compare the internal ConformerConvolution output
// against the baseline `l0_conv_out`.
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE");
    if (!gguf || !base) { std::fprintf(stderr, "env not set; skip\n"); return 77; }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) return 1;

    // Layer input: baseline "enc_pre_layers" is [T, d_model] row-major.
    std::vector<float> xin; std::vector<int64_t> xshape;
    if (!pktest::load_baseline(base, "enc_pre_layers", xin, xshape)) return 1;
    if (xshape.size() != 2) { std::fprintf(stderr, "enc_pre_layers rank=%zu\n", xshape.size()); return 1; }
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
    // key/query 25 in attention and zeros time-position 25 before the conv.
    const int valid_len = T - 1;

    pk::ConformerLayer layer(ml, /*layer_idx*/0);
    std::vector<float> out, conv_out;
    layer.forward_with_conv(xin, T, pos, pos_len, valid_len, out, conv_out);

    bool ok = true;

    // Localization: internal conv-module output vs baseline l0_conv_out.
    std::vector<float> cref; std::vector<int64_t> cshape;
    if (pktest::load_baseline(base, "l0_conv_out", cref, cshape)) {
        if (cshape.size() == 2 && (int)cshape[0] == T && (int)cshape[1] == d_model) {
            // Localization only; do not fail the test on this sub-block (full
            // layer parity is the authoritative check). Print the diff.
            pktest::compare(conv_out, cref, "conformer.conv", /*atol*/3e-2f, /*rtol*/3e-2f);
        }
    }

    // Reference: enc_layer_0 is [T, d_model] row-major.
    std::vector<float> ref; std::vector<int64_t> rshape;
    if (!pktest::load_baseline(base, "enc_layer_0", ref, rshape)) return 1;
    if (rshape.size() != 2 || (int)rshape[0] != T || (int)rshape[1] != d_model) {
        std::fprintf(stderr, "enc_layer_0 shape=[%lld,%lld] expected [%d,%d]\n",
                     (long long)rshape[0], (long long)rshape[1], T, d_model);
        return 1;
    }

    ok = pktest::compare(out, ref, "conformer", /*atol*/3e-2f, /*rtol*/3e-2f) && ok;
    return ok ? 0 : 1;
}
