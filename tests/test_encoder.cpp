#include "encoder.hpp"
#include "pos_enc.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>

// Parity test for the full FastConformer encoder vs NeMo's ConformerEncoder.
//
//   (a) Positional encoding: compute pos_emb [2T'-1, d_model] in C++ and assert
//       it matches the baseline `pos_emb` (atol 1e-3). De-risks the whole stack:
//       a wrong pos_emb breaks every layer's relative-position attention.
//   (b) Full encoder: from baseline `mel` run subsampling + pos-enc + 17 layers,
//       transpose to [d_model, T'], compare to baseline `encoder_out`
//       (atol/rtol 5e-2 — error accumulates over 17 layers).
//   (c) Localization: capture layer 0 / n//2 / last outputs and print their diffs
//       against `enc_layer_0` / `enc_layer_mid` / `enc_layer_last` so a divergence
//       can be pinned to a layer. Only the FINAL `encoder_out` diff fails the test.
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE");
    if (!gguf || !base) { std::fprintf(stderr, "env not set; skip\n"); return 77; }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) return 1;

    // ---- (a) positional encoding parity ----
    // baseline pos_emb is [2T'-1, d_model] row-major.
    std::vector<float> posref; std::vector<int64_t> pshape;
    if (!pktest::load_baseline(base, "pos_emb", posref, pshape)) return 1;
    if (pshape.size() != 2) { std::fprintf(stderr, "pos_emb rank=%zu\n", pshape.size()); return 1; }
    const int pos_len = (int)pshape[0];       // 2T'-1
    const int d_model = (int)pshape[1];
    const int Tp_ref  = (pos_len + 1) / 2;    // T'

    std::vector<float> poscomp;
    pk::rel_pos_encoding(Tp_ref, d_model, poscomp);
    bool ok_pos = pktest::compare(poscomp, posref, "pos_emb", /*atol*/1e-3f, /*rtol*/1e-3f);

    // ---- (b)+(c) full encoder ----
    // Input: baseline "mel" is [n_mels, T] row-major (feat-major inner=T).
    std::vector<float> mel; std::vector<int64_t> mshape;
    if (!pktest::load_baseline(base, "mel", mel, mshape)) return 1;
    if (mshape.size() != 2) { std::fprintf(stderr, "mel shape rank=%zu\n", mshape.size()); return 1; }
    const int n_mels = (int)mshape[0];
    const int T      = (int)mshape[1];

    const int n_layers = (int)ml.config().n_layers;
    std::vector<int> capture = {0, n_layers / 2, n_layers - 1};

    pk::Encoder enc(ml);
    std::vector<float> enc_out; int dm = 0, Tout = 0;
    std::vector<std::vector<float>> layer_outs;
    enc.forward_capture(mel, n_mels, T, enc_out, dm, Tout, capture, layer_outs);

    // Localization (print only; do not fail on intermediates).
    const char* layer_names[3] = {"enc_layer_0", "enc_layer_mid", "enc_layer_last"};
    const char* labels[3]      = {"encoder.layer0", "encoder.layerMid", "encoder.layerLast"};
    for (int c = 0; c < 3; ++c) {
        std::vector<float> lref; std::vector<int64_t> lshape;
        if (pktest::load_baseline(base, layer_names[c], lref, lshape)) {
            // baseline layer outputs are [T', d_model] row-major, matching the
            // ConformerLayer output orientation captured here.
            pktest::compare(layer_outs[c], lref, labels[c], /*atol*/5e-2f, /*rtol*/5e-2f);
        }
    }

    // Final: baseline encoder_out is [d_model, T'] row-major (channels-first).
    std::vector<float> ref; std::vector<int64_t> rshape;
    if (!pktest::load_baseline(base, "encoder_out", ref, rshape)) return 1;
    if (rshape.size() != 2 || (int)rshape[0] != dm || (int)rshape[1] != Tout) {
        std::fprintf(stderr, "encoder_out shape=[%lld,%lld] expected [%d,%d]\n",
                     (long long)rshape[0], (long long)rshape[1], dm, Tout);
        return 1;
    }
    bool ok_enc = pktest::compare(enc_out, ref, "encoder", /*atol*/5e-2f, /*rtol*/5e-2f);

    return (ok_pos && ok_enc) ? 0 : 1;
}
