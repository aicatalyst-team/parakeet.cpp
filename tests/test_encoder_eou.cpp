#include "encoder.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>

// MODEL: nvidia/parakeet_realtime_eou_120m-v1 (cache-aware streaming FastConformer)
// WORKING_DIRECTORY: the repo root (build/tests run from there).
//
// Full OFFLINE encoder parity for the streaming EOU model. This exercises the
// three streaming-specific numerics together, in whole-sequence (offline) mode:
//   - causal subsampling   (causal_downsampling=True)
//   - layer_norm conv + causal depthwise conv   (Task 2)
//   - chunked-limited attention mask   (att_context_size=[70,1], chunked_limited)
//
// NeMo's normal offline forward for this model already applies all three, so the
// baseline `encoder_out` is the ground truth. We compare pk::Encoder::forward(mel)
// against it (atol/rtol 5e-2 — error accumulates over 17 conformer layers), and
// localize divergence with the layer-0 capture (`enc_layer_0`).
//
// Skips (77) unless BOTH PARAKEET_TEST_GGUF_EOU and PARAKEET_TEST_BASELINE_EOU are
// set, so the default suite stays green without the (large) streaming model.
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_EOU");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE_EOU");
    if (!gguf || !base) { std::fprintf(stderr, "eou env not set; skip\n"); return 77; }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "[encoder_eou] failed to load %s\n", gguf); return 1; }

    // Input: baseline "mel" is [n_mels, T] row-major (feat-major inner=T).
    std::vector<float> mel; std::vector<int64_t> mshape;
    if (!pktest::load_baseline(base, "mel", mel, mshape)) return 1;
    if (mshape.size() != 2) { std::fprintf(stderr, "mel shape rank=%zu\n", mshape.size()); return 1; }
    const int n_mels = (int)mshape[0];
    const int T      = (int)mshape[1];

    const int n_layers = (int)ml.config().n_layers;
    // Localize across layer 0, the middle, and the last layer when the baseline
    // carries those captures; the EOU baseline currently dumps only enc_layer_0.
    std::vector<int> capture = {0, n_layers / 2, n_layers - 1};

    pk::Encoder enc(ml);
    std::vector<float> enc_out; int dm = 0, Tout = 0;
    std::vector<std::vector<float>> layer_outs;
    enc.forward_capture(mel, n_mels, T, enc_out, dm, Tout, capture, layer_outs);

    // Localization (print only; do not fail on intermediates). Baseline layer
    // outputs are [T', d_model] row-major (ConformerLayer output orientation).
    const char* layer_names[3] = {"enc_layer_0", "enc_layer_mid", "enc_layer_last"};
    const char* labels[3]      = {"encoder_eou.layer0", "encoder_eou.layerMid", "encoder_eou.layerLast"};
    for (int c = 0; c < 3; ++c) {
        std::vector<float> lref; std::vector<int64_t> lshape;
        if (pktest::load_baseline(base, layer_names[c], lref, lshape)) {
            pktest::compare(layer_outs[c], lref, labels[c], /*atol*/5e-2f, /*rtol*/5e-2f);
        }
    }

    // Final: baseline encoder_out is [d_model, T'] row-major (channels-first).
    std::vector<float> ref; std::vector<int64_t> rshape;
    if (!pktest::load_baseline(base, "encoder_out", ref, rshape)) return 1;
    if (rshape.size() != 2 || (int)rshape[0] != dm || (int)rshape[1] != Tout) {
        std::fprintf(stderr, "[encoder_eou] encoder_out shape=[%lld,%lld] expected [%d,%d]\n",
                     (long long)rshape[0], (long long)rshape[1], dm, Tout);
        return 1;
    }
    bool ok_enc = pktest::compare(enc_out, ref, "encoder_eou", /*atol*/5e-2f, /*rtol*/5e-2f);

    return ok_enc ? 0 : 1;
}
