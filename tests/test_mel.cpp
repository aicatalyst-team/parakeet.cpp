#include "mel.hpp"
#include "model_loader.hpp"
#include "audio_io.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
int main(){
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE");
    if(!gguf||!base){ std::fprintf(stderr,"env not set; skip\n"); return 77; }
    pk::ModelLoader ml; if(!ml.load(gguf)) return 1;
    pk::Audio a; if(!pk::load_audio_16k_mono("tests/fixtures/clip.wav", a)) return 1;
    pk::MelFrontend mel(ml);            // pulls fb/window/config from loader
    std::vector<float> feats; int n_mels=0, T=0;
    mel.compute(a.samples, feats, n_mels, T);   // feats row-major [n_mels, T]
    std::vector<float> ref; std::vector<int64_t> shape;
    if(!pktest::load_baseline(base, "mel", ref, shape)) return 1;
    // baseline "mel" is [n_mels, T] (squeezed). Sizes must match.
    bool ok = pktest::compare(feats, ref, "mel", /*atol*/1e-2f, /*rtol*/1e-2f);
    return ok?0:1;
}
