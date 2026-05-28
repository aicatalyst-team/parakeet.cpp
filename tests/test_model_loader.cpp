#include "model_loader.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

int main() {
    const char* env = std::getenv("PARAKEET_TEST_GGUF");
    if (!env) { std::fprintf(stderr, "PARAKEET_TEST_GGUF not set; skipping\n"); return 77; }
    pk::ModelLoader ml;
    if (!ml.load(env)) { std::fprintf(stderr, "load failed\n"); return 1; }
    const pk::ParakeetConfig& c = ml.config();
    if (c.arch.empty())   { std::fprintf(stderr, "empty arch\n"); return 1; }
    if (c.d_model == 0 || c.n_layers == 0 || c.n_heads == 0) { std::fprintf(stderr, "bad encoder dims\n"); return 1; }
    if (c.vocab_size == 0) { std::fprintf(stderr, "bad vocab\n"); return 1; }
    if (c.blank_id != c.vocab_size) { std::fprintf(stderr, "blank!=vocab\n"); return 1; }
    // mel filterbank tensor must be present
    if (ml.tensor("preprocessor.featurizer.fb") == nullptr) { std::fprintf(stderr, "no fb\n"); return 1; }
    // first conformer layer norm must be present (verbatim name)
    if (ml.tensor("encoder.layers.0.norm_feed_forward1.weight") == nullptr) {
        std::fprintf(stderr, "no layer0 norm\n"); return 1;
    }
    std::printf("loader ok: arch=%s d_model=%u layers=%u heads=%u vocab=%u\n",
                c.arch.c_str(), c.d_model, c.n_layers, c.n_heads, c.vocab_size);
    return 0;
}
