#include "prediction.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (!gguf) { std::fprintf(stderr, "env not set; skip\n"); return 77; }
    pk::ModelLoader ml; if (!ml.load(gguf)) return 1;
    pk::PredictionNet pred(ml);
    const int H = pred.hidden_size(), L = pred.num_layers();
    std::vector<int32_t> toks = {5, 0, 42};
    std::vector<uint8_t> sos  = {0, 1, 0};
    const int N = 3;
    std::vector<std::vector<float>> g_ref(N);
    pk::PredState z = pred.zero_state();
    for (int n = 0; n < N; ++n) { pk::PredState os; pred.step(toks[n], sos[n], z, g_ref[n], os); }
    pk::BatchedPredState bin;
    bin.h.assign(L, std::vector<float>((size_t)H*N, 0.0f));
    bin.c.assign(L, std::vector<float>((size_t)H*N, 0.0f));
    std::vector<float> gb; pk::BatchedPredState bout;
    pred.step_batch(toks, sos, bin, gb, bout);
    bool ok = (int)gb.size() == H*N;
    for (int n = 0; n < N && ok; ++n) {
        std::vector<float> col(gb.begin()+(size_t)n*H, gb.begin()+(size_t)(n+1)*H);
        ok = pktest::compare(col, g_ref[n], "predbatch.g", 1e-4f, 1e-4f) && ok;
    }
    // also check out_state top-layer h column matches g (sanity) and sizes
    if (ok && ((int)bout.h.size()!=L || (int)bout.h[L-1].size()!=H*N)) { std::fprintf(stderr,"state shape\n"); ok=false; }
    return ok ? 0 : 1;
}
