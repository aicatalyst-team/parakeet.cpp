// Validates the ggml DFT-matmul GpuMel against the FFT-based MelFrontend. On
// this CPU box both run on the CPU backend — the point is to check the GpuMel
// graph math (DFT-as-matmul, power, filterbank, log, transpose, per-feature
// norm) reproduces MelFrontend's output to within a loose f32 tolerance.
#include "mel.hpp"
#include "mel_gpu.hpp"
#include "model_loader.hpp"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (!gguf) { std::fprintf(stderr, "PARAKEET_TEST_GGUF not set; skip\n"); return 77; }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load failed\n"); return 1; }

    // Deterministic 4 s test signal: a fixed mix of sines + seeded LCG noise.
    const int S = 16000 * 4;
    std::vector<float> samples((size_t)S);
    uint32_t seed = 0x12345678u;
    for (int i = 0; i < S; ++i) {
        const double t = (double)i / 16000.0;
        double v = 0.6 * std::sin(2.0 * M_PI * 220.0 * t)
                 + 0.3 * std::sin(2.0 * M_PI * 740.0 * t)
                 + 0.1 * std::sin(2.0 * M_PI * 3100.0 * t);
        seed = seed * 1664525u + 1013904223u;
        const double noise = ((double)(seed >> 9) / (double)(1u << 23)) - 0.5; // [-0.5,0.5)
        v += 0.05 * noise;
        samples[i] = (float)v;
    }

    pk::MelFrontend mel(ml);
    std::vector<float> feats_cpu; int n_mels_c = 0, T_c = 0;
    mel.compute(samples, feats_cpu, n_mels_c, T_c);

    pk::GpuMel gmel(ml);
    std::vector<float> feats_gpu; int n_mels_g = 0, T_g = 0;
    gmel.compute(samples, feats_gpu, n_mels_g, T_g);

    if (n_mels_c != n_mels_g || T_c != T_g) {
        std::fprintf(stderr, "shape mismatch: cpu (%d,%d) vs gpu (%d,%d)\n",
                     n_mels_c, T_c, n_mels_g, T_g);
        return 1;
    }
    if (feats_cpu.size() != feats_gpu.size()) {
        std::fprintf(stderr, "size mismatch: %zu vs %zu\n",
                     feats_cpu.size(), feats_gpu.size());
        return 1;
    }

    double max_abs = 0.0, sum_abs = 0.0;
    for (size_t i = 0; i < feats_cpu.size(); ++i) {
        const double d = std::fabs((double)feats_cpu[i] - (double)feats_gpu[i]);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
    }
    const double mean_abs = feats_cpu.empty() ? 0.0 : sum_abs / (double)feats_cpu.size();

    std::fprintf(stderr, "GpuMel vs MelFrontend: shape (%d,%d) max_abs_diff=%.6g mean_abs_diff=%.6g\n",
                 n_mels_c, T_c, max_abs, mean_abs);

    if (max_abs >= 0.05) {
        std::fprintf(stderr, "FAIL: max_abs_diff %.6g >= 0.05\n", max_abs);
        return 1;
    }
    std::fprintf(stderr, "PASS\n");
    return 0;
}
