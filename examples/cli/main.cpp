#include "parakeet.h"
#include "model_loader.hpp"
#include <cstdio>
#include <cstring>
#include <string>

static int cmd_info(const char* path) {
    pk::ModelLoader ml;
    if (!ml.load(path)) { std::fprintf(stderr, "failed to load %s\n", path); return 1; }
    const pk::ParakeetConfig& c = ml.config();
    std::printf("parakeet.cpp %s\n", parakeet_version());
    std::printf("model: %s\n", path);
    std::printf("  arch            : %s\n", c.arch.c_str());
    std::printf("  d_model/layers/heads: %u / %u / %u\n", c.d_model, c.n_layers, c.n_heads);
    std::printf("  conv_kernel/norm: %u / %s\n", c.conv_kernel, c.conv_norm_type.c_str());
    std::printf("  xscaling        : %s\n", c.xscaling ? "true" : "false");
    std::printf("  subsampling     : x%u (ch=%u)\n", c.subsampling_factor, c.subsampling_conv_channels);
    std::printf("  mel/n_fft/win/hop: %u / %u / %u / %u\n", c.n_mels, c.n_fft, c.win_length, c.hop_length);
    std::printf("  vocab/blank     : %u / %u\n", c.vocab_size, c.blank_id);
    if (!c.tdt_durations.empty()) {
        std::printf("  tdt durations   : [");
        for (size_t i=0;i<c.tdt_durations.size();++i) std::printf("%s%d", i?",":"", c.tdt_durations[i]);
        std::printf("]\n");
    }
    return 0;
}

// parakeet-cli transcribe --model <m.gguf> --input <wav> -> prints transcript.
static int cmd_transcribe(int argc, char** argv) {
    std::string model, input;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model = argv[++i];
        } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input = argv[++i];
        }
    }
    if (model.empty() || input.empty()) {
        std::fprintf(stderr, "usage: parakeet-cli transcribe --model <m.gguf> --input <wav>\n");
        return 2;
    }
    char* out = nullptr;
    int rc = parakeet_transcribe_file(model.c_str(), input.c_str(), &out);
    if (rc != 0) {
        std::fprintf(stderr, "transcribe failed (code %d): model=%s input=%s\n",
                     rc, model.c_str(), input.c_str());
        return 1;
    }
    std::printf("%s\n", out);
    parakeet_free_string(out);
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "info") == 0) return cmd_info(argv[2]);
    if (argc >= 2 && std::strcmp(argv[1], "transcribe") == 0)
        return cmd_transcribe(argc - 2, argv + 2);
    std::fprintf(stderr,
        "usage:\n"
        "  parakeet-cli info <model.gguf>\n"
        "  parakeet-cli transcribe --model <model.gguf> --input <wav>\n");
    return 2;
}
