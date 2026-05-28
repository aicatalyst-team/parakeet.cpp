#include "mel.hpp"
#include "model_loader.hpp"
#include "audio_io.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Bit-identity test for the incremental (frame-local) streaming log-mel.
//
// Loads the cache-aware streaming EOU model (normalize="NA", preemph=0.97),
// computes the reference full-buffer mel via MelFrontend::compute on the whole
// clip, then feeds the SAME PCM through StreamingMel in several different chunk
// sizes (1, 137, 512, 4000 samples per feed) followed by finalize(). The
// concatenation of the emitted frames must equal the reference mel EXACTLY
// (atol 1e-6) with the same total frame count T — since the streaming mel is
// frame-local (no per-feature normalization), incremental == full-buffer.
//
// Skips (exit 77) unless PARAKEET_TEST_GGUF_EOU is set (the streaming EOU model
// is a ~480MB download, not in CI).
//
// LABEL model
// WORKING_DIRECTORY (run from project root; wav path is relative)

namespace {

// Run StreamingMel over `pcm` in fixed-size feeds, then finalize. Returns the
// concatenated feat-major mel [n_mels, T_out] and writes T_out + n_mels.
std::vector<float> run_streaming(pk::StreamingMel& sm, const std::vector<float>& pcm,
                                 int chunk, int& n_mels, int& T_out) {
    n_mels = sm.n_mels();
    std::vector<std::vector<float>> blocks;  // each block is [n_mels, k] feat-major
    std::vector<int> block_frames;
    const int n = (int)pcm.size();
    for (int off = 0; off < n; off += chunk) {
        const int len = std::min(chunk, n - off);
        int nf = 0;
        std::vector<float> b = sm.feed(pcm.data() + off, len, nf);
        blocks.push_back(std::move(b));
        block_frames.push_back(nf);
    }
    int tail = 0;
    std::vector<float> ftail = sm.finalize(tail);
    blocks.push_back(std::move(ftail));
    block_frames.push_back(tail);

    T_out = 0;
    for (int f : block_frames) T_out += f;

    // Concatenate feat-major blocks along the time axis into [n_mels, T_out].
    std::vector<float> out((size_t)n_mels * T_out);
    int t0 = 0;
    for (size_t bi = 0; bi < blocks.size(); ++bi) {
        const int k = block_frames[bi];
        if (k <= 0) continue;
        const std::vector<float>& blk = blocks[bi];  // [n_mels, k]
        for (int m = 0; m < n_mels; ++m)
            for (int t = 0; t < k; ++t)
                out[(size_t)m * T_out + (t0 + t)] = blk[(size_t)m * k + t];
        t0 += k;
    }
    return out;
}

} // namespace

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_EOU");
    if (!gguf) {
        std::fprintf(stderr,
            "test_streaming_mel: PARAKEET_TEST_GGUF_EOU not set; skip (streaming "
            "EOU model is a ~480MB download, not in CI)\n");
        return 77;
    }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) {
        std::fprintf(stderr, "test_streaming_mel: failed to load %s\n", gguf);
        return 1;
    }

    pk::Audio audio;
    if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio)) {
        std::fprintf(stderr, "test_streaming_mel: failed to load speech.wav\n");
        return 1;
    }

    // --- Reference full-buffer mel ---
    pk::MelFrontend mel_fe(ml);
    std::vector<float> ref;
    int ref_mels = 0, ref_T = 0;
    mel_fe.compute(audio.samples, ref, ref_mels, ref_T);
    std::fprintf(stderr, "test_streaming_mel: reference mel [%d, %d]\n", ref_mels, ref_T);

    // --- Several chunk sizes, each must reproduce the reference EXACTLY ---
    const int chunks[] = {1, 137, 512, 4000};
    const float atol = 1e-6f;
    float global_max_d = 0.0f;
    bool all_ok = true;

    for (int chunk : chunks) {
        pk::StreamingMel sm(ml);
        int sm_mels = 0, sm_T = 0;
        std::vector<float> got = run_streaming(sm, audio.samples, chunk, sm_mels, sm_T);

        if (sm_mels != ref_mels || sm_T != ref_T) {
            std::fprintf(stderr,
                "test_streaming_mel: SHAPE MISMATCH (chunk=%d) got [%d, %d] vs ref [%d, %d]\n",
                chunk, sm_mels, sm_T, ref_mels, ref_T);
            all_ok = false;
            continue;
        }

        float max_d = 0.0f;
        size_t argmax = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const float d = std::fabs(got[i] - ref[i]);
            if (d > max_d) { max_d = d; argmax = i; }
        }
        if (max_d > global_max_d) global_max_d = max_d;
        const bool ok = (max_d <= atol);
        std::fprintf(stderr,
            "test_streaming_mel: chunk=%-5d T=%d max|d|=%.3e (argmax idx %zu) %s\n",
            chunk, sm_T, max_d, argmax, ok ? "OK" : "FAIL");
        if (!ok) all_ok = false;
    }

    std::fprintf(stderr, "[streaming_mel] max|d| across chunk sizes = %.3e (atol %.1e)\n",
                 global_max_d, atol);

    if (!all_ok) {
        std::fprintf(stderr, "test_streaming_mel: FAIL — incremental mel != full-buffer\n");
        return 1;
    }
    std::fprintf(stderr,
        "test_streaming_mel: PASS — incremental (frame-local) mel bit-identical to "
        "MelFrontend::compute across chunk sizes {1,137,512,4000} (T=%d)\n", ref_T);
    return 0;
}
