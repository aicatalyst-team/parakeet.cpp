#include "streaming_encoder.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <algorithm>

// MODEL: nvidia/parakeet_realtime_eou_120m-v1 (cache-aware streaming FastConformer)
// WORKING_DIRECTORY: the repo root (build/tests run from there).
//
// Cache-aware streaming encoder parity (Phase 5b, Task 5). Feeds the speech-clip
// mel to pk::StreamingEncoder in the model's EXACT chunk schedule (mirroring
// NeMo's CacheAwareStreamingAudioBuffer + cache_aware_stream_step loop), then:
//   1. asserts the concatenated streaming output == NeMo's stream_encoder_out
//      from /tmp/baseline_eou_stream.gguf (the PRIMARY parity gate), and
//   2. asserts it == the OFFLINE encoder_out from /tmp/baseline_eou.gguf over the
//      LEADING frames (cache-aware equivalence). The final chunk's trailing
//      valid_out_len frames (the streaming tail) have incomplete right context
//      and diverge from offline by design (~2.5 abs), so they are excluded from
//      the offline comparison — they ARE checked against NeMo's stream baseline.
//
// Chunk schedule (from the streaming KV): chunk0 = chunk_size[0]=9 mel frames (no
// pre-encode overlap, drop_extra=0); chunk k>0 = pre_encode_cache_size=9 overlap
// + chunk_size[1]=16 = 25 mel frames (buffer_idx advances by shift 9 then 16);
// keep_all_outputs only on the last chunk.
//
// Skips (77) unless PARAKEET_TEST_GGUF_EOU + PARAKEET_TEST_BASELINE_EOU +
// PARAKEET_TEST_BASELINE_EOU_STREAM are all set.
int main() {
    const char* gguf   = std::getenv("PARAKEET_TEST_GGUF_EOU");
    const char* base   = std::getenv("PARAKEET_TEST_BASELINE_EOU");
    const char* bstream = std::getenv("PARAKEET_TEST_BASELINE_EOU_STREAM");
    if (!gguf || !base || !bstream) {
        std::fprintf(stderr, "eou stream env not set; skip\n");
        return 77;
    }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "[stream] load failed %s\n", gguf); return 1; }
    if (!ml.config().streaming.present) {
        std::fprintf(stderr, "[stream] model has no streaming config\n"); return 1;
    }

    // mel [n_mels, T] row-major (feat-major inner=T) from the offline baseline.
    std::vector<float> mel; std::vector<int64_t> mshape;
    if (!pktest::load_baseline(base, "mel", mel, mshape)) return 1;
    if (mshape.size() != 2) { std::fprintf(stderr, "[stream] mel rank=%zu\n", mshape.size()); return 1; }
    const int n_mels = (int)mshape[0];
    const int T      = (int)mshape[1];

    pk::StreamingEncoder enc(ml);
    const int D          = (int)ml.config().d_model;
    const int chunk0     = enc.chunk_size_first();   // 9
    const int chunk_main = enc.chunk_size();         // 16
    const int pre_cache  = enc.pre_encode_cache_size(); // 9

    // Extract a [n_mels, len] window mel[:, lo:hi] in feat-major layout.
    auto window = [&](int lo, int hi) {
        const int len = hi - lo;
        std::vector<float> w((size_t)n_mels * len);
        for (int m = 0; m < n_mels; ++m)
            for (int t = 0; t < len; ++t)
                w[(size_t)m * len + t] = mel[(size_t)m * T + (lo + t)];
        return w;
    };

    // Drive the chunk loop (CacheAwareStreamingAudioBuffer schedule).
    std::vector<float> stream_out;  // concatenated [valid_total, D] row-major
    int n_chunks = 0;
    int buffer_idx = 0;
    bool first = true;
    while (buffer_idx < T) {
        const int chunk_size = first ? chunk0 : chunk_main;
        const int shift      = chunk_size;  // shift_size == chunk_size here
        int chunk_hi = std::min(buffer_idx + chunk_size, T);
        if (chunk_hi - buffer_idx <= 0) break;
        // pre-encode cache overlap: chunk 0 none; chunk k>0 the previous
        // pre_cache mel frames (zero-padded if before start, which never happens
        // after chunk 0 since buffer_idx >= chunk0 >= pre_cache here).
        int lo = first ? buffer_idx : std::max(0, buffer_idx - pre_cache);
        std::vector<float> win = window(lo, chunk_hi);
        const int win_frames = chunk_hi - lo;
        const bool is_last = (chunk_hi >= T);

        int valid = 0;
        std::vector<float> e = enc.step(win, win_frames, is_last, valid);
        if (valid > 0)
            stream_out.insert(stream_out.end(), e.begin(), e.end());
        if (n_chunks < 4 || is_last) {
            std::fprintf(stderr,
                "[stream] chunk %d: win_mel=%d is_last=%d valid=%d (running T=%d)\n",
                n_chunks, win_frames, (int)is_last, valid,
                (int)(stream_out.size() / D));
        }
        ++n_chunks;
        buffer_idx += shift;
        first = false;
    }
    const int Tout = (int)(stream_out.size() / D);
    std::fprintf(stderr, "[stream] n_chunks=%d Tout=%d\n", n_chunks, Tout);

    // ---- Compare 1 (PRIMARY): vs NeMo stream_encoder_out [d_model, T'] ----
    // The baseline is channels-first [D, T']; our stream_out is [T', D]. Transpose
    // the baseline to [T', D] for an element-wise compare.
    std::vector<float> sref; std::vector<int64_t> sshape;
    if (!pktest::load_baseline(bstream, "stream_encoder_out", sref, sshape)) return 1;
    if (sshape.size() != 2 || (int)sshape[0] != D) {
        std::fprintf(stderr, "[stream] stream_encoder_out shape=[%lld,%lld] D=%d\n",
                     (long long)sshape[0], (long long)sshape[1], D);
        return 1;
    }
    const int Tref = (int)sshape[1];
    if (Tref != Tout) {
        std::fprintf(stderr, "[stream] T mismatch: ours=%d nemo=%d\n", Tout, Tref);
        return 1;
    }
    std::vector<float> sref_td((size_t)Tout * D);  // [T', D] row-major
    for (int c = 0; c < D; ++c)
        for (int t = 0; t < Tout; ++t)
            sref_td[(size_t)t * D + c] = sref[(size_t)c * Tout + t];
    bool ok_stream = pktest::compare(stream_out, sref_td, "streaming_encoder.vsNeMo",
                                     /*atol*/5e-2f, /*rtol*/5e-2f);

    // ---- Compare 2: vs OFFLINE encoder_out over the LEADING frames ----
    // offline_match_T = T' - valid_out_len (exclude the final-chunk tail).
    std::vector<float> oref; std::vector<int64_t> oshape;
    if (!pktest::load_baseline(base, "encoder_out", oref, oshape)) return 1;
    bool ok_offline = true;
    if (oshape.size() == 2 && (int)oshape[0] == D) {
        const int Toff = (int)oshape[1];
        int match_T = std::min(Tout, Toff) - enc.valid_out_len();
        if (match_T < 0) match_T = 0;
        std::vector<float> got_lead((size_t)match_T * D);
        std::vector<float> ref_lead((size_t)match_T * D);
        for (int t = 0; t < match_T; ++t)
            for (int c = 0; c < D; ++c) {
                got_lead[(size_t)t * D + c] = stream_out[(size_t)t * D + c];
                ref_lead[(size_t)t * D + c] = oref[(size_t)c * Toff + t];
            }
        ok_offline = pktest::compare(got_lead, ref_lead, "streaming_encoder.vsOffline",
                                     /*atol*/5e-2f, /*rtol*/5e-2f);
    }

    return (ok_stream && ok_offline) ? 0 : 1;
}
