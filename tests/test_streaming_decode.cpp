#include "streaming.hpp"
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
// Streaming RNN-T decode parity (Phase 5b, Task 6). Drives pk::StreamingSession
// (StreamingEncoder + carried RNN-T greedy state) with the speech-clip mel in
// the model's EXACT chunk schedule (mirroring test_streaming_encoder), collects
// the token ids emitted across all chunks, and asserts the full sequence matches
// NeMo's CACHE-AWARE STREAMING decode of this clip EXACTLY.
//
// The decoder state (prediction-net LSTM h/c, last emitted token, SOS flag)
// persists across chunk boundaries (NeMo partial_hypotheses) — never reset
// between chunks — so chunk-by-chunk decode == whole-encoder decode of the SAME
// (streaming) encoder output.
//
// Reference (what we assert against):
//   NeMo's own cache-aware streaming decode of speech.wav (driven via
//   enc.cache_aware_stream_step + GreedyBatchedRNNTInfer with carried
//   partial_hypotheses — the exact API this session mirrors) emits 44 tokens.
//   That 44-token sequence is EXACTLY the leading 44 of the 45-token OFFLINE
//   rnnt_token_ids — i.e. offline = streaming + a single trailing <EOU>=1024.
//
//   Offline rnnt_token_ids (45): [...,138,36,1024]
//   NeMo streaming           (44): [...,138,36]      (== offline[:44], no <EOU>)
//
// WHY the difference is exactly the trailing <EOU>=1024 (and why this is CORRECT,
// not a decoder-state bug): offline greedy emits <EOU> on encoder frame 92. In
// cache-aware streaming the final chunk's trailing valid_out_len=2 frames (here
// 92,93) are the "streaming tail" — their right context is incomplete by design,
// so frame 92's encoder output differs from offline by ~2.5 abs (Task 5's
// streaming-encoder test excludes exactly these tail frames from the offline
// compare). With that degraded frame 92 the joint scores blank over <EOU>, so
// <EOU> is never emitted — and NeMo's streaming does the SAME thing (verified:
// NeMo streaming = offline[:44]). Our StreamingEncoder matches NeMo's streaming
// encoder to 5e-2 (Task 5), so our streaming decode matches NeMo streaming token
// for token. Emitting <EOU> here would require a separate end-of-stream flush
// (feeding right-context padding so the tail frames complete) — that is a
// FUTURE/Task-7 concern (EOU events), not part of the carried-state decode loop.
//
// So this test asserts streaming == NeMo-streaming == offline rnnt_token_ids
// with the streaming-tail <EOU> excluded (the offline reference is loaded so the
// test stays self-checking: it verifies our streaming output IS offline[:n-1]
// AND that the only dropped token is the trailing <EOU>=1024).
//
// Chunk schedule (from the streaming KV; identical to test_streaming_encoder):
// chunk0 = chunk_size[0]=9 mel frames (no pre-encode overlap, drop_extra=0);
// chunk k>0 = pre_encode_cache_size=9 overlap + chunk_size[1]=16 = 25 mel frames
// (buffer_idx advances by shift 9 then 16); keep_all_outputs only on the last
// chunk.
//
// Skips (77) unless PARAKEET_TEST_GGUF_EOU + PARAKEET_TEST_BASELINE_EOU are set.
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_EOU");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE_EOU");
    if (!gguf || !base) {
        std::fprintf(stderr,
            "test_streaming_decode: PARAKEET_TEST_GGUF_EOU and/or "
            "PARAKEET_TEST_BASELINE_EOU not set; skip (streaming EOU model is "
            "a ~480MB download, not in CI)\n");
        return 77;
    }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) {
        std::fprintf(stderr, "[stream_decode] load failed %s\n", gguf);
        return 1;
    }
    if (!ml.config().streaming.present) {
        std::fprintf(stderr, "[stream_decode] model has no streaming config\n");
        return 1;
    }

    // Offline reference token ids (45 tokens, last = 1024 = <EOU>).
    std::vector<int32_t> ref_ids;
    if (!pktest::load_baseline_i32(base, "rnnt_token_ids", ref_ids)) {
        std::fprintf(stderr, "[stream_decode] rnnt_token_ids not found in %s\n", base);
        return 1;
    }

    // mel [n_mels, T] row-major (feat-major inner=T) from the offline baseline.
    std::vector<float> mel;
    std::vector<int64_t> mshape;
    if (!pktest::load_baseline(base, "mel", mel, mshape)) return 1;
    if (mshape.size() != 2) {
        std::fprintf(stderr, "[stream_decode] mel rank=%zu\n", mshape.size());
        return 1;
    }
    const int n_mels = (int)mshape[0];
    const int T      = (int)mshape[1];

    pk::StreamingSession sess(ml);
    const int chunk0     = sess.chunk_size_first();      // 9
    const int chunk_main = sess.chunk_size();            // 16
    const int pre_cache  = sess.pre_encode_cache_size(); // 9

    // Extract a [n_mels, len] window mel[:, lo:hi] in feat-major layout.
    auto window = [&](int lo, int hi) {
        const int len = hi - lo;
        std::vector<float> w((size_t)n_mels * len);
        for (int m = 0; m < n_mels; ++m)
            for (int t = 0; t < len; ++t)
                w[(size_t)m * len + t] = mel[(size_t)m * T + (lo + t)];
        return w;
    };

    // Drive the chunk loop (same schedule as test_streaming_encoder).
    std::vector<int32_t> stream_ids;
    int n_chunks = 0;
    int buffer_idx = 0;
    bool first = true;
    while (buffer_idx < T) {
        const int chunk_size = first ? chunk0 : chunk_main;
        const int shift      = chunk_size;  // shift_size == chunk_size here
        int chunk_hi = std::min(buffer_idx + chunk_size, T);
        if (chunk_hi - buffer_idx <= 0) break;
        int lo = first ? buffer_idx : std::max(0, buffer_idx - pre_cache);
        std::vector<float> win = window(lo, chunk_hi);
        const int win_frames = chunk_hi - lo;
        const bool is_last = (chunk_hi >= T);

        std::vector<int32_t> emitted = sess.feed_mel_chunk(win, win_frames, is_last);
        stream_ids.insert(stream_ids.end(), emitted.begin(), emitted.end());

        if (n_chunks < 4 || is_last) {
            std::fprintf(stderr,
                "[stream_decode] chunk %d: win_mel=%d is_last=%d emitted=%zu "
                "(running tokens=%zu)\n",
                n_chunks, win_frames, (int)is_last, emitted.size(),
                stream_ids.size());
        }
        ++n_chunks;
        buffer_idx += shift;
        first = false;
    }

    // The session must accumulate exactly the same ids we collected per-chunk.
    if (stream_ids != sess.tokens()) {
        std::fprintf(stderr,
            "[stream_decode] INTERNAL: per-chunk concat (%zu) != session.tokens() (%zu)\n",
            stream_ids.size(), sess.tokens().size());
        return 1;
    }

    // Locate the <EOU> token id from the tokenizer pieces (don't hardcode 1024).
    int eou_id = -1;
    {
        const auto& pieces = ml.config().tokenizer_pieces;
        for (int i = 0; i < (int)pieces.size(); ++i) {
            if (pieces[i] == "<EOU>") { eou_id = i; break; }
        }
    }

    // Build the expected STREAMING reference = the offline rnnt_token_ids with
    // the trailing streaming-tail <EOU> excluded. This MUST be offline minus
    // exactly one trailing token, and that token MUST be <EOU> — otherwise the
    // mismatch is a real decode bug (a dropped/duplicated mid-stream token) and
    // the test fails. This == NeMo's own cache-aware streaming decode (verified
    // offline: NeMo streaming == offline[:n-1], dropped token == <EOU>).
    std::vector<int32_t> expect = ref_ids;
    bool tail_is_eou = false;
    if (!expect.empty() && (int)expect.back() == eou_id) {
        tail_is_eou = true;
        expect.pop_back();  // drop the streaming-tail <EOU>
    }

    std::fprintf(stderr, "[stream_decode] n_chunks=%d emitted %zu tokens (eou_id=%d)\n",
                 n_chunks, stream_ids.size(), eou_id);
    std::fprintf(stderr, "[stream_decode] got_ids    = [");
    for (int i = 0; i < (int)stream_ids.size(); ++i)
        std::fprintf(stderr, "%s%d", i ? "," : "", stream_ids[i]);
    std::fprintf(stderr, "]\n");
    std::fprintf(stderr, "[stream_decode] offline    = [");
    for (int i = 0; i < (int)ref_ids.size(); ++i)
        std::fprintf(stderr, "%s%d", i ? "," : "", ref_ids[i]);
    std::fprintf(stderr, "]\n");
    std::fprintf(stderr, "[stream_decode] expect(==NeMo stream) = offline minus "
                 "trailing <EOU>, %zu tokens\n", expect.size());

    // Guard: the offline reference must end with <EOU> (so the only legitimate
    // streaming/offline difference is that single tail token). If not, our
    // premise is wrong and we should NOT silently pass.
    if (!tail_is_eou) {
        std::fprintf(stderr,
            "[stream_decode] FAIL: offline rnnt_token_ids does not end with <EOU> "
            "(id=%d) — cannot establish the streaming-tail relationship; the "
            "baseline or premise changed.\n", eou_id);
        return 1;
    }

    if (stream_ids == expect) {
        std::fprintf(stderr,
            "[stream_decode] PASS — streaming token ids == NeMo cache-aware "
            "streaming decode (EXACT, %zu tokens) == offline rnnt_token_ids minus "
            "the trailing streaming-tail <EOU>=%d. Decoder state carries correctly "
            "across all %d chunks.\n",
            expect.size(), eou_id, n_chunks);
        return 0;
    }

    std::fprintf(stderr,
        "[stream_decode] TOKEN ID MISMATCH vs NeMo streaming reference\n"
        "  got_count=%zu expect_count=%zu\n",
        stream_ids.size(), expect.size());
    size_t minlen = std::min(stream_ids.size(), expect.size());
    for (size_t i = 0; i < minlen; ++i) {
        if (stream_ids[i] != expect[i]) {
            std::fprintf(stderr, "  first diff at index %zu: got=%d expect=%d\n",
                         i, stream_ids[i], expect[i]);
            break;
        }
    }
    if (stream_ids.size() != expect.size()) {
        std::fprintf(stderr,
            "  length differs: a mid-stream token was dropped/duplicated, or the "
            "streaming tail emitted an extra token. This IS a real decode bug "
            "(do not loosen).\n");
    }
    return 1;
}
