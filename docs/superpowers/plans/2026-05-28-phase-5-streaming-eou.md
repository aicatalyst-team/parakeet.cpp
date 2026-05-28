# parakeet.cpp Phase 5 — Cache-aware streaming + EOU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a real-time streaming inference path for the cache-aware FastConformer model `nvidia/parakeet_realtime_eou_120m-v1`, with end-of-utterance (EOU) detection — exposed as a streaming C-API/CLI that matches NeMo.

**Architecture (validated from the real model):** `EncDecRNNTBPEModel` — **pure RNNT** (reuses `pk::PredictionNet` + `pk::Joint` + `pk::rnnt_greedy`). EOU is **not a head**: `<EOU>` (id 1024) and `<EOB>` (id 1025) are special vocab tokens (vocab 1026) the transducer emits. The new work is encoder-side: `conv_norm_type=layer_norm` (we only have batch_norm), **causal** conv + `causal_downsampling=True`, and **chunked-limited attention** (`att_context_size=[70,1]`, `att_context_style=chunked_limited`). Cache-aware streaming is built so **streaming output == offline-with-limited-context output numerically** — so we validate the new numerics *offline* first (5a), then build the true chunk-by-chunk caches (5b) and validate streaming == offline, then EOU + the streaming API (5c).

**Real config of `parakeet_realtime_eou_120m-v1`:** feat_in 128 (mel 128), n_layers 17, d_model 512, n_heads 8, ff×4, subsampling dw_striding ÷8 (causal_downsampling=True), conv_kernel 9, **conv_norm_type=layer_norm**, conv_context_size=causal, **att_context_size=[70,1]**, att_context_style=chunked_limited, xscaling=False, vocab 1026 (incl `<EOU>`=1024, `<EOB>`=1025), RNNT joint (no durations). `streaming_cfg`: chunk_size=[9,16], shift_size=[9,16], cache_drop_size=0, last_channel_cache_size=70, valid_out_len=2, pre_encode_cache_size=[0,9], drop_extra_pre_encoded=2.

**Tech Stack:** C++17, ggml v0.13.0, `.venv` NeMo, ctest. Branch `impl/phase-5-streaming`.

**Reference (read first):**
- Spec §2.1 (streaming deferred), and the Phase 5 trace in this plan.
- NeMo (cite while implementing): `nemo/collections/asr/modules/conformer_encoder.py` (`forward_internal` 617-779, `setup_streaming_params` 977-1085, `get_initial_cache_state` 1087-1125, `streaming_post_process` 548-571, `_create_masks` 814-895), `parts/submodules/multi_head_attention.py` (`update_cache` 204-209, RelPosMHA fwd), `parts/submodules/causal_convs.py` (`CausalConv1D.update_cache` 131-151), `parts/mixins/streaming.py` (`cache_aware_stream_step` 40-76), `models/asr_eou_models.py` (`_get_eou_predictions_from_hypotheses` 114-168 — EOU = emitting token `<EOU>`).
- Existing code: `pk::Encoder`, `pk::ConformerLayer` (asserts batch_norm — to extend), `pk::RelPosAttention`, `pk::Subsampling`, `pk::PredictionNet`, `pk::Joint`, `pk::rnnt_greedy`, `pk::Model`, `pk::ModelLoader`, the C-API (`include/parakeet_capi.h`), `scripts/gen_nemo_baseline.py`, `scripts/validate_vs_nemo.py`.

**Conventions:** ctest 0/77/fail; commit per green step; CPU-only; `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

---

## Orientation + Entry gate (Phase 4 complete)
Read spec + this plan. Verify env. Resume via `git log`. Entry gate: `cmake -B build -DPARAKEET_BUILD_TESTS=ON -DGGML_NATIVE=ON && cmake --build build -j`; convert `parakeet-tdt_ctc-110m` + gen baselines; `ctest` ⇒ 25 tests (22 pass + 3 big-model skips). The streaming model `nvidia/parakeet_realtime_eou_120m-v1` is cached (~480MB). If not green, STOP.

---

# Part 5a — Streaming-encoder numerics, validated OFFLINE

## Task 1: Streaming config in converter/loader + streaming baseline
**Files:** `scripts/convert_parakeet_to_gguf.py`, `src/model_loader.{hpp,cpp}`, `examples/cli/main.cpp` (info), `scripts/gen_nemo_baseline.py`, `docs/conversion.md`.
- [ ] **Step 1:** Converter — emit new KV: `parakeet.encoder.att_context_left`/`att_context_right` (from `att_context_size`), `parakeet.encoder.att_context_style` (str), `parakeet.encoder.causal_downsampling` (bool), and the streaming params (`parakeet.streaming.{chunk_size,shift_size,cache_drop_size,last_channel_cache_size,valid_out_len,pre_encode_cache_size,drop_extra_pre_encoded}` — read from `m.encoder.streaming_cfg` after `setup_streaming_params`, or from cfg). `conv_norm_type` (=`layer_norm` here) is already emitted; `conv_context_size=causal` → emit `parakeet.encoder.conv_causal=true`. Keep verbatim tensor names (the RNNT weights load as-is).
- [ ] **Step 2:** Loader — read these into `ParakeetConfig` (fields: `att_context_left/right`, `att_context_style`, `causal_downsampling`, `conv_causal`, plus a `StreamingCfg` sub-struct). `info` prints them.
- [ ] **Step 3:** Baseline dumper — add a mode/flag to dump for `parakeet_realtime_eou_120m-v1` (offline): `mel`, `subsampling_out`, `enc_layer_0`, `l0_conv_out`, `encoder_out` (the model's normal offline forward already uses limited-context+causal+layer_norm), the RNNT greedy token ids (`rnnt_token_ids`, incl `<EOU>`), and the text (`baseline.rnnt_text`). Run on `tests/fixtures/speech.wav` → `/tmp/baseline_eou.gguf`.
- [ ] **Step 4:** Convert the model (`/tmp/eou.gguf`), `info` shows the streaming config; commit: `git commit -m "feat: streaming config (att_context/causal/layer_norm/streaming params) in converter+loader + EOU baseline"`.

## Task 2: layer_norm conv + causal conv in the conformer conv module
**Files:** `src/conformer.cpp`, `tests/test_conformer_eou.cpp`.
- [ ] **Step 1:** Generalize `ConformerLayer`'s conv module: support `conv_norm_type=="layer_norm"` (LayerNorm over the channel dim between depthwise conv and SiLU; tensor names `conv.norm.{weight,bias}` for layer_norm vs `conv.batch_norm.*` — VERIFY the actual names in the eou GGUF) in addition to the existing batch_norm path. And support **causal depthwise conv** (`conv_causal`): left pad = `conv_kernel-1`, right pad = 0 (vs symmetric `(k-1)/2`). Gate both on config; keep the batch_norm + symmetric path unchanged for the existing models. Remove the hard `assert(batch_norm)`.
- [ ] **Step 2:** `tests/test_conformer_eou.cpp`: parity of conformer layer 0 vs `enc_layer_0` from `/tmp/baseline_eou.gguf` (atol/rtol 3e-2), and the conv submodule vs `l0_conv_out`. Skip 77 unless `PARAKEET_TEST_GGUF_EOU` + `PARAKEET_TEST_BASELINE_EOU` set. Build + validate; debug per-submodule if off (suspect layer_norm axis or causal padding). Commit: `git commit -m "feat: layer_norm + causal conv in conformer (streaming models)"`.

## Task 3: Causal subsampling + chunked-limited attention mask + offline encoder parity
**Files:** `src/subsampling.cpp`, `src/relpos_attention.cpp`/`src/encoder.cpp`, `tests/test_encoder_eou.cpp`.
- [ ] **Step 1:** `causal_downsampling`: the dw_striding subsampling uses causal (left) padding when `cfg.causal_downsampling`. VERIFY against NeMo `ConvSubsampling` causal branch; keep the existing non-causal path for other models.
- [ ] **Step 2:** **Chunked-limited attention mask:** in offline mode the encoder must apply the same mask NeMo uses for `att_context_style=chunked_limited`, `att_context_size=[70,1]` — frame `i` attends to `[chunk*chunk_size - left, chunk*chunk_size + chunk_size-1 + right]` (trace `_create_masks` 814-895 for the exact chunk/window formula with left=70,right=1). Add this masked path to the attention/encoder (gate on `att_context_style != "regular"`), building an additive -inf mask over the scores. Keep the full-context path for existing models.
- [ ] **Step 3:** `tests/test_encoder_eou.cpp`: full offline encoder parity vs `encoder_out` in `/tmp/baseline_eou.gguf` (atol/rtol 5e-2), with mid/last-layer localization. Skip-if-absent. Validate; debug (mask formula / causal conv / layer_norm accumulation). Commit: `git commit -m "feat: causal subsampling + chunked-limited attention; offline encoder parity (eou model)"`.

## Task 4: Offline end-to-end (5a milestone)
**Files:** `tests/test_transcribe_eou.cpp`, `docs/parity.md`.
- [ ] **Step 1:** `pk::transcribe(eou.gguf, speech.wav)` (arch=rnnt → rnnt_greedy, limited-context encoder) → assert it equals NeMo's offline transcript for this model (incl. how `<EOU>`/`<EOB>` appear — decide: strip them from text + expose separately, OR keep raw; match what `baseline.rnnt_text` / NeMo `transcribe` returns). Token-id parity vs `rnnt_token_ids`. Skip-if-absent test. Validate WER 0 vs NeMo.
- [ ] **Step 2:** Record in `docs/parity.md` (streaming model, offline-limited-context, WER). Commit: `git commit -m "feat: offline transcription of parakeet_realtime_eou_120m-v1 (WER 0 vs NeMo)"`.

---

# Part 5b — True cache-aware chunk-by-chunk streaming

## Task 5: Streaming encoder (conv + attention caches, chunk step)
**Files:** `src/streaming_encoder.{hpp,cpp}` (or extend `encoder`), `tests/test_streaming_encoder.cpp`. Baseline: extend `gen_nemo_baseline.py` to dump NeMo's `cache_aware_stream_step` per-chunk outputs for the speech clip (`stream_chunk_outs`, the concatenated streaming encoder output) so we can diff.
- [ ] **Step 1:** Implement the streaming step from the trace: maintain `cache_last_time` (per-layer conv left-context `[d_model, left_pad]`), `cache_last_channel` (per-layer attention left-context `[last_channel_cache_size=70, d_model]`), `cache_last_channel_len`, and the pre-encode cache (`pre_encode_cache_size`, `drop_extra_pre_encoded`). Per chunk: prepend conv cache in each layer's depthwise conv (causal), prepend attention KV cache, run the layers, slice `valid_out_len` outputs, update caches (drop oldest, append current). Use the `streaming_cfg` values from KV. `get_initial_cache_state` = zeros.
- [ ] **Step 2:** `class StreamingEncoder { StreamingEncoder(const ModelLoader&); void reset(); std::vector<float> step(const std::vector<float>& mel_chunk, ...); };` (emits the valid encoder frames for that chunk).
- [ ] **Step 3:** `tests/test_streaming_encoder.cpp`: feed the speech-clip mel in chunks → concatenate the per-step encoder outputs → assert it equals the OFFLINE `encoder_out` (the cache-aware equivalence property) within tolerance, AND matches NeMo's `stream_chunk_outs`. Skip-if-absent. Debug: a mismatch localizes to cache update (drop/append indices), the pre-encode cache, or the per-chunk attention mask. Commit: `git commit -m "feat: cache-aware streaming encoder (conv+attn caches) == offline"`.

## Task 6: Streaming decode loop + carried RNNT state
**Files:** `src/streaming.{hpp,cpp}` (a `pk::StreamingSession`), `tests/test_streaming_decode.cpp`.
- [ ] **Step 1:** `pk::StreamingSession` owns a `StreamingEncoder` + the RNNT decoder state (prediction-net `PredState` + last token + accumulated hypothesis). Per chunk: `StreamingEncoder::step` → for each new encoder frame run the RNNT inner loop (emit-until-blank, carrying decoder state across chunks; do NOT reset between chunks) → append tokens. (Mirror NeMo `rnnt_decoder_predictions_tensor(..., partial_hypotheses=...)`.)
- [ ] **Step 2:** `tests/test_streaming_decode.cpp`: stream the speech clip → assert the full emitted token-id sequence equals the offline `rnnt_token_ids` (and NeMo streaming) EXACTLY. Skip-if-absent. Commit: `git commit -m "feat: streaming RNNT decode with carried state (== offline tokens)"`.

---

# Part 5c — EOU + streaming API

## Task 7: EOU detection + streaming C-API + CLI
**Files:** `src/streaming.cpp` (EOU), `include/parakeet_capi.h` + `src/parakeet_capi.cpp` (streaming session API), `examples/cli/main.cpp` (`transcribe --stream`), `tests/test_capi_stream.cpp`, `docs/parity.md`, README/AGENTS.
- [ ] **Step 1: EOU.** In `StreamingSession`, detect emission of `<EOU>` (id from KV; find by piece `"<EOU>"`) and `<EOB>`: expose an EOU event (with the encoder-frame index / time = frame×hop×subsampling) and strip `<EOU>`/`<EOB>` from the emitted text (surface them as events, not text). Confirm the time/segmentation matches NeMo's `_get_eou_predictions_from_hypotheses`.
- [ ] **Step 2: Streaming C-API** (extend `parakeet_capi.h`):
```c
typedef struct parakeet_stream parakeet_stream;
parakeet_stream* parakeet_capi_stream_begin(parakeet_ctx* ctx);
// feed 16k mono f32 PCM; returns newly-finalized text since last call (malloc'd; "" if none). Sets *eou=1 if EOU emitted in this feed.
char* parakeet_capi_stream_feed(parakeet_stream* s, const float* pcm, int n_samples, int* eou_out);
char* parakeet_capi_stream_finalize(parakeet_stream* s);   // flush tail
void  parakeet_capi_stream_free(parakeet_stream* s);
```
Wrap `pk::StreamingSession`; no-throw boundary; malloc'd strings. The session buffers PCM into encoder chunks (chunk_size×hop samples) and runs steps as enough audio arrives.
- [ ] **Step 3: CLI** `parakeet-cli transcribe --model eou.gguf --input speech.wav --stream`: feed the file in real-time-sized chunks through the session, printing partial text incrementally and `[EOU]` markers when detected. Keep non-stream `transcribe` unchanged.
- [ ] **Step 4: `tests/test_capi_stream.cpp`:** stream `speech.wav` via the C-API → assert the concatenated text equals the offline transcript and that an `<EOU>` event fires (if the clip ends an utterance). Skip-if-absent. Compare EOU timing/text to NeMo.
- [ ] **Step 5:** Record streaming results in `docs/parity.md`; update README/AGENTS (streaming usage, the new `parakeet_capi_stream_*` symbols, EOU semantics). Commit: `git commit -m "feat: EOU detection + streaming C-API + CLI --stream (matches NeMo)"`.

---

## Phase 5 done-when
- Offline (5a): `parakeet_realtime_eou_120m-v1` transcribes `speech.wav` at WER 0 vs NeMo with layer_norm conv + causal conv + chunked-limited attention; encoder/conformer offline parity tests pass.
- Streaming (5b): the cache-aware streaming encoder output == offline encoder output, and streaming decode tokens == offline tokens == NeMo streaming.
- EOU + API (5c): `<EOU>`/`<EOB>` detected as events; streaming C-API (`parakeet_capi_stream_*`) + `parakeet-cli transcribe --stream` work and match NeMo; `test_capi_stream` passes.
- Full local suite green (new eou/streaming tests skip cleanly without `PARAKEET_TEST_GGUF_EOU` etc.); no regression on the offline models.
- `docs/parity.md` + README/AGENTS updated.

## Handoff
Phase 5 completes the streaming roadmap. This is the last planned phase — parakeet.cpp would then cover the full offline + streaming Parakeet family. Per the chaining convention, any further phase opens with Orientation + this done-when as its entry gate.
