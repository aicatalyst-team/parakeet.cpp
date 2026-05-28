# parakeet.cpp Phase 4 — Productionize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the working, NeMo-faithful engine shippable: a load-once flat C-API + shared library for dlopen/LocalAI, model quantization (F16 / Q8_0 / K-quants) with a WER parity gate, an HF model-publishing script, and a closed-loop CI job.

**Architecture:** Refactor transcription into a load-once `pk::Model` (build the component objects once, transcribe many) so the flat C-API can hold a context. Quantization is applied **only to the large linear weights consumed via `ggml_mul_mat`** (encoder FFN + attention, joint, CTC head); everything the hand-rolled C++ reads directly as F32 (mel filterbank, LSTM weights, batch_norm stats, conv kernels, norms, biases, pos_bias) stays F32. A WER-vs-NeMo parity gate guards every quantization type.

**Tech Stack:** C++17, ggml v0.13.0, `.venv` NeMo, ctest, HuggingFace Hub. Branch `impl/phase-4-productionize`.

**Reference (read first):**
- Spec `docs/superpowers/specs/2026-05-28-parakeet-cpp-design.md` (§7 quant, §10 C-API/build).
- Sibling C-API + publish patterns: `/home/mudler/_git/rt-detr.cpp/include/rfdetr_capi.h`, `/home/mudler/_git/rt-detr.cpp/src/rfdetr_capi.cpp`, `scripts/publish_hf.py`, `.github/workflows/ci.yml`; and `/home/mudler/_git/vibevoice.cpp` for the quantize-gguf approach.
- Existing code: `pk::transcribe(model_path, wav, Decoder)` (`src/parakeet.cpp` — currently loads per call), `pk::ModelLoader`, the component classes (`MelFrontend`, `Encoder`, `CTCDecoder`, `PredictionNet`, `Joint`, `tdt_greedy`, `rnnt_greedy`, `ctc_greedy`, `detokenize`), `scripts/validate_vs_nemo.py`, CLI (`src/.../examples/cli/main.cpp`: `info`, `transcribe`).

**Conventions:** ctest 0/77/fail; commit per green step; CPU-only; `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

## Disk / models
~180 GB free. Anchor for tests: `nvidia/parakeet-tdt_ctc-110m` (cached; `/tmp/pk110m.gguf`). For quantization parity, also spot-check one larger model (e.g. `parakeet-tdt-0.6b-v2`) — convert/prune per the Phase 3.5 DISK note. Keep `/tmp/pk110m.gguf` + `/tmp/baseline*.gguf` (committed suite depends on them).

---

## Orientation + Entry gate (Phase 3.5 complete)
Read spec + this plan. Verify env. Resume via `git log`. Entry gate:
```bash
cmake -B build -DPARAKEET_BUILD_TESTS=ON -DGGML_NATIVE=ON && cmake --build build -j
.venv/bin/python scripts/convert_parakeet_to_gguf.py --model nvidia/parakeet-tdt_ctc-110m --output /tmp/pk110m.gguf
.venv/bin/python scripts/gen_nemo_baseline.py --model nvidia/parakeet-tdt_ctc-110m --audio tests/fixtures/clip.wav --output /tmp/baseline.gguf
.venv/bin/python scripts/gen_nemo_baseline.py --model nvidia/parakeet-tdt_ctc-110m --audio tests/fixtures/speech.wav --output /tmp/baseline_speech.gguf
export PARAKEET_TEST_GGUF=/tmp/pk110m.gguf PARAKEET_TEST_BASELINE=/tmp/baseline.gguf PARAKEET_TEST_BASELINE_SPEECH=/tmp/baseline_speech.gguf
ctest --test-dir build --output-on-failure   # expect 24 tests, 21 pass + 3 big-model skips
```
If not green, STOP.

---

## File structure
```
src/
  model.hpp / model.cpp          # NEW: load-once pk::Model (owns loader + components), transcribe(audio, decoder)
  parakeet.cpp                   # transcribe(path,wav) becomes a thin wrapper over pk::Model
  parakeet_capi.cpp              # NEW: flat C-API impl
include/
  parakeet_capi.h                # NEW: flat C-API (dlopen / LocalAI)
examples/cli/main.cpp            # ADD: `quantize` subcommand
scripts/
  publish_hf.py                  # NEW: convert+quantize variants, upload to HF (user-run)
  (convert_parakeet_to_gguf.py)  # EXTEND: --dtype f32|f16|q8_0 + should_quantize policy
tests/
  test_capi.cpp                  # NEW: C-API load → transcribe → free
  test_quant_parity is exercised via validate_vs_nemo.py / a ctest on the 110m
models/MANIFEST.md               # NEW: expected published variant set
.github/workflows/ci.yml         # EXTEND: closed-loop dispatch job
docs/quantization.md             # NEW: which tensors quantize + measured WER per type
```

---

## Task 1: Load-once `pk::Model` + flat C-API + shared lib

**Files:** Create `src/model.hpp`, `src/model.cpp`, `include/parakeet_capi.h`, `src/parakeet_capi.cpp`, `tests/test_capi.cpp`. Modify `src/parakeet.cpp`, `CMakeLists.txt`, `tests/CMakeLists.txt`.

- [ ] **Step 1: `pk::Model`** — a class that loads a GGUF ONCE and builds the reusable component objects (`ModelLoader`, `MelFrontend`, `Encoder`, decoders), exposing:
  `std::string transcribe(const std::vector<float>& pcm16k_mono, Decoder decoder = Decoder::kDefault) const;`
  and a convenience `std::string transcribe_path(const std::string& wav_path, Decoder) const;` (uses `load_audio_16k_mono`). Move the orchestration logic currently inside `pk::transcribe(path,wav,decoder)` into `pk::Model`; rewrite `pk::transcribe(path,wav,decoder)` to construct a `pk::Model` and call it (keeps existing tests passing). This avoids reloading the model per call — essential for the C-API.
- [ ] **Step 2: Flat C-API** `include/parakeet_capi.h` (extern "C", opaque handle, no exceptions across the boundary):
```c
typedef struct parakeet_ctx parakeet_ctx;
int           parakeet_capi_abi_version(void);                 // bump on breaking change
parakeet_ctx* parakeet_capi_load(const char* gguf_path);       // NULL on failure
void          parakeet_capi_free(parakeet_ctx* ctx);
// decoder: 0=default(by arch), 1=ctc, 2=tdt/rnnt(transducer). Returns malloc'd UTF-8 (free with parakeet_capi_free_string), NULL on error.
char*         parakeet_capi_transcribe_path(parakeet_ctx* ctx, const char* wav_path, int decoder);
char*         parakeet_capi_transcribe_pcm (parakeet_ctx* ctx, const float* samples, int n_samples, int sample_rate, int decoder);
void          parakeet_capi_free_string(char* s);
const char*   parakeet_capi_last_error(parakeet_ctx* ctx);     // human-readable last error, or ""
```
`src/parakeet_capi.cpp`: implement over `pk::Model` (the ctx wraps a `pk::Model` + a last-error string). `transcribe_pcm` resamples if `sample_rate != 16000` (reuse `pk::resample_linear`). Catch all exceptions → return NULL + set last_error. `decoder` int maps to `pk::Decoder`.
- [ ] **Step 3: Shared lib build.** Ensure `PARAKEET_SHARED=ON` builds `libparakeet.so` exporting the `parakeet_capi_*` symbols (and the existing `parakeet_*` ones). Verify `nm -D` shows them.
- [ ] **Step 4: `tests/test_capi.cpp`** — `parakeet_capi_load($PARAKEET_TEST_GGUF)` → `parakeet_capi_transcribe_path(ctx, "tests/fixtures/speech.wav", 2)` (TDT) → assert the returned string equals the known 110m TDT transcript (`Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.`) → `parakeet_capi_free_string` + `parakeet_capi_free`. Skip 77 if `PARAKEET_TEST_GGUF` unset. LABEL model. Also assert `parakeet_capi_load("/nonexistent")` returns NULL (no crash). Register it.
- [ ] **Step 5: Build + run + symbol check**
```
cmake -B build -DPARAKEET_BUILD_TESTS=ON -DGGML_NATIVE=ON && cmake --build build -j
PARAKEET_TEST_GGUF=/tmp/pk110m.gguf ctest --test-dir build -R test_capi --output-on-failure
cmake -B build-shared -DPARAKEET_SHARED=ON -DPARAKEET_BUILD_CLI=ON && cmake --build build-shared -j
nm -D build-shared/libparakeet.so | grep parakeet_capi   # confirm exports
```
Run the full suite (no regression).
- [ ] **Step 6: Commit** — `git commit -m "feat: load-once pk::Model + flat C-API (parakeet_capi) + shared lib"`

---

## Task 2: Quantization audit + converter F16 + should_quantize policy

**Files:** Modify `scripts/convert_parakeet_to_gguf.py`; create `docs/quantization.md`. Possibly minor `ModelLoader`/component adjustments if the audit finds a blocker.

- [ ] **Step 1: Audit which weights are `ggml_mul_mat`-consumed vs hand-read.** Read the component sources and classify every loaded tensor:
  - **Quantizable** (passed to `ggml_mul_mat` in a graph, ggml dequantizes on the fly): encoder per-layer `feed_forward{1,2}.linear{1,2}.weight`, attention `linear_{q,k,v,out,pos}.weight`, `joint.{enc,pred,joint_net.2}.weight`, the CTC head weight (`ctc_decoder`/`decoder`.`decoder_layers.0.weight`). **VERIFY** each is passed directly to `ggml_mul_mat` and NOT manually `ggml_cont`/transposed/`get_data_f32`'d in a way that breaks on f16/quantized storage. If a path does `ggml_cont` on the weight, either (a) transpose at convert-time so the runtime needs no cont, or (b) keep that tensor F32. Document the final allowlist.
  - **Must stay F32** (hand-read by C++ via `ggml_get_data_f32`/pointer, or used by ops that don't support quant): `preprocessor.featurizer.{fb,window}`, all `decoder.prediction.*` (LSTM, hand-rolled), all `*norm*` weights/biases, all `*.bias`, `pos_bias_{u,v}`, subsampling `encoder.pre_encode.*` conv kernels, depthwise `conv.depthwise_conv.*`, `conv.batch_norm.*`, embeddings.
- [ ] **Step 2: `--dtype` in the converter** — add `--dtype {f32,f16,q8_0}` (default f32). A `should_quantize(name, shape, dtype)` returns the target ggml type for a tensor: the requested dtype for allowlisted linear weights with both dims ≥ 32 and divisible by the type's block size (32 for q8_0); F32 otherwise. Write each tensor in its chosen type (the `gguf` Python writer supports f16 and q8_0/q4_0/q5_0; use it). Keep verbatim names.
- [ ] **Step 3: Parity gate.** Convert the 110m to f16 and to q8_0; run `scripts/validate_vs_nemo.py --model nvidia/parakeet-tdt_ctc-110m --gguf /tmp/pk110m_f16.gguf --head tdt` (and q8_0). F16 should be WER 0; Q8_0 WER 0 or near-0. ALSO run the C++ component parity tests against an f16/q8_0 GGUF where meaningful (e.g. point `test_encoder`/`test_ctc` — note: the baseline is f32, so expect a slightly larger but still-small diff; if the per-stage tolerance is too tight for q8_0, rely on the end-to-end WER gate instead and note it). Spot-check one larger model (`parakeet-tdt-0.6b-v2`) at q8_0 → WER. If any tensor in the allowlist breaks the run, remove it from the allowlist (keep F32) and re-test.
- [ ] **Step 4: `docs/quantization.md`** — the allowlist (what gets quantized vs stays F32 and why), the supported `--dtype` values, measured size + WER per type for the 110m and the 0.6B model, and the rule that conv/LSTM/featurizer tensors stay F32.
- [ ] **Step 5: Commit** — `git commit -m "feat: converter f16/q8_0 quantization (mul_mat linear weights only) + parity gate + docs"`

---

## Task 3: CLI `quantize` (K-quants)

**Files:** Modify `examples/cli/main.cpp`; possibly `src/` for a quantize helper.

- [ ] **Step 1:** Add `parakeet-cli quantize <in.gguf> <out.gguf> <type>` where type ∈ `{q4_0,q5_0,q8_0,q4_k,q5_k,q6_k}`. Read the input GGUF, re-quantize the **same allowlisted linear tensors** (apply the Task 2 policy — quantize only those, copy everything else verbatim) to the target ggml type via `ggml_quantize_chunk` (verify the API in `third_party/ggml/include/ggml.h`), and write a new GGUF (copy all KV unchanged). This enables K-quants (`Q4_K`/`Q5_K`/`Q6_K`) that the Python `gguf` writer can't produce. Tensors not in the allowlist are written as-is (F32).
- [ ] **Step 2: Parity.** Quantize `/tmp/pk110m.gguf` (f32) → q4_k, q6_k; run `validate_vs_nemo.py` end-to-end on each → record WER (q6_k ~0; q4_k small WER acceptable, report it). Confirm `parakeet-cli info` still reads the quantized GGUF.
- [ ] **Step 3:** Update `docs/quantization.md` with the CLI usage + K-quant WER/size numbers. Commit: `git commit -m "feat: parakeet-cli quantize (K-quants) with parity"`. Run full suite.

---

## Task 4: HF publishing script + manifest (artifact — user runs upload)

**Files:** Create `scripts/publish_hf.py`, `models/MANIFEST.md`.

- [ ] **Step 1:** `scripts/publish_hf.py` modeled on `/home/mudler/_git/rt-detr.cpp/scripts/publish_hf.py`: for a given source HF model id, convert to the variant set (f16 + q8_0 via converter; q4_k via CLI quantize), and upload the GGUFs + a per-model README to a target HF repo (e.g. `mudler/parakeet.cpp-<variant>`). Use `huggingface_hub`; require a token at `~/.cache/huggingface/token`. Make it **dry-run by default** (`--upload` to actually push) so it can't accidentally publish. Do NOT run the upload in this task — just build + lint the script and verify the convert+quantize steps it calls work.
- [ ] **Step 2:** `models/MANIFEST.md` — the expected published set (per checkpoint × {f16, q8_0, q4_k}), with the validated WER from `docs/parity.md`. `models/` stays gitignored for the GGUFs themselves.
- [ ] **Step 3:** Commit: `git commit -m "feat: HF publishing script (dry-run default) + model manifest"`. (Actual upload is user-run with their HF token.)

---

## Task 5: Closed-loop CI job + productionization docs

**Files:** Modify `.github/workflows/ci.yml`, `README.md`, `AGENTS.md`.

- [ ] **Step 1:** Add a `closed-loop` job to `.github/workflows/ci.yml`, gated on `workflow_dispatch` (not every push — it needs network + a model). It: sets up the `.venv` (or downloads a published GGUF from HF), builds, converts/downloads the 110m, runs `parakeet-cli transcribe` on `tests/fixtures/speech.wav`, and asserts it matches the committed reference transcript. Keep the existing `build` job (model-independent ctests) as the per-push gate.
- [ ] **Step 2:** Update `README.md` (build, quantization, C-API usage, model coverage) and `AGENTS.md` (the C-API symbols LocalAI depends on, the quantization policy, the publish flow) — mirror the structure of the rt-detr.cpp docs. Note the LocalAI backend itself lives in the LocalAI repo and dlopens `libparakeet.so` via `parakeet_capi.h`.
- [ ] **Step 3:** Commit: `git commit -m "ci: closed-loop dispatch job; docs: C-API + quantization + coverage"`. Run the full local suite.

---

## Phase 4 done-when
- `libparakeet.so` exports `parakeet_capi_*`; `test_capi` passes (load → transcribe → free) on the anchor.
- Converter produces working **f16** and **q8_0** GGUFs; `parakeet-cli quantize` produces working **K-quant** GGUFs; each passes the WER-vs-NeMo gate (WER documented per type in `docs/quantization.md`).
- `scripts/publish_hf.py` (dry-run) + `models/MANIFEST.md` exist; closed-loop CI job added.
- Full local suite green (new `test_capi` runs on the anchor; big-model/quant checks skip cleanly without their inputs).

## Handoff
Phase 4 completes v1 productionization. Future work (Phase 5): cache-aware streaming + EOU (`parakeet_realtime_eou_120m-v1`) — chunked attention + conv/attention caches + the layer_norm conv path. Per the chaining convention (Phase 0 plan), a Phase 5 plan would open with Orientation + this phase's done-when as its entry gate.
