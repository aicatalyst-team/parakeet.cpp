# parakeet.cpp — Close the NeMo speed gap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Make the C++/ggml engine match (or beat) NeMo's CPU speed, without regressing accuracy (all parity tests stay green; WER vs NeMo stays ~0). Currently ~3–4× slower than NeMo.

**Root cause (profiled):** the inference path builds **hundreds of tiny ggml graphs per utterance** (272 on the 110m for a 10s clip — ~5 per conformer layer × N layers + one per RNNT/TDT joint step), each doing a full `ggml_init` → build → `ggml_graph_compute_with_ctx` → `ggml_free`. ~40% of wall time is allocator churn (`init+alloc` 29–37%, `free` 8–10%), and the tiny graphs don't parallelize (threads barely help). This is exactly the wall rf-detr.cpp hit; their fix was a **persistent ggml graph allocator + persistent CPU backend**, which closed the gap (~3×, see `/home/mudler/_git/rt-detr.cpp/BENCHMARK.md` "persistent gallocr" and `src/backend.cpp`).

**Strategy (rf-detr's proven playbook, in impact order):** persistent CPU backend + `ggml_gallocr` reused across inferences (kills churn + per-call threadpool spawn) → **fuse the encoder into one graph** (so compute parallelizes) → reuse the backend for the decoder's per-step joint → build flags (GGML_NATIVE/LLAMAFILE + tinyBLAS broadcast-fold patch). **Guardrail:** the existing parity tests (mel/subsampling/relpos/conformer/encoder/ctc/transcribe/tdt/streaming, all matching NeMo) MUST stay green at every step — they are how we refactor the graph safely.

**Tech Stack:** C++17, ggml v0.13.0 (`third_party/ggml`), `.venv` NeMo, the benchmark harness (Tasks from the bench plan: `parakeet-cli bench`, `scripts/benchmark.py`, `plot_benchmark.py`). Branch `perf/close-nemo-gap` (forked from the benchmark work).

**Reference:**
- `/home/mudler/_git/rt-detr.cpp/src/backend.cpp` (persistent `ggml_backend_cpu_init` + `ggml_gallocr` pattern; `BackendCtx` holding `galloc_*`), `third_party/ggml-patches/` (the two patches), `BENCHMARK.md` "Build-time optimizations that matter".
- ggml headers: `third_party/ggml/include/ggml-backend.h` (`ggml_backend_cpu_init`, `ggml_backend_cpu_set_n_threads`, `ggml_gallocr_new`, `ggml_gallocr_alloc_graph`, `ggml_backend_graph_compute`), `ggml-alloc.h`.
- Our inference path: `src/ggml_graph.{hpp,cpp}` (`run_graph` — the thing to replace), `src/{subsampling,relpos_attention,conformer,encoder,joint,ctc_decoder,streaming_encoder}.cpp` (callers), `src/model.cpp`, `src/prediction.cpp` (LSTM, plain C++ — no graph), `src/rnnt.cpp`/`src/tdt.cpp` (greedy loops calling the joint per step).

**Conventions:** ctest 0/77/fail; commit per green step; CPU-only; `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`. **At every task, the full parity suite (35 tests) must stay green AND transcripts byte-identical to before** (the refactor changes performance, not numerics).

---

## Orientation + Entry gate
Read plan + rf-detr `src/backend.cpp` + `BENCHMARK.md`. Verify env + build green (35 tests). Establish the BEFORE baseline: `parakeet-cli bench` on the 110m over the LibriSpeech manifest → record RTFx (the number to beat). The env-gated `PARAKEET_GRAPH_PROFILE=1` in `src/ggml_graph.cpp` prints the run_graph call count + init/compute/free split — use it to measure progress at each step.

## Task 0: Build-flag wins (cheap, do first)
**Files:** `CMakeLists.txt` (+ maybe apply the ggml patches).
- [ ] Confirm/enable ggml build flags that the benchmark binary uses: `GGML_NATIVE=ON` (AVX-512), `GGML_LLAMAFILE=ON` (tinyBLAS SGEMM — check ggml's default in our build; force ON if not). Verify the CLI is actually compiled with them (the benchmark must rebuild after). Optionally `GGML_OPENMP=ON`.
- [ ] Apply rf-detr's `0001-ggml-cpu-fold-broadcast-iterations-in-llamafile_sgem.patch` (copy to `third_party/ggml-patches/` + a `scripts/apply_ggml_patches.sh` applied at configure time, mirroring rf-detr) — only if it applies cleanly to our ggml SHA; if it conflicts, skip and note it.
- [ ] Re-bench the 110m; record the delta. Commit: `git commit -m "perf: enable GGML_NATIVE/LLAMAFILE + tinyBLAS broadcast-fold patch"`.

## Task 1: Persistent CPU backend + gallocr (`pk::Backend`)
**Files:** `src/backend.{hpp,cpp}` (new), `src/ggml_graph.{hpp,cpp}` (replace/augment), `src/model.{hpp,cpp}`.
- [ ] `src/backend.hpp`: a `pk::Backend` holding a `ggml_backend_t` (`ggml_backend_cpu_init`, `ggml_backend_cpu_set_n_threads(b, n)`) + one or more reusable `ggml_gallocr_t` (compute-buffer allocators), created once and reused. Mirror `/home/mudler/_git/rt-detr.cpp/src/backend.cpp` (init/free order: free gallocrs before backend).
- [ ] Provide a new graph-run primitive that REUSES the backend + gallocr: `bool Backend::compute(const std::function<ggml_tensor*(ggml_context*)>& build, std::vector<float>& out, size_t graph_overhead_bytes)` that: builds the graph in a `no_alloc=true` context (only metadata, no inline data), `ggml_gallocr_alloc_graph(galloc, gf)` (reuses scratch across calls — DOES NOT realloc when the shape is stable), sets the input tensor data after alloc, `ggml_backend_graph_compute(backend, gf)`, copies output. Thread count from `pk::num_threads()`/`--threads` via `ggml_backend_cpu_set_n_threads`. This replaces the per-call `ggml_init`+`ggml_graph_compute_with_ctx`+`ggml_free` churn.
- [ ] Wire `pk::Model` to own a `pk::Backend` (created at load with the configured thread count). The simplest migration: keep the `run_graph(...)` signature but route it through a process-/model-global `pk::Backend` so existing callers work unchanged initially (then Task 2 fuses). Ensure inputs are written AFTER `ggml_gallocr_alloc_graph` (with no_alloc, tensor->data is null until alloc).
- [ ] **Guardrail:** full suite green + transcripts unchanged. Re-bench 110m + profile (the init/free % should collapse). Commit: `git commit -m "perf: persistent CPU backend + gallocr (eliminate per-call allocator churn)"`.

## Task 2: Fuse the encoder into one graph
**Files:** `src/encoder.{hpp,cpp}`, `src/conformer.{hpp,cpp}`, `src/relpos_attention.{hpp,cpp}`, `src/subsampling.{hpp,cpp}`.
- [ ] Refactor the component `forward`s from "run a graph and return floats" into "append ops to a shared `ggml_context`/graph and return the output `ggml_tensor*`". The encoder builds ONE graph: subsampling → for each layer: FFN1+attn+conv+FFN2+LN (all in-graph, reusing `RelPosAttention`/conv as graph-builders, not graph-runners) → output, computed in a SINGLE `Backend::compute`. (Mel stays plain C++ → fed as the input tensor.) This is the structural win: ~85 encoder graphs → 1, so threads parallelize over the whole encoder.
- [ ] Keep per-component **unit parity tests** working: provide a thin wrapper so `test_subsampling`/`test_relpos_attention`/`test_conformer`/`test_encoder` can still build+run just their sub-graph via the Backend (each test builds its piece and computes it). The graph-builder functions are reusable in both the fused encoder and the unit tests.
- [ ] **Guardrail:** every encoder-related parity test green (max|d| unchanged from before — the math is identical, only the graph grouping changed); `test_transcribe*` byte-identical. Re-bench + profile (run_graph/compute calls should drop to ~1/encoder; threads should now scale). Commit: `git commit -m "perf: fuse FastConformer encoder into a single ggml graph"`.

## Task 3: Decoder joint — reuse backend, precompute projections
**Files:** `src/joint.{hpp,cpp}`, `src/rnnt.cpp`, `src/tdt.cpp`, `src/model.cpp`.
- [ ] The greedy loops call the joint per (t,u) — each was a fresh graph. Two wins: (a) **precompute `joint.enc` projection over all encoder frames once** (one graph: `[T, joint_hidden]`), reused for every u; (b) for the per-step joint (`relu(enc_proj[t] + pred_proj[u])` → output linear), either reuse `Backend::compute` (no churn) or — since it's tiny (a 640→joint_hidden and joint_hidden→V matmul) — compute it in plain C++ (it may beat ggml graph overhead for 1×N). Measure both; pick the faster. Prediction net (LSTM) is already plain C++.
- [ ] **Guardrail:** `test_joint`/`test_transducer_core`/`test_tdt_greedy`/`test_transcribe_tdt`/`test_streaming_decode` green; token sequences byte-identical. Re-bench (the 272→ small count). Commit: `git commit -m "perf: precompute joint enc-projection + churn-free per-step joint"`.

## Task 4: Streaming path + thread default
**Files:** `src/streaming_encoder.cpp`, `src/streaming.cpp`, `src/model.cpp`/CLI.
- [ ] Apply the same persistent-backend + fused-graph treatment to the streaming encoder step (it has the same per-layer churn). Keep `test_streaming_encoder`/`test_streaming_decode`/`test_capi_stream` green (streaming output still == offline == NeMo).
- [ ] Pick a good default thread count (rf-detr found a single-CCD count beats all-cores; sweep 1,2,4,8,16,20 with the now-fused graph and set a sensible default; expose `--threads`). Commit: `git commit -m "perf: persistent-backend streaming encoder + thread default"`.

## Task 5: Re-benchmark + write-up
**Files:** `benchmarks/results/*`, `benchmarks/plots/*`, `benchmarks/BENCHMARK.md`.
- [ ] Remove the temporary `PARAKEET_GRAPH_PROFILE` instrumentation from `src/ggml_graph.cpp` (or fold it into the Backend cleanly).
- [ ] Re-run the full benchmark (all 10 models, LibriSpeech + diverse, thread sweep) → regenerate plots → update `BENCHMARK.md` with before/after RTFx and a "How we closed the gap" section (persistent gallocr, fused graph, build flags — mirroring rf-detr's narrative). Confirm **agreement WER still ~0 across all models** (the refactor preserved correctness) and report the new speedup vs NeMo. Commit: `git commit -m "bench: post-optimization NeMo-vs-ggml results + how-we-closed-the-gap write-up"`.

---

## Done-when
- Parity suite green throughout; transcripts byte-identical to pre-optimization (correctness preserved — agreement WER vs NeMo still ~0 on the benchmark).
- run_graph/allocator churn eliminated (profile: per-call init/free ~0; encoder = 1 graph); threads scale.
- RTFx materially improved vs the baseline (target: match NeMo; rf-detr got ~3× from the same fixes) — documented in BENCHMARK.md with before/after.
