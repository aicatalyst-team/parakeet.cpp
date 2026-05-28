# parakeet.cpp — Design Spec

Date: 2026-05-28
Status: Approved (brainstorming) — pending implementation plan

## 1. Goal

A C++17 / [ggml](https://github.com/ggml-org/ggml) inference port of NVIDIA NeMo
**Parakeet** ASR models. A 1:1 functional port traced from the NeMo reference
implementation (`github.com/NVIDIA-NeMo/NeMo`, cloned locally at
`/home/mudler/_git/NeMo`).

Conventions mirror the sibling projects `rt-detr.cpp` and `vibevoice.cpp`:

- Stock ggml as a pinned submodule (no fork).
- C++17, no exceptions across the public API (status-code returns).
- Public flat C-API for dlopen / FFI, intended for a LocalAI native backend.
- Python converter: HuggingFace / `.nemo` → GGUF.
- ctest parity baselines diffed against the NeMo reference.
- Quantized GGUFs published to HuggingFace.

## 2. Architecture background (from NeMo trace)

Parakeet = a shared **FastConformer encoder** + one of three decoders. Variants
differ only in the decoder; the encoder is identical across them.

### Preprocessing — log-mel front end
NeMo `AudioToMelSpectrogramPreprocessor` / `FilterbankFeatures`:

| Param | Value |
| --- | --- |
| sample_rate | 16000 |
| features (n_mels) | 80 |
| n_fft | 512 |
| window | Hann, win_length 400 (0.025 s) |
| hop_length | 160 (0.01 s) |
| mag_power | 2.0 (power spectrogram) |
| log | yes, zero-guard add 2⁻²⁴ |
| preemph | 0.97 |
| mel norm | slaney |
| normalize | per-feature mean/var (offline default) |
| dither | forced 0 at inference (parity) |

**Validated against `nvidia/parakeet-ctc-0.6b`:** the restored config leaves
`mag_power`, `preemph`, `mel_norm`, and the log-guard keys as `None`, so NeMo
falls back to `FilterbankFeatures` defaults (mag_power 2.0, preemph 0.97,
mel_norm slaney, log_zero_guard add 2⁻²⁴). The converter therefore reads the
**effective** attributes off the instantiated `m.preprocessor.featurizer`, not
the raw cfg, and the baseline dumper sets `featurizer.dither = 0.0` for a
deterministic reference.

### FastConformer encoder
- **Subsampling:** `dw_striding`, factor **8** (3× stride-2 stages), depthwise
  3×3 + pointwise convs + ReLU; conv channels (e.g. 256) → `d_model`.
- **Conformer block order:** FFN (½-residual) → rel-pos MHSA → conv module →
  FFN (½-residual) → final LayerNorm. All norms are LayerNorm. Activation SiLU
  (swish). Input scaled by √d_model (`xscaling`).
- **Self-attention:** `RelPositionMultiHeadAttention` (Transformer-XL style)
  with learned `pos_bias_u` / `pos_bias_v` and the relative-shift trick.
- **Convolution module:** pointwise (d→2d) → GLU → depthwise convK
  (kernel 9) → norm → SiLU → pointwise (d→d). Offline = **symmetric
  (non-causal)** padding `(K-1)/2`. The norm is **batch_norm** in the published
  `parakeet-*-0.6b` checkpoints (folded to an affine scale/shift from running
  mean/var at inference); streaming configs use layer_norm. The loader is
  variant-aware (`conv_norm_type` from KV) and the converter exports
  `running_mean`/`running_var` for the batch_norm case.
- **0.6B variant** dims, confirmed against `parakeet-ctc-0.6b`: d_model 1024,
  n_layers 24, n_heads 8, ff_expansion_factor 4 (d_ff 4096), conv_kernel 9,
  subsampling_conv_channels 256, xscaling true, att_context_size [-1,-1]. All
  read from GGUF KV — the loader is metadata-driven, not hardcoded.
- **Tensor layouts** (confirmed via forward hooks): conformer layers operate in
  `[B, T, D]`; the encoder returns `[B, D, T]` (channels-first) to feed the
  Conv1d-style decoders. B is always 1.

### Decoders
- **CTC:** `ConvASRDecoder` — a single 1×1 Conv1d `d_model → vocab+1`
  (blank = index `vocab`), then log_softmax. Greedy = argmax + collapse repeats
  + drop blank.
- **RNNT:** prediction network (Embedding + LSTM, typically 1 layer,
  `pred_hidden` ≈ 640) + joint network (project encoder & pred to `joint_hidden`,
  sum, ReLU, linear → vocab+1). Greedy transducer loop emits tokens, advancing
  the time index only on blank.
- **TDT:** RNNT + a **duration head**. Joint output is `vocab+1+len(durations)`,
  `durations` e.g. `[0,1,2,4,8]`. Greedy loop jumps the time index forward by the
  predicted duration, reducing joint calls.

### Tokenizer
SentencePiece BPE, vocab ≈ 1024. Inference only needs id→piece detokenization
(map `▁`→space). Vocab pieces are stored in the GGUF; no SentencePiece runtime
dependency.

## 3. Scope & phasing

The FastConformer encoder is the shared backbone, built and proven first; then
each decoder is layered on.

- **Phase 0 — Scaffolding:** Python reference environment (see §8), repo,
  CMake + ggml submodule, `audio_io`, GGUF `model_loader` skeleton, CLI
  skeleton, converter skeleton, Python baseline-dumper harness.
- **Phase 1 — Backbone (milestone):** log-mel preprocessing + FastConformer
  encoder + CTC decoder → end-to-end CTC transcription. Tensor parity through
  every stage; token parity end-to-end. Anchor: `nvidia/parakeet-ctc-0.6b`.
- **Phase 2 — RNNT:** prediction net (LSTM) + joint + greedy transducer loop.
  Token/WER parity. Anchor: `nvidia/parakeet-rnnt-0.6b`.
- **Phase 3 — TDT (north star):** duration head + duration-aware greedy loop →
  `nvidia/parakeet-tdt-0.6b-v2` works. Token/WER parity. Headline deliverable.
- **Phase 4 — Productionize:** quantization (Q8_0 / Q4_K …), flat C-API, LocalAI
  backend, HF model publishing, full CI.
- **Phase 5 — Streaming (deferred):** cache-aware chunked attention + conv /
  attention caches.

`parakeet-tdt-0.6b-v2` is TDT-only (no CTC head); that is why Phase 1's
end-to-end CTC validation anchors on a separate CTC checkpoint. Both are 0.6B
FastConformer, so the encoder code is identical.

## 4. Repository layout

```
include/
  parakeet.h            # native C++/C API
  parakeet_capi.h       # flat C-API (FFI / dlopen / LocalAI)
src/
  audio_io.{hpp,cpp}    # dr_wav load + resample to 16kHz mono
  mel.{hpp,cpp}         # STFT + mel filterbank + log + normalize (FFT outside ggml graph)
  model_loader.{hpp,cpp}# gguf mmap → name → ggml_tensor; metadata-driven config
  subsampling.{hpp,cpp} # dw_striding 8x conv front end
  relpos_attention.{hpp,cpp} # RelPositionMultiHeadAttention (pos_bias_u/v + rel shift)
  conformer.{hpp,cpp}   # ConformerLayer: FFN½ → MHSA → Conv → FFN½ → LN
  encoder.{hpp,cpp}     # FastConformer stack + xscaling + pos emb
  ctc_decoder.{hpp,cpp} # 1x1 conv → logits  (Phase 1)
  rnnt.{hpp,cpp}        # prediction net (LSTM) + joint net  (Phase 2)
  tdt.{hpp,cpp}         # duration head + duration-aware greedy  (Phase 3)
  search.{hpp,cpp}      # CTC/RNNT/TDT greedy decoding loops
  tokenizer.{hpp,cpp}   # SentencePiece BPE detokenization (vocab in gguf)
  parakeet.cpp          # public-API impl / orchestrator (thin)
  parakeet_capi.cpp     # flat C-API shim
examples/cli/           # parakeet-cli: transcribe, info, bench, quantize
scripts/                # convert_parakeet_to_gguf.py, gen_nemo_baseline.py, quantize, publish_hf
tests/                  # ctest parity + smoke; fixtures/ baselines
docs/                   # conversion.md, parity.md, architecture.md
third_party/ggml        # pinned submodule
```

## 5. Components & ggml notes

- **mel:** NeMo FilterbankFeatures values above. FFT has no good ggml op, so it
  is hand-rolled plain C++ outside the graph (whisper.cpp-style), producing an
  `[80, T]` tensor fed into ggml. The **mel filterbank basis is precomputed by
  the converter and stored in the GGUF** so C++ does not re-derive librosa.
- **subsampling:** `dw_striding` factor 8; `ggml_conv_1d_dw` / 2d (B=1 only —
  acceptable, always B=1).
- **relpos_attention:** numerically riskiest piece (the matrix rel-shift +
  pos_bias_u/v). Gets a dedicated parity test.
- **conformer block:** SiLU = swish; GLU = split + sigmoid-mul; LayerNorm for
  the FFN/MHSA pre-norms; the conv module's norm is batch_norm (inference =
  affine scale/shift) for published 0.6b, layer_norm for streaming — all native
  ggml.
- **rnnt prediction net:** ggml has no LSTM op → hand-rolled gates (1 layer,
  manageable).
- **model_loader:** metadata-driven — all dims / durations / vocab from GGUF KV;
  new variants need no C++ changes.
- **tokenizer:** id→piece detok only; vocab pieces in GGUF.

## 6. Data flow

```
WAV → audio_io(16k mono) → mel[80,T] → subsampling(÷8) → FastConformer encoder[D, T/8]
   ├─ CTC:  1x1 conv → logits → greedy collapse-blank → detok → text
   └─ RNNT/TDT: greedy transducer loop(encoder, pred-net, joint[, duration]) → tokens → detok → text
```

## 7. GGUF schema & converter

- **GGUF v3, fully metadata-driven.** KV: `parakeet.arch`
  (`fastconformer_ctc`/`_rnnt`/`_tdt`), encoder dims (d_model, n_layers, n_heads,
  ff_dim, conv_kernel, subsampling_factor, xscaling), preprocessor params,
  decoder params (pred_hidden, pred_layers, joint_hidden, durations list,
  blank_id, vocab_size), tokenizer vocab pieces.
- **Mel filterbank tensor** precomputed (librosa slaney, 80×(n_fft/2+1)) and
  stored in the GGUF — guarantees mel parity.
- **Tensor naming** mirrors NeMo state dict with a short prefix map
  (`encoder.*`→`enc.*`, `ctc_decoder.*`→`ctc.*`, prediction→`pred.*`,
  joint→`joint.*`); documented in `docs/conversion.md`. `--strict` flags
  unmapped keys.
- **Quant:** 2D weights above block threshold quantized; norms, biases,
  embeddings, `pos_bias_u/v`, and **conv kernels stay F32** (conv path casts to
  f16 — quantized conv kernels would silently corrupt). K-quants via CLI
  `quantize`.
- `scripts/convert_parakeet_to_gguf.py` loads via `nemo_toolkit[asr]` (or direct
  `.nemo` tarball parse).

## 8. Python reference environment (hard prerequisite)

The A+C parity strategy depends on running the real NeMo model to dump reference
tensors. This is a hard prerequisite for Phase 1+, set up in Phase 0.

- A project-local venv at `.venv` (managed with `uv`), CPU torch (the dev box
  has no GPU), plus `nemo_toolkit[asr]`:

  ```
  uv venv .venv --python 3.12
  uv pip install --python .venv torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venv "nemo_toolkit[asr]"
  ```

- `scripts/requirements.txt` pins the reference deps so CI / contributors can
  reproduce the baselines.
- Both `scripts/gen_nemo_baseline.py` (tensor + token dumps) and
  `scripts/convert_parakeet_to_gguf.py` run inside this venv: the converter reads
  the `.nemo` checkpoint via NeMo, the baseline dumper hooks NeMo module forwards
  to capture intermediate tensors.
- Reference checkpoints are pulled from HuggingFace on first use and cached;
  baseline fixtures committed to the repo are small (single short clip).
- **Validated 2026-05-28:** the env loads `parakeet-ctc-0.6b` on CPU, runs
  `transcribe`, and forward hooks on `preprocessor` / `encoder.layers[i]` /
  `encoder` capture intermediate tensors — confirming the baseline-dumper
  mechanism works end-to-end.

## 9. Parity & testing strategy (A+C blended)

- Per-stage **tensor parity** for the deterministic numerical path
  (preprocessing + encoder + CTC logits); **token-id / WER parity** for the
  control-flow-heavy RNNT and TDT decoders.
- `scripts/gen_nemo_baseline.py` runs the real NeMo model on a fixed test WAV and
  dumps a `baseline.gguf` bundle: mel, post-subsampling, first/mid/last + final
  encoder-layer outputs, encoder_out, CTC logits, and the reference token-id
  sequence.
- ctests, bottom-up: `test_mel` → `test_subsampling` → `test_relpos_attention` →
  `test_conformer_block` → `test_encoder` → `test_ctc` (tensor parity within
  tolerance), then `test_rnnt_tokens` / `test_tdt_tokens` (token match + WER on a
  small clip set).
- Model-dependent tests use `SKIP_RETURN_CODE=77`. Small baseline fixtures are
  committed.

## 10. C-API & build

- `parakeet_capi.h` flat surface (Phase 4): `load` / `unload`,
  `transcribe_path`, `transcribe_buffer` (f32 PCM), result getters (text +
  optional token timestamps), `free_string`. Versioned, dlopen-friendly for a
  LocalAI native backend.
- CMake options: `PARAKEET_BUILD_TESTS` / `PARAKEET_BUILD_CLI` /
  `PARAKEET_SHARED` + `PARAKEET_GGML_{CUDA,METAL,VULKAN,HIPBLAS}` passthrough.
  CPU is the supported path; GPU wired but not CI-exercised.

## 11. Definition of done per phase

- **Phase 1:** `parakeet-ctc-0.6b` transcribes a test clip correctly;
  mel/subsampling/encoder/CTC tensor parity within tolerance; `parakeet-cli
  transcribe` works; CI smoke.
- **Phase 2:** `parakeet-rnnt-0.6b` greedy token parity.
- **Phase 3:** `parakeet-tdt-0.6b-v2` greedy token parity + WER on clip set.
- **Phase 4:** quantized GGUFs published; C-API + LocalAI backend; full CI.

## 12. Out of scope (v1)

- Beam search / external LM fusion (greedy only initially).
- Cache-aware streaming (Phase 5).
- Training / fine-tuning.
- Batch size > 1 (always B=1).
- AED / multitask (Canary) models.

## 13. Reference

- NeMo (local): `/home/mudler/_git/NeMo`
  - `nemo/collections/asr/modules/audio_preprocessing.py`
  - `nemo/collections/asr/modules/conformer_encoder.py`
  - `nemo/collections/asr/parts/submodules/{conformer_modules,subsampling,multi_head_attention}.py`
  - `nemo/collections/asr/modules/conv_asr.py` (CTC)
  - `nemo/collections/asr/modules/rnnt.py` (RNNT pred + joint)
  - `nemo/collections/asr/parts/submodules/transducer_decoding/` (TDT greedy)
- Sibling ports: `/home/mudler/_git/rt-detr.cpp`, `/home/mudler/_git/vibevoice.cpp`
