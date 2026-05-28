# parakeet.cpp

parakeet.cpp is a C++17/ggml inference port of NVIDIA NeMo Parakeet ASR models,
providing fast, dependency-light automatic speech recognition on CPU (and GPU
backends via ggml) without a Python runtime at inference time.

Supports all offline Parakeet families — CTC, RNNT, TDT, and hybrid TDT-CTC
(0.6B/1.1B/110M, EN + multilingual v3) — validated at WER 0 vs NeMo on every
published checkpoint. See `docs/parity.md` for the full coverage matrix.
Cache-aware streaming + EOU is future work (Phase 5).

---

## Build

Clone with submodules (ggml is vendored at `third_party/ggml`):

```sh
git clone --recursive https://github.com/mudler/parakeet.cpp
cd parakeet.cpp
cmake -B build -DPARAKEET_BUILD_TESTS=ON && cmake --build build -j
```

Use `-DGGML_NATIVE=OFF` for portable/CI builds (disables host-specific ISA
extensions). For the shared library (LocalAI / dlopen):

```sh
cmake -B build-shared -DPARAKEET_SHARED=ON -DPARAKEET_BUILD_CLI=ON
cmake --build build-shared -j
# -> build-shared/libparakeet.so
```

### CMake options

| Option                   | Default | Purpose                                    |
| ------------------------ | ------- | ------------------------------------------ |
| `PARAKEET_BUILD_TESTS`   | OFF     | Compile and register ctest targets         |
| `PARAKEET_BUILD_CLI`     | ON      | Build `parakeet-cli`                       |
| `PARAKEET_SHARED`        | OFF     | Build libparakeet as a shared library      |
| `PARAKEET_GGML_CUDA`     | OFF     | Forward GGML_CUDA to the submodule         |
| `PARAKEET_GGML_METAL`    | OFF     | Forward GGML_METAL to the submodule        |
| `PARAKEET_GGML_VULKAN`   | OFF     | Forward GGML_VULKAN to the submodule       |
| `PARAKEET_GGML_HIPBLAS`  | OFF     | Forward GGML_HIPBLAS to the submodule      |

---

## Python environment setup

Required once, for model conversion and validation (not for inference):

```sh
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r scripts/requirements.txt   # nemo_toolkit[asr] + gguf
```

NeMo 2.7.3 is the validated version. The anchor checkpoint
`nvidia/parakeet-tdt_ctc-110m` (~440 MB) is auto-downloaded by NeMo on first use.

---

## Converting a model

Convert a HuggingFace or local `.nemo` checkpoint to GGUF:

```sh
# Default (F32) — lossless, largest file
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/parakeet-tdt_ctc-110m \
    --output m.gguf

# F16 — ~0.58x size, WER 0 vs NeMo
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/parakeet-tdt_ctc-110m --dtype f16 --output m.gguf

# Q8_0 — ~0.39x size, WER 0 vs NeMo
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/parakeet-tdt_ctc-110m --dtype q8_0 --output m.gguf
```

Supported `--dtype`: `f32` (default), `f16`, `q8_0`.

---

## Quantization

For K-quants (`q4_k`, `q5_k`, `q6_k`) — which the Python `gguf` writer does
not support — re-quantize an existing F32 GGUF with the CLI:

```sh
parakeet-cli quantize <in.gguf> <out.gguf> <type>
# e.g.
parakeet-cli quantize m.gguf m_q4k.gguf q4_k
parakeet-cli quantize m.gguf m_q6k.gguf q6_k
```

Supported types: `q4_0`, `q5_0`, `q8_0`, `q4_k`, `q5_k`, `q6_k`.

Only the large linear `ggml_mul_mat`-consumed weights (encoder FFN, attention
projections, joint enc/pred projections, subsampling output projection) are
quantized; conv/LSTM/featurizer/batch_norm/bias tensors stay F32. See
`docs/quantization.md` for the full policy, allowlist, and measured size + WER
per type.

---

## Running inference

```sh
# Default decoder (TDT for hybrid/TDT models, CTC for standalone CTC)
parakeet-cli transcribe --model m.gguf --input audio.wav

# Force a decoder
parakeet-cli transcribe --model m.gguf --input audio.wav --decoder ctc
parakeet-cli transcribe --model m.gguf --input audio.wav --decoder tdt

# Print model metadata (arch, dims, mel params, vocab size, TDT durations)
parakeet-cli info m.gguf
```

The `parakeet-cli` binary is at `build/examples/cli/parakeet-cli`.

---

## C-API (`libparakeet.so`)

`include/parakeet_capi.h` defines a flat, exception-free C-API designed for
`dlopen` / FFI / LocalAI integration. Build the shared library with
`-DPARAKEET_SHARED=ON`:

```c
#include "parakeet_capi.h"

parakeet_ctx *ctx = parakeet_capi_load("model.gguf");  // load ONCE
if (!ctx) { fprintf(stderr, "%s\n", parakeet_capi_last_error(ctx)); return 1; }

char *text = parakeet_capi_transcribe_path(ctx, "audio.wav", 0 /*default*/);
if (text) { printf("%s\n", text); parakeet_capi_free_string(text); }

parakeet_capi_free(ctx);
```

In-memory PCM:
```c
char *text = parakeet_capi_transcribe_pcm(ctx, samples, n_samples,
                                          sample_rate, 0 /*default*/);
```

The LocalAI backend (in the LocalAI repo) dlopens `libparakeet.so` and uses
these symbols directly. See `include/parakeet_capi.h` for the full API.

---

## Model coverage

See `docs/parity.md` for the full coverage matrix. Summary:

| Family | Representative checkpoints | Heads | WER vs NeMo |
| --- | --- | --- | --- |
| Hybrid TDT+CTC | `parakeet-tdt_ctc-110m`, `parakeet-tdt_ctc-1.1b` | TDT + CTC | 0.0 |
| TDT (hybrid) | `parakeet-tdt-0.6b-v2`, `parakeet-tdt-0.6b-v3` (multilingual) | TDT | 0.0 |
| Pure TDT | `parakeet-tdt-1.1b` | TDT | 0.0 |
| CTC | `parakeet-ctc-0.6b`, `parakeet-ctc-1.1b` | CTC | 0.0 |
| RNNT | `parakeet-rnnt-0.6b`, `parakeet-rnnt-1.1b` | RNNT | 0.0 |

All 10 published offline checkpoints validated at WER 0 vs NeMo 2.7.3.
Sizes: 110M (512/17 layers), 0.6B (1024/24), 1.1B (1024/42).
Cache-aware streaming + EOU (`parakeet_realtime_eou_120m-v1`) is Phase 5
(future work).

---

## Running tests

Model-independent (run anywhere):

```sh
ctest --test-dir build --output-on-failure -LE model
```

Model-dependent (need venv + checkpoint):

```sh
export PARAKEET_TEST_GGUF=/tmp/pk110m.gguf
export PARAKEET_TEST_BASELINE=/tmp/baseline.gguf
export PARAKEET_TEST_BASELINE_SPEECH=/tmp/baseline_speech.gguf
ctest --test-dir build --output-on-failure
```

Tests labelled `model` return exit code 77 (ctest SKIP) when their required
env vars are absent, so they never break a CI environment without a model.
