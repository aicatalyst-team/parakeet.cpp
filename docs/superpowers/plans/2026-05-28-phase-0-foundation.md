# parakeet.cpp Phase 0 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the buildable project skeleton, the GGUF converter, the C++ model loader, audio I/O, and the NeMo baseline dumper — everything Phase 1 inference needs, with nothing that does ASR math yet.

**Architecture:** C++17 + a pinned stock ggml submodule. A Python converter turns a NeMo `.nemo` Parakeet checkpoint into a metadata-driven GGUF (tensor names kept **verbatim** from the NeMo `state_dict` for a 1:1 port). A C++ `ModelLoader` mmaps that GGUF and exposes config + `name → ggml_tensor`. A second Python script dumps NeMo intermediate tensors into a `baseline.gguf` for the Phase 1 parity tests. CPU is the supported path.

**Tech Stack:** C++17, CMake, ggml (submodule), dr_wav (vendored single header), Python 3.12 venv with `nemo_toolkit[asr]` + `gguf`, ctest.

**Reference (read before starting):**
- Spec: `docs/superpowers/specs/2026-05-28-parakeet-cpp-design.md` (§2.1 validated config, §7 GGUF schema)
- Sibling ports for conventions: `/home/mudler/_git/rt-detr.cpp`, `/home/mudler/_git/vibevoice.cpp`
- NeMo source: `/home/mudler/_git/NeMo`
- Python env already exists at `.venv` (CPU torch + NeMo 2.7.3). The anchor checkpoint `nvidia/parakeet-tdt_ctc-110m` is already cached in `~/.cache/huggingface`.

**Conventions:**
- C++ test executables return `0` = pass, `77` = skip (model/file absent), other = fail. No gtest.
- Commit after every green step. Use `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- Don't skip hooks. CPU-only; GPU CMake flags are wired but not exercised.

---

## Orientation — run BEFORE Task 1 (you are a fresh sub-agent with zero context)

This plan is self-contained. Do not assume any memory of prior conversations.
Before starting, establish the ground truth of the working tree:

1. **Read the spec** `docs/superpowers/specs/2026-05-28-parakeet-cpp-design.md`
   in full — especially §2.1 (validated config + arch detection), §7 (GGUF
   schema), §8 (Python env). It is the source of truth; this plan implements it.

2. **Verify the environment** earlier setup already established (these are
   prerequisites, NOT tasks in this plan):
   - `git rev-parse --is-inside-work-tree` → `true` (the spec + plans are already committed).
   - `.venv/bin/python -c "import nemo, gguf, torch; print(nemo.__version__)"` → `2.7.3`.
     If this errors, the Python reference env is missing — recreate it per spec §8 before proceeding.
   - `/home/mudler/_git/NeMo` exists (reference source for tracing).
   - Anchor checkpoint: `ls ~/.cache/huggingface/hub/ | grep parakeet-tdt_ctc-110m`.
     If absent it auto-downloads (~440 MB) on first converter/baseline run — fine.

3. **Find your resume point** (this plan may be partially complete). Run
   `git log --oneline -20`. Each task ends in a commit with a recognizable
   message (e.g. `feat: audio_io …`, `feat: GGUF model loader …`). Identify the
   highest completed task, then **start at the first incomplete task**. Re-run
   that task's verification step first to confirm the tree is green before
   continuing. Do not redo committed work.

4. **No predecessor phase.** Phase 0 is the first plan, so there is no prior
   phase done-when checklist to validate — the "previous state" is simply the
   committed spec/plans + the working venv verified above. (Every later phase
   plan begins by re-running the *previous* phase's "done-when" checklist as its
   entry gate; see "Handoff" at the end.)

---

## File structure created in this phase

```
.gitignore, .gitmodules, LICENSE, README.md, AGENTS.md
CMakeLists.txt
third_party/ggml/                 # submodule
third_party/dr_wav.h              # vendored single header
include/
  parakeet.h                      # native C/C++ API (version + load/free stubs)
src/
  parakeet.cpp                    # version() + public-API shims
  common.hpp / common.cpp         # logging/trace helpers
  audio_io.hpp / audio_io.cpp     # dr_wav load + linear resample to 16k mono
  model_loader.hpp / model_loader.cpp   # GGUF → ParakeetConfig + name→tensor
examples/cli/
  main.cpp                        # parakeet-cli: `info` subcommand
  CMakeLists.txt
scripts/
  requirements.txt
  convert_parakeet_to_gguf.py     # .nemo → GGUF
  gen_nemo_baseline.py            # NeMo intermediates → baseline.gguf
tests/
  CMakeLists.txt
  test_smoke.cpp
  test_audio_io.cpp
  test_model_loader.cpp
  python/
    check_convert.py
    check_baseline.py
docs/
  conversion.md                   # GGUF schema reference
.github/workflows/ci.yml
```

---

## Task 1: Repo skeleton + ggml submodule

**Files:**
- Create: `.gitignore`, `.gitmodules`, `LICENSE`, `README.md`, `third_party/ggml` (submodule)

- [ ] **Step 1: Create `.gitignore`**

```gitignore
/build*/
/.venv/
/models/
*.gguf
__pycache__/
*.pyc
/.cache/
```

- [ ] **Step 2: Add the ggml submodule**

Run:
```bash
git submodule add https://github.com/ggml-org/ggml third_party/ggml
git -C third_party/ggml checkout master && git -C third_party/ggml rev-parse --short HEAD
```
Expected: a `.gitmodules` file is created and `third_party/ggml` is populated. Note the SHA in the commit message so the pin is recorded.

- [ ] **Step 3: Minimal `LICENSE` + `README.md`**

`LICENSE`: MIT, copyright 2026 the parakeet.cpp authors.
`README.md`: one paragraph — "C++/ggml inference port of NVIDIA NeMo Parakeet ASR. See `docs/superpowers/specs/` for the design." plus a Build section placeholder.

- [ ] **Step 4: Commit**

```bash
git add .gitignore .gitmodules LICENSE README.md third_party/ggml
git commit -m "chore: repo skeleton + pinned ggml submodule (<sha>)"
```

---

## Task 2: CMake build + parakeet static lib + smoke test

**Files:**
- Create: `CMakeLists.txt`, `include/parakeet.h`, `src/parakeet.cpp`, `src/common.hpp`, `src/common.cpp`, `tests/CMakeLists.txt`, `tests/test_smoke.cpp`

- [ ] **Step 1: Write the failing smoke test**

`tests/test_smoke.cpp`:
```cpp
#include "parakeet.h"
#include <cstdio>
#include <cstring>

int main() {
    const char* v = parakeet_version();
    if (v == nullptr || std::strlen(v) == 0) {
        std::fprintf(stderr, "version string is empty\n");
        return 1;
    }
    std::printf("parakeet.cpp version: %s\n", v);
    return 0;
}
```

- [ ] **Step 2: Public header**

`include/parakeet.h`:
```cpp
#ifndef PARAKEET_H
#define PARAKEET_H
#ifdef __cplusplus
extern "C" {
#endif
// Returns a static version string. Never null.
const char* parakeet_version(void);
#ifdef __cplusplus
}
#endif
#endif // PARAKEET_H
```

- [ ] **Step 3: Implementation + common helpers**

`src/parakeet.cpp`:
```cpp
#include "parakeet.h"
#define PARAKEET_VERSION "0.0.1"
extern "C" const char* parakeet_version(void) { return PARAKEET_VERSION; }
```

`src/common.hpp`:
```cpp
#pragma once
#include <cstdio>
#define PK_LOG(...)  do { std::fprintf(stderr, "[parakeet] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
```
`src/common.cpp`: `#include "common.hpp"` (placeholder TU so the file exists for future helpers).

- [ ] **Step 4: Root `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.18)
project(parakeet_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PARAKEET_BUILD_TESTS "Build ctest targets" OFF)
option(PARAKEET_BUILD_CLI   "Build parakeet-cli" ON)
option(PARAKEET_SHARED      "Build libparakeet as shared" OFF)
option(PARAKEET_GGML_CUDA   "Forward GGML_CUDA" OFF)
option(PARAKEET_GGML_METAL  "Forward GGML_METAL" OFF)
option(PARAKEET_GGML_VULKAN "Forward GGML_VULKAN" OFF)
option(PARAKEET_GGML_HIPBLAS "Forward GGML_HIPBLAS" OFF)

set(GGML_CUDA    ${PARAKEET_GGML_CUDA}   CACHE BOOL "" FORCE)
set(GGML_METAL   ${PARAKEET_GGML_METAL}  CACHE BOOL "" FORCE)
set(GGML_VULKAN  ${PARAKEET_GGML_VULKAN} CACHE BOOL "" FORCE)
set(GGML_HIPBLAS ${PARAKEET_GGML_HIPBLAS} CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml)

set(PARAKEET_SRC
    src/parakeet.cpp
    src/common.cpp)

if(PARAKEET_SHARED)
  add_library(parakeet SHARED ${PARAKEET_SRC})
else()
  add_library(parakeet STATIC ${PARAKEET_SRC})
endif()
target_include_directories(parakeet PUBLIC include PRIVATE src)
target_link_libraries(parakeet PUBLIC ggml)

if(PARAKEET_BUILD_CLI)
  add_subdirectory(examples/cli)
endif()
if(PARAKEET_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
```

`tests/CMakeLists.txt`:
```cmake
function(pk_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE parakeet)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

pk_add_test(test_smoke)
```

For `examples/cli` to not break the build before Task 6, create a stub `examples/cli/CMakeLists.txt` now:
```cmake
add_executable(parakeet-cli main.cpp)
target_link_libraries(parakeet-cli PRIVATE parakeet)
target_include_directories(parakeet-cli PRIVATE ${CMAKE_SOURCE_DIR}/src)
```
and a stub `examples/cli/main.cpp`:
```cpp
#include "parakeet.h"
#include <cstdio>
int main() { std::printf("parakeet-cli %s\n", parakeet_version()); return 0; }
```

- [ ] **Step 5: Configure, build, run the test**

Run:
```bash
cmake -B build -DPARAKEET_BUILD_TESTS=ON -DGGML_NATIVE=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```
Expected: build succeeds; `test_smoke` PASSES and prints the version.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt include src tests examples
git commit -m "feat: cmake build, libparakeet, version smoke test"
```

---

## Task 3: Vendor dr_wav + audio I/O (load + resample to 16k mono)

**Files:**
- Create: `third_party/dr_wav.h`, `src/audio_io.hpp`, `src/audio_io.cpp`, `tests/test_audio_io.cpp`
- Modify: `CMakeLists.txt` (add `src/audio_io.cpp`), `tests/CMakeLists.txt` (add test)

- [ ] **Step 1: Vendor dr_wav**

Run:
```bash
curl -L -o third_party/dr_wav.h https://raw.githubusercontent.com/mackron/dr_libs/master/dr_wav.h
```
Expected: `third_party/dr_wav.h` exists (single-header, ~6k lines).

- [ ] **Step 2: Write the failing test**

`tests/test_audio_io.cpp`:
```cpp
#include "audio_io.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdint>

// dr_wav writer is only needed in the test; pull it in here.
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

static void write_sine(const char* path, int sr, int n, float freq) {
    std::vector<float> s(n);
    for (int i = 0; i < n; ++i) s[i] = 0.25f * std::sin(2.0*M_PI*freq*i/sr);
    drwav_data_format fmt{};
    fmt.container = drwav_container_riff;
    fmt.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels = 1; fmt.sampleRate = (drwav_uint32)sr; fmt.bitsPerSample = 32;
    drwav w; drwav_init_file_write(&w, path, &fmt, nullptr);
    drwav_write_pcm_frames(&w, n, s.data());
    drwav_uninit(&w);
}

int main() {
    const char* path = "/tmp/pk_test_44k.wav";
    write_sine(path, 44100, 44100, 440.0f); // 1s @ 44.1k

    pk::Audio a;
    if (!pk::load_audio_16k_mono(path, a)) { std::fprintf(stderr, "load failed\n"); return 1; }
    if (a.sample_rate != 16000) { std::fprintf(stderr, "sr=%d\n", a.sample_rate); return 1; }
    // ~1s of audio resampled to 16k → ~16000 samples (allow small edge slack)
    if (a.samples.size() < 15800 || a.samples.size() > 16200) {
        std::fprintf(stderr, "n=%zu\n", a.samples.size()); return 1;
    }
    // not silent
    double e = 0; for (float v : a.samples) e += (double)v*v;
    if (e < 1.0) { std::fprintf(stderr, "energy too low %f\n", e); return 1; }
    std::printf("audio_io ok: %zu samples @ %d Hz\n", a.samples.size(), a.sample_rate);
    return 0;
}
```

- [ ] **Step 3: Interface**

`src/audio_io.hpp`:
```cpp
#pragma once
#include <string>
#include <vector>
#include <cstdint>
namespace pk {
struct Audio {
    std::vector<float> samples; // mono, [-1,1]
    int sample_rate = 0;
};
// Loads any WAV, downmixes to mono, linearly resamples to 16 kHz. Returns false on error.
bool load_audio_16k_mono(const std::string& path, Audio& out);
// Resample mono float PCM from in_sr to out_sr (linear). Exposed for reuse/testing.
std::vector<float> resample_linear(const std::vector<float>& in, int in_sr, int out_sr);
}
```

- [ ] **Step 4: Implementation**

`src/audio_io.cpp`:
```cpp
#include "audio_io.hpp"
#include "common.hpp"
#include "dr_wav.h"   // declarations only; DR_WAV_IMPLEMENTATION lives in the test / one TU
#include <cmath>

namespace pk {

std::vector<float> resample_linear(const std::vector<float>& in, int in_sr, int out_sr) {
    if (in_sr == out_sr || in.empty()) return in;
    const double ratio = (double)out_sr / (double)in_sr;
    const size_t n_out = (size_t)std::floor(in.size() * ratio);
    std::vector<float> out(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        const double src = i / ratio;
        const size_t i0 = (size_t)src;
        const double frac = src - i0;
        const float a = in[i0];
        const float b = (i0 + 1 < in.size()) ? in[i0 + 1] : a;
        out[i] = (float)(a + (b - a) * frac);
    }
    return out;
}

bool load_audio_16k_mono(const std::string& path, Audio& out) {
    unsigned int ch = 0, sr = 0; drwav_uint64 frames = 0;
    float* pcm = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &ch, &sr, &frames, nullptr);
    if (!pcm) { PK_LOG("failed to open wav: %s", path.c_str()); return false; }
    std::vector<float> mono(frames);
    for (drwav_uint64 i = 0; i < frames; ++i) {
        double acc = 0; for (unsigned int c = 0; c < ch; ++c) acc += pcm[i*ch + c];
        mono[i] = (float)(acc / (ch ? ch : 1));
    }
    drwav_free(pcm, nullptr);
    out.samples = resample_linear(mono, (int)sr, 16000);
    out.sample_rate = 16000;
    return true;
}

} // namespace pk
```
Note: `DR_WAV_IMPLEMENTATION` is defined exactly once. In Task 6 the CLI will define it; for the library TU we only need declarations. To guarantee one definition in the library, add it at the top of `audio_io.cpp` instead: change the include to:
```cpp
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
```
and **remove** the `#define DR_WAV_IMPLEMENTATION` from the test (the test links `parakeet`, which now provides the symbols). Update the test's include to plain `#include "dr_wav.h"`.

- [ ] **Step 5: Wire CMake**

In root `CMakeLists.txt` add `src/audio_io.cpp` to `PARAKEET_SRC`, and:
```cmake
target_include_directories(parakeet PUBLIC include PRIVATE src ${CMAKE_SOURCE_DIR}/third_party)
```
In `tests/CMakeLists.txt` add `pk_add_test(test_audio_io)` and ensure the test target also sees `third_party`:
```cmake
target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)
```

- [ ] **Step 6: Build + run**

Run:
```bash
cmake --build build -j && ctest --test-dir build -R test_audio_io --output-on-failure
```
Expected: PASS, prints `audio_io ok: ~16000 samples @ 16000 Hz`.

- [ ] **Step 7: Commit**

```bash
git add third_party/dr_wav.h src/audio_io.* CMakeLists.txt tests/CMakeLists.txt tests/test_audio_io.cpp
git commit -m "feat: audio_io — wav load + linear resample to 16k mono"
```

---

## Task 4: Python converter — `.nemo` → GGUF

**Files:**
- Create: `scripts/requirements.txt`, `scripts/convert_parakeet_to_gguf.py`, `tests/python/check_convert.py`, `docs/conversion.md`

The GGUF uses **verbatim NeMo `state_dict` tensor names** (no remap) plus the two featurizer buffers. All config lives in KV. This is the canonical schema; `docs/conversion.md` documents it.

- [ ] **Step 1: `scripts/requirements.txt`**

```
nemo_toolkit[asr]
gguf
```
(The `.venv` already has these; this file pins them for CI/contributors.)

- [ ] **Step 2: Write the converter**

`scripts/convert_parakeet_to_gguf.py`:
```python
#!/usr/bin/env python3
"""Convert a NeMo Parakeet checkpoint to GGUF (f32). Tensor names verbatim."""
import argparse, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import gguf
from nemo.collections.asr.models import ASRModel

def _get(cfg, key, default=None):
    try: return cfg[key]
    except Exception: return getattr(cfg, key, default)

def detect_arch(m):
    cfg = m.cfg
    if _get(cfg, "aux_ctc") is not None:
        loss = _get(_get(cfg, "loss", {}), "loss_name", "")
        durs = _get(_get(cfg, "decoding", {}), "durations")
        return "hybrid_tdt_ctc" if (loss == "tdt" or durs) else "hybrid_rnnt_ctc"
    if _get(cfg, "joint") is not None:
        durs = _get(_get(cfg, "decoding", {}), "durations")
        nxo = _get(_get(cfg, "joint", {}), "num_extra_outputs", 0)
        return "tdt" if (durs or (nxo and nxo > 0)) else "rnnt"
    return "ctc"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local .nemo")
    ap.add_argument("--output", required=True)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    m = ASRModel.from_pretrained(args.model, map_location="cpu") if "/" in args.model and not args.model.endswith(".nemo") \
        else ASRModel.restore_from(args.model, map_location="cpu")
    m.eval()
    arch = detect_arch(m)
    cfg = m.cfg
    enc = cfg.encoder
    feat = m.preprocessor.featurizer  # effective runtime values live here

    w = gguf.GGUFWriter(args.output, "parakeet")
    w.add_string("general.name", args.model)
    w.add_string("parakeet.arch", arch)

    # encoder
    w.add_uint32("parakeet.encoder.feat_in", int(_get(enc, "feat_in")))
    w.add_uint32("parakeet.encoder.d_model", int(_get(enc, "d_model")))
    w.add_uint32("parakeet.encoder.n_layers", int(_get(enc, "n_layers")))
    w.add_uint32("parakeet.encoder.n_heads", int(_get(enc, "n_heads")))
    ffx = int(_get(enc, "ff_expansion_factor", 4))
    w.add_uint32("parakeet.encoder.ff_dim", int(_get(enc, "d_model")) * ffx)
    w.add_uint32("parakeet.encoder.conv_kernel", int(_get(enc, "conv_kernel_size")))
    w.add_string("parakeet.encoder.conv_norm_type", str(_get(enc, "conv_norm_type", "batch_norm")))
    w.add_uint32("parakeet.encoder.subsampling_factor", int(_get(enc, "subsampling_factor")))
    w.add_uint32("parakeet.encoder.subsampling_conv_channels", int(_get(enc, "subsampling_conv_channels")))
    w.add_bool("parakeet.encoder.xscaling", bool(_get(enc, "xscaling", True)))
    w.add_uint32("parakeet.encoder.pos_emb_max_len", int(_get(enc, "pos_emb_max_len", 5000)))

    # preprocessor (effective values off the featurizer object)
    w.add_uint32("parakeet.preprocessor.sample_rate", int(getattr(feat, "sample_rate", 16000)))
    w.add_uint32("parakeet.preprocessor.n_mels", int(getattr(feat, "nfilt")))
    w.add_uint32("parakeet.preprocessor.n_fft", int(getattr(feat, "n_fft")))
    w.add_uint32("parakeet.preprocessor.win_length", int(getattr(feat, "win_length")))
    w.add_uint32("parakeet.preprocessor.hop_length", int(getattr(feat, "hop_length")))
    pre = getattr(feat, "preemph", None)
    w.add_float32("parakeet.preprocessor.preemph", float(pre) if pre is not None else 0.0)
    w.add_float32("parakeet.preprocessor.mag_power", float(getattr(feat, "mag_power", 2.0)))
    w.add_string("parakeet.preprocessor.normalize", str(getattr(feat, "normalize", "per_feature")))
    w.add_float32("parakeet.preprocessor.log_zero_guard", float(getattr(feat, "log_zero_guard_value", 2**-24)) if isinstance(getattr(feat,"log_zero_guard_value",None),(int,float)) else 2**-24)

    # vocab / tokenizer
    vocab = int(m.tokenizer.vocab_size)
    w.add_uint32("parakeet.vocab_size", vocab)
    w.add_uint32("parakeet.blank_id", vocab)  # blank always == vocab_size
    pieces = [m.tokenizer.ids_to_tokens([i])[0] for i in range(vocab)]
    w.add_array("parakeet.tokenizer.pieces", pieces)

    # transducer config
    if arch in ("rnnt", "tdt", "hybrid_rnnt_ctc", "hybrid_tdt_ctc"):
        prednet = _get(cfg.decoder, "prednet", {})
        w.add_uint32("parakeet.decoder.pred_hidden", int(_get(prednet, "pred_hidden")))
        w.add_uint32("parakeet.decoder.pred_rnn_layers", int(_get(prednet, "pred_rnn_layers", 1)))
        jn = _get(cfg.joint, "jointnet", {})
        w.add_uint32("parakeet.joint.joint_hidden", int(_get(jn, "joint_hidden")))
        w.add_string("parakeet.joint.activation", str(_get(jn, "activation", "relu")))
    if arch in ("tdt", "hybrid_tdt_ctc"):
        durs = list(_get(_get(cfg, "decoding", {}), "durations") or
                    _get(_get(cfg, "model_defaults", {}), "tdt_durations"))
        w.add_array("parakeet.tdt.durations", [int(d) for d in durs])

    # tensors: verbatim names, f32. Include featurizer buffers explicitly.
    sd = m.state_dict()
    written = 0
    keep_buffers = {"preprocessor.featurizer.fb", "preprocessor.featurizer.window"}
    for name, t in sd.items():
        if name.startswith("preprocessor.") and name not in keep_buffers:
            continue  # skip preprocessor internals except fb/window
        if not hasattr(t, "detach"):
            continue
        arr = t.detach().cpu().float().numpy()
        if arr.ndim == 0:
            continue
        w.add_tensor(name, np.ascontiguousarray(arr))
        written += 1

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {args.output}: arch={arch} vocab={vocab} tensors={written}")

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Write the converter check (acts as the test)**

`tests/python/check_convert.py`:
```python
#!/usr/bin/env python3
import os, subprocess, sys, tempfile
import gguf

MODEL = os.environ.get("PARAKEET_TEST_MODEL", "nvidia/parakeet-tdt_ctc-110m")
out = os.path.join(tempfile.gettempdir(), "pk_check.gguf")

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
conv = os.path.join(root, "scripts", "convert_parakeet_to_gguf.py")
r = subprocess.run([sys.executable, conv, "--model", MODEL, "--output", out], capture_output=True, text=True)
print(r.stdout); print(r.stderr, file=sys.stderr)
if r.returncode != 0:
    print("converter failed"); sys.exit(1)

reader = gguf.GGUFReader(out)
kv = {f.name: f for f in reader.fields.values()}
def kvval(name):
    f = kv[name]; return f.parts[f.data[0]]

assert "parakeet.arch" in kv, "missing parakeet.arch"
names = {t.name for t in reader.tensors}
# encoder + heads present (verbatim NeMo names)
assert any(n.startswith("encoder.layers.0.") for n in names), "no encoder layer 0 tensors"
assert "preprocessor.featurizer.fb" in names, "mel filterbank not exported"
assert any(n.startswith("encoder.pre_encode") for n in names), "no subsampling tensors"
# hybrid anchor must carry both heads
assert any(n.startswith("ctc_decoder.") for n in names), "no ctc head"
assert any(n.startswith("joint.") for n in names), "no joint"
print("check_convert OK:", len(names), "tensors")
sys.exit(0)
```

- [ ] **Step 4: Run the check (needs the cached model)**

Run:
```bash
.venv/bin/python tests/python/check_convert.py
```
Expected: prints `wrote …: arch=hybrid_tdt_ctc vocab=1024 tensors=…` then `check_convert OK: N tensors`. Exit 0.
If it fails on a KV/array API mismatch, inspect with `.venv/bin/python -c "import gguf; help(gguf.GGUFWriter.add_array)"` and adjust (the `gguf` package API is the source of truth).

- [ ] **Step 5: `docs/conversion.md`**

Document the GGUF schema: `general.architecture="parakeet"`, every `parakeet.*` KV key with type and meaning, the verbatim-tensor-name rule, the two featurizer buffers, and the arch-detection table (copy from spec §2.1). Include the validated `parakeet-tdt_ctc-110m` values as a worked example.

- [ ] **Step 6: Register as a ctest (skips when model absent)**

Append to `tests/CMakeLists.txt`:
```cmake
find_program(PARAKEET_PY NAMES python3 HINTS ${CMAKE_SOURCE_DIR}/.venv/bin)
if(PARAKEET_PY)
  add_test(NAME check_convert COMMAND ${PARAKEET_PY} ${CMAKE_SOURCE_DIR}/tests/python/check_convert.py)
  set_tests_properties(check_convert PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")
endif()
```
(The script should `sys.exit(77)` if `from nemo...` import fails — wrap the import and return 77 so CI without the venv skips cleanly.)

- [ ] **Step 7: Commit**

```bash
git add scripts/requirements.txt scripts/convert_parakeet_to_gguf.py tests/python/check_convert.py tests/CMakeLists.txt docs/conversion.md
git commit -m "feat: NeMo->GGUF converter + schema docs + conversion check"
```

---

## Task 5: C++ model loader — config + tensors from GGUF

**Files:**
- Create: `src/model_loader.hpp`, `src/model_loader.cpp`, `tests/test_model_loader.cpp`
- Modify: `CMakeLists.txt`, `tests/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

`tests/test_model_loader.cpp`:
```cpp
#include "model_loader.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

int main() {
    const char* env = std::getenv("PARAKEET_TEST_GGUF");
    if (!env) { std::fprintf(stderr, "PARAKEET_TEST_GGUF not set; skipping\n"); return 77; }
    pk::ModelLoader ml;
    if (!ml.load(env)) { std::fprintf(stderr, "load failed\n"); return 1; }
    const pk::ParakeetConfig& c = ml.config();
    if (c.arch.empty())   { std::fprintf(stderr, "empty arch\n"); return 1; }
    if (c.d_model == 0 || c.n_layers == 0 || c.n_heads == 0) { std::fprintf(stderr, "bad encoder dims\n"); return 1; }
    if (c.vocab_size == 0) { std::fprintf(stderr, "bad vocab\n"); return 1; }
    if (c.blank_id != c.vocab_size) { std::fprintf(stderr, "blank!=vocab\n"); return 1; }
    // mel filterbank tensor must be present
    if (ml.tensor("preprocessor.featurizer.fb") == nullptr) { std::fprintf(stderr, "no fb\n"); return 1; }
    // first conformer layer norm must be present (verbatim name)
    if (ml.tensor("encoder.layers.0.norm_feed_forward1.weight") == nullptr) {
        std::fprintf(stderr, "no layer0 norm\n"); return 1;
    }
    std::printf("loader ok: arch=%s d_model=%u layers=%u heads=%u vocab=%u\n",
                c.arch.c_str(), c.d_model, c.n_layers, c.n_heads, c.vocab_size);
    return 0;
}
```

- [ ] **Step 2: Interface**

`src/model_loader.hpp`:
```cpp
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
struct ggml_tensor;
struct ggml_context;
struct gguf_context;
namespace pk {
struct ParakeetConfig {
    std::string arch;
    // encoder
    uint32_t feat_in=0, d_model=0, n_layers=0, n_heads=0, ff_dim=0, conv_kernel=0;
    std::string conv_norm_type;
    uint32_t subsampling_factor=0, subsampling_conv_channels=0, pos_emb_max_len=5000;
    bool xscaling=true;
    // preprocessor
    uint32_t sample_rate=16000, n_mels=0, n_fft=0, win_length=0, hop_length=0;
    float preemph=0.0f, mag_power=2.0f, log_zero_guard=0.0f;
    std::string normalize;
    // transducer (optional)
    uint32_t pred_hidden=0, pred_rnn_layers=0, joint_hidden=0;
    std::string joint_activation;
    std::vector<int32_t> tdt_durations;
    // vocab
    uint32_t vocab_size=0, blank_id=0;
    std::vector<std::string> tokenizer_pieces;
};
class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader();
    bool load(const std::string& path);
    const ParakeetConfig& config() const { return cfg_; }
    ggml_tensor* tensor(const std::string& name) const; // nullptr if absent
    ggml_context* ggml_ctx() const { return ctx_; }
private:
    ParakeetConfig cfg_;
    gguf_context* gguf_ = nullptr;
    ggml_context* ctx_ = nullptr;
    std::unordered_map<std::string, ggml_tensor*> tensors_;
};
}
```

- [ ] **Step 3: Implementation**

`src/model_loader.cpp` — use `gguf_init_from_file` with `no_alloc=false` so ggml owns the tensor data; read KV via `gguf_find_key` + typed getters; populate the map by iterating `gguf_get_n_tensors` + `ggml_get_tensor`. Helper readers:
```cpp
#include "model_loader.hpp"
#include "common.hpp"
#include "ggml.h"
#include "gguf.h"
#include <cstring>
namespace pk {
static uint32_t kv_u32(gguf_context* g, const char* k, uint32_t d=0){
    int id = gguf_find_key(g,k); return id<0 ? d : (uint32_t)gguf_get_val_u32(g,id);
}
static float kv_f32(gguf_context* g, const char* k, float d=0){
    int id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_f32(g,id);
}
static bool kv_bool(gguf_context* g, const char* k, bool d=false){
    int id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_bool(g,id);
}
static std::string kv_str(gguf_context* g, const char* k, const char* d=""){
    int id = gguf_find_key(g,k); return id<0 ? std::string(d) : std::string(gguf_get_val_str(g,id));
}
ModelLoader::~ModelLoader(){ if(gguf_) gguf_free(gguf_); if(ctx_) ggml_free(ctx_); }
bool ModelLoader::load(const std::string& path){
    struct gguf_init_params p{ /*no_alloc*/false, /*ctx*/&ctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if(!gguf_){ PK_LOG("gguf open failed: %s", path.c_str()); return false; }
    cfg_.arch        = kv_str(gguf_, "parakeet.arch");
    cfg_.feat_in     = kv_u32(gguf_, "parakeet.encoder.feat_in");
    cfg_.d_model     = kv_u32(gguf_, "parakeet.encoder.d_model");
    cfg_.n_layers    = kv_u32(gguf_, "parakeet.encoder.n_layers");
    cfg_.n_heads     = kv_u32(gguf_, "parakeet.encoder.n_heads");
    cfg_.ff_dim      = kv_u32(gguf_, "parakeet.encoder.ff_dim");
    cfg_.conv_kernel = kv_u32(gguf_, "parakeet.encoder.conv_kernel");
    cfg_.conv_norm_type = kv_str(gguf_, "parakeet.encoder.conv_norm_type", "batch_norm");
    cfg_.subsampling_factor = kv_u32(gguf_, "parakeet.encoder.subsampling_factor");
    cfg_.subsampling_conv_channels = kv_u32(gguf_, "parakeet.encoder.subsampling_conv_channels");
    cfg_.xscaling    = kv_bool(gguf_, "parakeet.encoder.xscaling", true);
    cfg_.pos_emb_max_len = kv_u32(gguf_, "parakeet.encoder.pos_emb_max_len", 5000);
    cfg_.sample_rate = kv_u32(gguf_, "parakeet.preprocessor.sample_rate", 16000);
    cfg_.n_mels      = kv_u32(gguf_, "parakeet.preprocessor.n_mels");
    cfg_.n_fft       = kv_u32(gguf_, "parakeet.preprocessor.n_fft");
    cfg_.win_length  = kv_u32(gguf_, "parakeet.preprocessor.win_length");
    cfg_.hop_length  = kv_u32(gguf_, "parakeet.preprocessor.hop_length");
    cfg_.preemph     = kv_f32(gguf_, "parakeet.preprocessor.preemph", 0.0f);
    cfg_.mag_power   = kv_f32(gguf_, "parakeet.preprocessor.mag_power", 2.0f);
    cfg_.normalize   = kv_str(gguf_, "parakeet.preprocessor.normalize", "per_feature");
    cfg_.log_zero_guard = kv_f32(gguf_, "parakeet.preprocessor.log_zero_guard", 0.0f);
    cfg_.pred_hidden = kv_u32(gguf_, "parakeet.decoder.pred_hidden");
    cfg_.pred_rnn_layers = kv_u32(gguf_, "parakeet.decoder.pred_rnn_layers");
    cfg_.joint_hidden = kv_u32(gguf_, "parakeet.joint.joint_hidden");
    cfg_.joint_activation = kv_str(gguf_, "parakeet.joint.activation");
    cfg_.vocab_size  = kv_u32(gguf_, "parakeet.vocab_size");
    cfg_.blank_id    = kv_u32(gguf_, "parakeet.blank_id");
    // durations array
    { int id = gguf_find_key(gguf_, "parakeet.tdt.durations");
      if(id>=0){ size_t n = gguf_get_arr_n(gguf_,id); const int32_t* a=(const int32_t*)gguf_get_arr_data(gguf_,id);
                 cfg_.tdt_durations.assign(a, a+n); } }
    // tensors
    const int nt = (int)gguf_get_n_tensors(gguf_);
    for(int i=0;i<nt;++i){ const char* nm = gguf_get_tensor_name(gguf_,i);
        ggml_tensor* t = ggml_get_tensor(ctx_, nm); if(t) tensors_[nm]=t; }
    return cfg_.d_model>0 && cfg_.vocab_size>0;
}
ggml_tensor* ModelLoader::tensor(const std::string& n) const {
    auto it = tensors_.find(n); return it==tensors_.end()? nullptr : it->second;
}
}
```
Note: verify the exact ggml/gguf C API names against `third_party/ggml/include/gguf.h` (e.g. `gguf_get_val_u32`, `gguf_get_arr_data`). If a typed getter differs, adjust — the header is authoritative. Tokenizer pieces array read can be added when Phase 1 needs it; loading config + tensors is enough for this task.

- [ ] **Step 4: Wire CMake + build + run**

Add `src/model_loader.cpp` to `PARAKEET_SRC`. Add `pk_add_test(test_model_loader)` and `set_tests_properties(test_model_loader PROPERTIES LABELS "model")`.
Run:
```bash
cmake --build build -j
.venv/bin/python scripts/convert_parakeet_to_gguf.py --model nvidia/parakeet-tdt_ctc-110m --output /tmp/pk110m.gguf
PARAKEET_TEST_GGUF=/tmp/pk110m.gguf ctest --test-dir build -R test_model_loader --output-on-failure
```
Expected: PASS, prints `loader ok: arch=hybrid_tdt_ctc d_model=512 layers=17 heads=8 vocab=1024`.

- [ ] **Step 5: Commit**

```bash
git add src/model_loader.* CMakeLists.txt tests/CMakeLists.txt tests/test_model_loader.cpp
git commit -m "feat: GGUF model loader — ParakeetConfig + name->tensor"
```

---

## Task 6: CLI `info` subcommand

**Files:**
- Modify: `examples/cli/main.cpp`

- [ ] **Step 1: Replace the stub with an `info` subcommand**

`examples/cli/main.cpp`:
```cpp
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

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "info") == 0) return cmd_info(argv[2]);
    std::fprintf(stderr, "usage: parakeet-cli info <model.gguf>\n");
    return 2;
}
```
Ensure `examples/cli/CMakeLists.txt` links `parakeet` and includes `src` (already set in Task 2).

- [ ] **Step 2: Build + run manually**

Run:
```bash
cmake --build build -j
./build/bin/parakeet-cli info /tmp/pk110m.gguf || ./build/examples/cli/parakeet-cli info /tmp/pk110m.gguf
```
Expected: prints the config block with `arch: hybrid_tdt_ctc`, durations `[0,1,2,3,4]`. (Confirm the binary path; adjust the install/runtime output dir in CMake if needed.)

- [ ] **Step 3: Commit**

```bash
git add examples/cli/main.cpp
git commit -m "feat: parakeet-cli info subcommand"
```

---

## Task 7: NeMo baseline dumper

**Files:**
- Create: `scripts/gen_nemo_baseline.py`, `tests/python/check_baseline.py`

Dumps deterministic NeMo intermediates into `baseline.gguf` for Phase 1 parity. Tensors stored (verbatim, f32): `mel`, `subsampling_out`, `enc_layer_0`, `enc_layer_mid`, `enc_layer_last`, `encoder_out`, `ctc_logits`, and `ctc_token_ids` (int32). Audio is a fixed committed clip; `dither` forced to 0.

- [ ] **Step 1: Write the dumper**

`scripts/gen_nemo_baseline.py`:
```python
#!/usr/bin/env python3
"""Dump NeMo Parakeet intermediates to baseline.gguf for C++ parity tests."""
import argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, gguf
from nemo.collections.asr.models import ASRModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet-tdt_ctc-110m")
    ap.add_argument("--audio", required=True, help="16k mono wav")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    m = ASRModel.from_pretrained(args.model, map_location="cpu"); m.eval()
    m.preprocessor.featurizer.dither = 0.0  # determinism

    cap = {}
    def save(name):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out,(tuple,list)) else out
            if isinstance(t, torch.Tensor): cap[name] = t.detach().cpu().float().numpy()
        return fn
    m.preprocessor.register_forward_hook(save("mel"))
    m.encoder.pre_encode.register_forward_hook(save("subsampling_out"))
    n = len(m.encoder.layers)
    m.encoder.layers[0].register_forward_hook(save("enc_layer_0"))
    m.encoder.layers[n//2].register_forward_hook(save("enc_layer_mid"))
    m.encoder.layers[n-1].register_forward_hook(save("enc_layer_last"))
    m.encoder.register_forward_hook(save("encoder_out"))
    # CTC head logits: hybrid uses ctc_decoder; pure-CTC uses decoder
    head = getattr(m, "ctc_decoder", None) or m.decoder
    head.register_forward_hook(save("ctc_logits"))

    with torch.no_grad():
        hyps = m.transcribe([args.audio], batch_size=1)

    w = gguf.GGUFWriter(args.output, "parakeet-baseline")
    for k, v in cap.items():
        w.add_tensor(k, np.ascontiguousarray(np.squeeze(v)))
    # also store the reference CTC token ids (greedy) for end-to-end token parity
    # (extracted from the transcribe hypothesis if available; else argmax of logits)
    logits = np.squeeze(cap["ctc_logits"])
    ids = logits.argmax(-1).astype(np.int32) if logits.ndim==2 else logits.argmax(0).astype(np.int32)
    w.add_tensor("ctc_argmax_ids", np.ascontiguousarray(ids))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("baseline tensors:", {k: tuple(np.squeeze(v).shape) for k,v in cap.items()})

if __name__ == "__main__":
    main()
```
Note: confirm `ctc_logits` orientation (`[T, V]` vs `[V, T]`) when wiring the Phase 1 test; store as captured and document the axis in `docs/conversion.md`.

- [ ] **Step 2: Provide a tiny committed test clip**

Create a ~2s 16k mono wav with NeMo's preprocessing-friendly content. Generate one and commit it:
```bash
.venv/bin/python - <<'PY'
import numpy as np, soundfile as sf
sr=16000; t=np.linspace(0,2,sr*2,endpoint=False)
# a couple of tones; content doesn't matter for tensor parity, only determinism
x=0.2*np.sin(2*np.pi*180*t)+0.1*np.sin(2*np.pi*320*t)
import os; os.makedirs("tests/fixtures",exist_ok=True)
sf.write("tests/fixtures/clip.wav", x.astype(np.float32), sr)
PY
```

- [ ] **Step 3: Write the baseline check**

`tests/python/check_baseline.py`:
```python
#!/usr/bin/env python3
import os, subprocess, sys, tempfile
import gguf
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(tempfile.gettempdir(), "pk_baseline.gguf")
clip = os.path.join(root, "tests", "fixtures", "clip.wav")
r = subprocess.run([sys.executable, os.path.join(root,"scripts","gen_nemo_baseline.py"),
                    "--audio", clip, "--output", out], capture_output=True, text=True)
print(r.stdout); print(r.stderr, file=sys.stderr)
if r.returncode != 0: sys.exit(1)
reader = gguf.GGUFReader(out)
names = {t.name for t in reader.tensors}
for req in ["mel","subsampling_out","enc_layer_0","encoder_out","ctc_logits","ctc_argmax_ids"]:
    assert req in names, f"missing baseline tensor {req}"
print("check_baseline OK:", sorted(names)); sys.exit(0)
```

- [ ] **Step 4: Run it**

Run:
```bash
.venv/bin/python tests/python/check_baseline.py
```
Expected: prints baseline tensor shapes and `check_baseline OK: [...]`. Exit 0.

- [ ] **Step 5: Register ctest + commit**

Append to `tests/CMakeLists.txt`:
```cmake
if(PARAKEET_PY)
  add_test(NAME check_baseline COMMAND ${PARAKEET_PY} ${CMAKE_SOURCE_DIR}/tests/python/check_baseline.py)
  set_tests_properties(check_baseline PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")
endif()
```
```bash
git add scripts/gen_nemo_baseline.py tests/python/check_baseline.py tests/fixtures/clip.wav tests/CMakeLists.txt
git commit -m "feat: NeMo baseline dumper + fixture clip + check"
```

---

## Task 8: CI workflow + AGENTS.md

**Files:**
- Create: `.github/workflows/ci.yml`, `AGENTS.md`

- [ ] **Step 1: CI workflow (build + non-model tests)**

`.github/workflows/ci.yml`:
```yaml
name: ci
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { submodules: recursive }
      - name: configure
        run: cmake -B build -DPARAKEET_BUILD_TESTS=ON -DGGML_NATIVE=OFF
      - name: build
        run: cmake --build build -j
      - name: test (model-independent only)
        run: ctest --test-dir build --output-on-failure -LE model
```
`-LE model` excludes the labelled model-dependent tests (converter/loader/baseline), which need the venv + checkpoint. A separate `workflow_dispatch` job can run those once a models bundle is published (Phase 4).

- [ ] **Step 2: AGENTS.md**

Write a durable maintainer guide modeled on `/home/mudler/_git/rt-detr.cpp/AGENTS.md`: what the project is, repo layout, build commands, the GGUF schema summary (point to `docs/conversion.md`), the Python env setup, how to convert + dump baselines, the verbatim-tensor-name convention, and the test labels (`model`-labelled tests need the venv + cached checkpoint).

- [ ] **Step 3: Verify the model-independent suite is green**

Run:
```bash
cmake -B build -DPARAKEET_BUILD_TESTS=ON && cmake --build build -j
ctest --test-dir build --output-on-failure -LE model
```
Expected: `test_smoke`, `test_audio_io` PASS (2/2). Model-labelled tests excluded.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml AGENTS.md
git commit -m "ci: build + model-independent ctest; add AGENTS.md"
```

---

## Phase 0 done-when

- `cmake -B build -DPARAKEET_BUILD_TESTS=ON && cmake --build build -j` succeeds.
- `ctest -LE model` → `test_smoke`, `test_audio_io` pass.
- `scripts/convert_parakeet_to_gguf.py --model nvidia/parakeet-tdt_ctc-110m --output m.gguf` produces a GGUF; `check_convert` passes.
- `parakeet-cli info m.gguf` prints the validated config (arch hybrid_tdt_ctc, d_model 512, durations [0,1,2,3,4]).
- `test_model_loader` passes with `PARAKEET_TEST_GGUF=m.gguf`.
- `scripts/gen_nemo_baseline.py` produces `baseline.gguf`; `check_baseline` passes.

This unblocks Phase 1 (mel → subsampling → relpos attention → conformer → encoder → CTC, each parity-tested against `baseline.gguf`).

---

## Handoff to Phase 1 (sequence gate)

When every "Phase 0 done-when" item above passes and is committed, the next
sub-agent picks up the Phase 1 plan
(`docs/superpowers/plans/2026-05-28-phase-1-*.md`, written separately).

**Convention for all phase plans (so a fresh agent can chain them):**
- Each phase plan opens with an **Orientation** section (read spec → verify env
  → find resume point via `git log`) and an **entry gate** that re-runs the
  *previous* phase's "done-when" checklist. If any prior-phase check fails, stop
  and finish the previous phase first — do not build on a broken base.
- Each phase plan closes with a **done-when** checklist and a **Handoff** to the
  next phase.

**Phase 1 entry gate (what the Phase 1 agent will verify first):**
```bash
cmake -B build -DPARAKEET_BUILD_TESTS=ON && cmake --build build -j
ctest --test-dir build --output-on-failure -LE model      # test_smoke, test_audio_io green
.venv/bin/python scripts/convert_parakeet_to_gguf.py --model nvidia/parakeet-tdt_ctc-110m --output /tmp/pk110m.gguf
PARAKEET_TEST_GGUF=/tmp/pk110m.gguf ctest --test-dir build -R test_model_loader --output-on-failure
.venv/bin/python scripts/gen_nemo_baseline.py --model nvidia/parakeet-tdt_ctc-110m --audio tests/fixtures/clip.wav --output /tmp/baseline.gguf
```
All must succeed (or skip cleanly with 77 only where a model is genuinely
unavailable) before Phase 1 work begins.
