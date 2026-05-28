# parakeet.cpp — Model Publishing Manifest

This file lists the expected set of published GGUF models for parakeet.cpp.
Each row is one source checkpoint × one quantization variant.

The GGUFs themselves are **not committed** (`models/` is git-ignored); only this
manifest is tracked. Run `scripts/publish_hf.py` to produce the GGUFs locally
and (with `--upload`) push them to HuggingFace.

WER (word error rate) is measured against the NeMo reference on
`tests/fixtures/speech.wav` (LibriSpeech `2086-149220-0033`, ~7.4 s, English).
0.0 = byte-for-byte identical transcript. Source: `docs/parity.md` and
`docs/quantization.md`.

---

## Publish command

```bash
# Dry-run (safe — converts locally, prints what would be uploaded, no HF contact)
.venv/bin/python scripts/publish_hf.py --model nvidia/parakeet-tdt_ctc-110m

# Real upload (requires HF token)
.venv/bin/python scripts/publish_hf.py --model nvidia/parakeet-tdt_ctc-110m --upload
```

---

## Expected published set

### `nvidia/parakeet-tdt_ctc-110m` — Hybrid TDT+CTC, 110 M params

| Variant | HF repo | Approx size | WER vs NeMo (TDT) | WER vs NeMo (CTC) | Validated |
|---|---|---:|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-tdt_ctc-110m-f16`  | 255.1 MB | **0.0** | — | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-tdt_ctc-110m-q8_0` | 169.6 MB | **0.0** | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-tdt_ctc-110m-q4_k` | 125.3 MB | **0.0** | — | PASS |

### `nvidia/parakeet-tdt-0.6b-v2` — TDT hybrid, 0.6 B params

| Variant | HF repo | Approx size | WER vs NeMo (TDT) | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-tdt-0.6b-v2-f16`  | ~450 MB  | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-tdt-0.6b-v2-q8_0` | 862.0 MB | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-tdt-0.6b-v2-q4_k` | ~350 MB  | not yet measured | — |

### `nvidia/parakeet-tdt-0.6b-v3` — TDT hybrid (multilingual), 0.6 B params

| Variant | HF repo | Approx size | WER vs NeMo (TDT) | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-tdt-0.6b-v3-f16`  | ~450 MB | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-tdt-0.6b-v3-q8_0` | ~360 MB | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-tdt-0.6b-v3-q4_k` | ~230 MB | not yet measured | — |

### `nvidia/parakeet-tdt-1.1b` — Pure TDT, 1.1 B params

| Variant | HF repo | Approx size | WER vs NeMo (TDT) | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-tdt-1.1b-f16`  | ~750 MB  | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-tdt-1.1b-q8_0` | ~400 MB  | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-tdt-1.1b-q4_k` | ~300 MB  | not yet measured | — |

### `nvidia/parakeet-tdt_ctc-1.1b` — Hybrid TDT+CTC, 1.1 B params

| Variant | HF repo | Approx size | WER vs NeMo | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-tdt_ctc-1.1b-f16`  | ~750 MB | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-tdt_ctc-1.1b-q8_0` | ~400 MB | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-tdt_ctc-1.1b-q4_k` | ~300 MB | not yet measured | — |

### `nvidia/parakeet-ctc-0.6b` — Standalone CTC, 0.6 B params

| Variant | HF repo | Approx size | WER vs NeMo (CTC) | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-ctc-0.6b-f16`  | ~450 MB | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-ctc-0.6b-q8_0` | ~240 MB | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-ctc-0.6b-q4_k` | ~160 MB | not yet measured | — |

### `nvidia/parakeet-ctc-1.1b` — Standalone CTC, 1.1 B params

| Variant | HF repo | Approx size | WER vs NeMo (CTC) | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-ctc-1.1b-f16`  | ~750 MB | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-ctc-1.1b-q8_0` | ~400 MB | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-ctc-1.1b-q4_k` | ~300 MB | not yet measured | — |

### `nvidia/parakeet-rnnt-0.6b` — Standard RNNT, 0.6 B params

| Variant | HF repo | Approx size | WER vs NeMo (RNNT) | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-rnnt-0.6b-f16`  | ~450 MB | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-rnnt-0.6b-q8_0` | ~240 MB | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-rnnt-0.6b-q4_k` | ~160 MB | not yet measured | — |

### `nvidia/parakeet-rnnt-1.1b` — Standard RNNT, 1.1 B params

| Variant | HF repo | Approx size | WER vs NeMo (RNNT) | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-rnnt-1.1b-f16`  | ~750 MB | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-rnnt-1.1b-q8_0` | ~400 MB | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-rnnt-1.1b-q4_k` | ~300 MB | not yet measured | — |

---

## Notes

- `parakeet_realtime_eou_120m-v1` (streaming + EOU) is **not** in this manifest — it is
  deferred to Phase 5 (streaming + EOU support).
- `nvidia/parakeet-tdt-0.6b` (v1) is **not published** on HuggingFace (404/401); it was
  superseded by `parakeet-tdt-0.6b-v2`. See `docs/parity.md`.
- Approximate sizes for the 0.6 B / 1.1 B models are estimates based on the 110m
  compression ratios; exact sizes will be populated after conversion.
- Q4_K WER for models other than the 110m anchor has not yet been measured (the 110m
  measured WER 0.0 — see `docs/quantization.md`).
