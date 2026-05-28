# parakeet.cpp — Parity report (Phase 1)

This document records the numerical and end-to-end parity of the C++/ggml
inference path against the NeMo reference implementation.

**Anchor checkpoint:** `nvidia/parakeet-tdt_ctc-110m` — the **CTC head** of the
hybrid TDT/CTC model (selected via `m.change_decoding_strategy(decoder_type='ctc')`).
NeMo version 2.7.3. All comparisons are CPU, batch size 1, deterministic
(CTC greedy decode).

---

## Real-speech end-to-end check (Task 12)

The decisive proof: a real English utterance transcribed by the full C++
pipeline (mel → subsampling → FastConformer encoder → CTC head → CTC greedy
decode → BPE detokenize) compared word-for-word against NeMo's CTC-head
transcript of the same clip.

**Fixture:** `tests/fixtures/speech.wav` — LibriSpeech dev-clean sample
`2086-149220-0033`, 16 kHz mono, 16-bit PCM, ~7.4 s. Obtained from the public
NeMo tutorial mirror
`https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav`
(already 16 kHz mono; re-encoded to canonical 16-bit PCM via soundfile).

This clip is substantially harder than the synthetic-tone clip used for the
per-stage tensor-parity tests: it is ~3.7x longer and contains real speech, so
it exercises per-feature normalization and valid-length masking at a different
sequence length `T`, and a non-trivial greedy collapse over a real token stream
— exactly the length-dependent code paths a single fixed-length tone clip could
not have validated.

| Source | Transcript |
| --- | --- |
| **NeMo CTC reference** | `Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.` |
| **C++ `parakeet-cli transcribe`** | `Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.` |

**Result:** byte-for-byte identical. **WER = 0.0** (0 edits over 23 reference
words).

Reproduce:

```bash
# NeMo CTC reference
.venv/bin/python -c "from nemo.collections.asr.models import ASRModel; \
m=ASRModel.from_pretrained('nvidia/parakeet-tdt_ctc-110m'); \
m.change_decoding_strategy(decoder_type='ctc'); \
print(m.transcribe(['tests/fixtures/speech.wav'])[0].text)"

# C++ pipeline
./build/examples/cli/parakeet-cli transcribe --model /tmp/pk110m.gguf \
    --input tests/fixtures/speech.wav
```

A committed regression test asserts this match deterministically:
`tests/test_transcribe_speech.cpp` (ctest `test_transcribe_speech`, label
`model`). It runs `pk::transcribe` on the committed `speech.wav` and asserts the
output equals the stored NeMo CTC reference string.

---

## Per-stage tensor parity (synthetic-tone clip)

Each stage diffs its C++ output against the matching tensor dumped from NeMo by
`scripts/gen_nemo_baseline.py` (stored in `baseline.gguf`). Numbers below are
the measured `max|diff|` from the ctest suite on `tests/fixtures/clip.wav`.

| Stage | ctest | max\|diff\| | Notes |
| --- | --- | --- | --- |
| Log-mel front end | `test_mel` | **4.13e-05** | `mel` `[80, T]` |
| dw_striding subsampling (÷8) | `test_subsampling` | **6.35e-03** | `subsampling_out`; largest-magnitude stage |
| Relative-position MHA (layer 0) | `test_relpos_attention` | **1.71e-03** | `l0_attn_out` |
| Conformer conv submodule | `test_conformer` | **1.58e-03** | `l0_conv_out` |
| Conformer layer 0 (full) | `test_conformer` | **5.34e-05** | `enc_layer_0` |
| Relative positional encoding | `test_encoder` | **1.55e-06** | `pos_emb` `[2T'-1, d_model]` |
| Encoder (full, last layer) | `test_encoder` | **2.40e-05** | `encoder_out` `[d_model, T']` |
| CTC head (log-softmax logits) | `test_ctc` | **1.72e-05** | `ctc_logits` `[T', vocab+1]` |

Notes:
- Diffs are absolute. The subsampling stage carries the largest absolute diff
  (6.3e-3) because its outputs have large magnitudes (O(1e3)); the relative
  error there is ~1e-6. Downstream stages (encoder, CTC) re-normalize and the
  absolute diff collapses back to O(1e-5).
- The CTC `exp`-row-sums equal 1.0 (the baseline stores post-`log_softmax`
  logits, and the C++ head matches that orientation).
- All stage diffs are well within the plan's tolerances and the greedy argmax /
  collapse is unaffected, which is why the end-to-end transcript is exact.

## Phase 2 — transducer core parity (max abs diff vs NeMo)

| Component | max\|diff\| | tolerance |
| --- | --- | --- |
| prediction net (embedding + 1-layer LSTM) | 1.5e-06 | 2e-3 |
| joint network (enc/pred proj → ReLU → linear, raw logits) | 1.1e-05 | 5e-3 |
| composed pred→joint (integration) | 1.3e-05 | 5e-3 |

- The joint emits **raw** logits `[T, U, 1030]` = 1024 vocab + 1 blank + 5 TDT
  durations `[0,1,2,3,4]`; the token/duration split + separate log_softmax is a
  Phase 3 (greedy) concern.
- The prediction net prepends a literal zero SOS step (`add_sos`), so 4 input
  ids → 5 hidden states. PyTorch LSTM gate order `[i,f,g,o]`, both biases summed.

## Phase 3 — North-star validation (TDT end-to-end vs NeMo)

The decisive Phase 3 deliverable: the real Parakeet-TDT models people use,
transcribed by the full C++ TDT path (mel → FastConformer encoder → prediction
net + joint → duration-aware greedy decode → BPE detokenize) and compared
word-for-word against NeMo's TDT-head transcript of the same clip. All on
`tests/fixtures/speech.wav` (LibriSpeech `2086-149220-0033`, English, ~7.4 s).
NeMo 2.7.3, CPU, batch 1, deterministic greedy.

### Model `info` (config read from the GGUF metadata)

| Model | arch | d_model / layers / heads | mels | conv norm | xscaling | durations | vocab | pred LSTM layers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `parakeet-tdt_ctc-110m` (TDT head) | `hybrid_tdt_ctc` | 512 / 17 / 8 | 80 | batch_norm | false | `[0,1,2,3,4]` | 1024 | 1 |
| `parakeet-tdt-0.6b-v2` | `hybrid_tdt_ctc` | 1024 / 24 / 8 | 128 | batch_norm | false | `[0,1,2,3,4]` | 1024 | 2 |
| `parakeet-tdt-0.6b-v3` | `hybrid_tdt_ctc` | 1024 / 24 / 8 | 128 | batch_norm | false | `[0,1,2,3,4]` | 8192 | 2 |

(`xscaling` is **false** on all three — NeMo's FastConformer `xscale=None`. The
0.6B checkpoints load in NeMo as `EncDecRNNTBPEModel`; the converter labels them
`hybrid_tdt_ctc` because `cfg.aux_ctc` is present, and the C++ default routes to
the TDT head either way, matching NeMo's default transducer transcription.)

### NeMo vs C++ transcripts + WER

| Model | NeMo (TDT) | C++ `parakeet-cli transcribe --decoder tdt` | WER |
| --- | --- | --- | --- |
| `parakeet-tdt_ctc-110m` | `Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.` | `Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.` | **0.0** |
| `parakeet-tdt-0.6b-v2` | `Well, I don't wish to see it any more, observed Phebe, turning away her eyes. It is certainly very like the old portrait.` | `Well, I don't wish to see it any more, observed Phebe, turning away her eyes. It is certainly very like the old portrait.` | **0.0** |
| `parakeet-tdt-0.6b-v3` | `Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.` | `Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.` | **0.0** |

All three are **byte-for-byte identical** to NeMo (0 edits over 23 reference
words). The only inter-model difference is the BPE spelling `Phebe` (v2 1024-token
vocab) vs `Phoebe` (v3 8192-token vocab) — both faithfully reproduce their own
tokenizer, exactly as NeMo does. This is the Phase 3 headline: the production
`parakeet-tdt-0.6b-v2` and multilingual `-v3` transcribe real speech identically
to NeMo through the C++/ggml port.

Reproduce:

```bash
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/parakeet-tdt-0.6b-v2 --output /tmp/tdt06v2.gguf
./build/examples/cli/parakeet-cli info /tmp/tdt06v2.gguf
./build/examples/cli/parakeet-cli transcribe \
    --model /tmp/tdt06v2.gguf --input tests/fixtures/speech.wav --decoder tdt
# NeMo reference (TDT is the default head for this model):
.venv/bin/python -c "from nemo.collections.asr.models import ASRModel; \
m=ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2'); \
print(m.transcribe(['tests/fixtures/speech.wav'])[0].text)"
```

### Code changes the 0.6B models required (metadata-honoring, not special-cased)

The 110m anchor (xscaling=False, 1-layer LSTM, conv/linear biases present) had
matched NeMo, so two latent assumptions surfaced only on the larger checkpoints.
The encoder output was already byte-exact on 0.6B-v2 (`enc_out` mean 1.80e-06,
std 0.0763, range ±0.69 — identical to NeMo); the divergences were:

1. **Optional Conv1d / Linear biases.** NeMo configures the FastConformer FFN
   and attention `nn.Linear`s and the conv submodule's `Conv1d`s with
   `bias=False` on `parakeet-tdt-0.6b-v2/-v3` (they are `bias=True` on the
   110m). The C++ conformer/attention unconditionally fetched the `.bias`
   tensors → null-deref / segfault. Fix: `src/conformer.cpp` and
   `src/relpos_attention.cpp` now add the bias only when the tensor is actually
   present in the checkpoint (`clone_weight_opt` / `ml.tensor(...)` guard).

2. **Stacked prediction LSTM.** `parakeet.decoder.pred_rnn_layers = 2` on the
   0.6B models (vs 1 on the 110m). The C++ `PredictionNet` ran only LSTM layer
   `_l0`, ignoring `_l1`, producing a wrong prediction-net output and a garbage
   transcript. Fix: `src/prediction.{hpp,cpp}` now read `pred_rnn_layers` from
   the GGUF and run a true stacked LSTM (`PredState` carries per-layer `(h,c)`;
   layer `l>0` consumes layer `l-1`'s hidden output), defaulting to 1 layer.

Both fixes honor the GGUF metadata / actual tensors; no model is special-cased.
Per-stage tensor parity (above) and the 110m end-to-end TDT/CTC transcripts are
unchanged (all earlier ctests still pass).

## Phase 3.5 — Standalone CTC coverage (xscaling=True, vs NeMo)

The standalone `EncDecCTCModelBPE` checkpoints (`parakeet-ctc-*`) are the first
end-to-end validation of two paths that no prior checkpoint exercised:

1. **Standalone CTC head name.** A hybrid model stores the CTC linear head at
   `ctc_decoder.decoder_layers.0.{weight,bias}`; a standalone CTC model stores
   the SAME layer at `decoder.decoder_layers.0.{weight,bias}`. `pk::CTCDecoder`
   now tries the hybrid name first and falls back to the standalone name (clear
   error only if neither exists) — `src/ctc_decoder.cpp`, no model special-cased.
2. **Encoder xscaling=True.** Every previously validated checkpoint had
   `xscaling=False`; `parakeet-ctc-0.6b`/`-1.1b` set NeMo's FastConformer
   `xscale=sqrt(d_model)`. This is the first real exercise of the C++ encoder's
   `*sqrt(d_model)` branch (`pk::Encoder`, gated on `cfg.xscaling`). It worked
   **out of the box** — no fix was needed; both transcripts matched NeMo exactly.

### Model `info` (config read from the GGUF metadata)

| Model | arch | d_model / layers / heads | mels | conv norm | xscaling | vocab | CTC head tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `parakeet-ctc-0.6b` | `ctc` | 1024 / 24 / 8 | 80 | batch_norm | **true** | 1024 | `decoder.decoder_layers.0.*` |
| `parakeet-ctc-1.1b` | `ctc` | 1024 / 42 / 8 | 80 | batch_norm | **true** | 1024 | `decoder.decoder_layers.0.*` |

### NeMo vs C++ transcripts + WER (`tests/fixtures/speech.wav`)

| Model | NeMo (CTC) | C++ `parakeet-cli transcribe --decoder ctc` | WER |
| --- | --- | --- | --- |
| `parakeet-ctc-0.6b` | `well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait` | `well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait` | **0.0** |
| `parakeet-ctc-1.1b` | `well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait` | `well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait` | **0.0** |

Both are **byte-for-byte identical** to NeMo (0 edits over 23 reference words).
These standalone CTC models emit lowercase, unpunctuated text (their own
tokenizer/training), distinct from the cased/punctuated hybrid + TDT models — and
the C++ port faithfully reproduces each model's own output. Validated with the
reusable harness `scripts/validate_vs_nemo.py`:

```bash
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/parakeet-ctc-0.6b --output /tmp/val_ctc06.gguf
./build/examples/cli/parakeet-cli info /tmp/val_ctc06.gguf   # arch=ctc, xscaling=true
.venv/bin/python scripts/validate_vs_nemo.py \
    --model nvidia/parakeet-ctc-0.6b --gguf /tmp/val_ctc06.gguf \
    --audio tests/fixtures/speech.wav --head ctc
# -> MODEL nvidia/parakeet-ctc-0.6b HEAD ctc arch=ctc xscaling=true WER 0.0000 ... PASS
```

The harness loads the NeMo model, auto-selects the head (CTC model→ctc,
RNNT/TDT→rnnt, hybrid→transducer), gets the NeMo reference via `m.transcribe`,
shells out to `parakeet-cli transcribe` (head ctc→`--decoder ctc`,
rnnt/tdt→`--decoder tdt`), computes word-level WER, and exits 0 on PASS / 1 on
WER>0 / 77 if NeMo can't be imported.

## Phase 3.5 — Standard RNNT coverage (vs NeMo)

The pure-RNNT `EncDecRNNTBPEModel` checkpoints (`parakeet-rnnt-*`) are the first
end-to-end validation of the **standard RNN-Transducer greedy loop** — a path no
prior checkpoint exercised. Every transducer validated in Phase 3 was a TDT model
(its joint emits `vocab+1+num_durations` and decoding is duration-aware). A pure
RNNT model has **no duration head**: its joint output is exactly `vocab+1`
(`Joint::num_durations()==0`, `V_plus=vocab+1`), and greedy decoding advances time
by exactly one frame on a blank and emits-and-stays on a non-blank (capped by
`max_symbols`).

1. **`pk::rnnt_greedy`** (`src/rnnt.hpp` / `src/rnnt.cpp`) ports NeMo
   `GreedyRNNTInfer._greedy_decode`. Versus `pk::tdt_greedy`: no duration logits
   (argmax over the full `vocab+1` joint output); blank → advance `t` by 1;
   non-blank → emit + commit state + stay at `t` (capped by `max_symbols=10`).
2. **Routing** (`src/parakeet.cpp`): when the transducer head is selected, the
   loader's `cfg.tdt_durations` now decides the loop — non-empty → `tdt_greedy`
   (unchanged); empty → `rnnt_greedy`. This makes `arch ∈ {rnnt, hybrid_rnnt_ctc}`
   (and any TDT-less transducer) work; it replaces the old guard that fell back to
   CTC when no durations were configured (which a pure-RNNT model lacks). TDT and
   CTC paths are unchanged. Both `parakeet-rnnt-*` models also set `xscaling=true`.

### Model `info` (config read from the GGUF metadata)

| Model | arch | d_model / layers / heads | mels | conv norm | xscaling | durations | vocab |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `parakeet-rnnt-0.6b` | `rnnt` | 1024 / 24 / 8 | 80 | batch_norm | **true** | none | 1024 |
| `parakeet-rnnt-1.1b` | `rnnt` | 1024 / 42 / 8 | 80 | batch_norm | **true** | none | 1024 |

### NeMo vs C++ transcripts + WER (`tests/fixtures/speech.wav`)

| Model | NeMo (RNNT) | C++ `parakeet-cli transcribe --decoder tdt` | WER |
| --- | --- | --- | --- |
| `parakeet-rnnt-0.6b` | `well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait` | `well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait` | **0.0** |
| `parakeet-rnnt-1.1b` | `well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait` | `well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait` | **0.0** |

Both are **byte-for-byte identical** to NeMo (0 edits over 23 reference words) —
the `rnnt_greedy` loop matched on the first run with no divergence debugging
needed. The CLI `--decoder tdt` selects the transducer head and now routes to
`rnnt_greedy` because the duration table is empty. Validated with the harness:

```bash
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/parakeet-rnnt-0.6b --output /tmp/val_rnnt06.gguf
./build/examples/cli/parakeet-cli info /tmp/val_rnnt06.gguf   # arch=rnnt, xscaling=true
.venv/bin/python scripts/validate_vs_nemo.py \
    --model nvidia/parakeet-rnnt-0.6b --gguf /tmp/val_rnnt06.gguf \
    --audio tests/fixtures/speech.wav --head rnnt
# -> MODEL nvidia/parakeet-rnnt-0.6b HEAD rnnt arch=rnnt xscaling=true WER 0.0000 ... PASS
```

## Phase 3.5 — Remaining TDT / hybrid checkpoints (vs NeMo)

The remaining published TDT and hybrid-TDT/CTC checkpoints, validated end-to-end
with `scripts/validate_vs_nemo.py` on `tests/fixtures/speech.wav` (NeMo 2.7.3,
CPU, batch 1, deterministic greedy). **No code change was required** — every one
loaded purely from GGUF metadata (the loader already handles 80/128 mel,
1024/8192 vocab, 1/2 LSTM layers, 24/42 layers, xscaling true/false, optional
biases). The 1.1B TDT model is the first checkpoint NeMo restores as a *pure*
`EncDecRNNTBPEModel` whose converter label is `tdt` (it has no `aux_ctc`), so it
also exercises the standalone-`tdt`-arch routing — distinct from the `-0.6b-v2/-v3`
hybrids that NeMo restores as RNNT but the converter labels `hybrid_tdt_ctc`.

### Model `info` (config read from the GGUF metadata)

| Model | HF id | arch | d_model / layers / heads | mels | conv norm | xscaling | durations | vocab |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parakeet-tdt-0.6b (v1) | `nvidia/parakeet-tdt-0.6b` | — | **N/A — not published** | — | — | — | — | — |
| parakeet-tdt-1.1b | `nvidia/parakeet-tdt-1.1b` | `tdt` | 1024 / 42 / 8 | 80 | batch_norm | false | `[0,1,2,3,4]` | 1024 |
| parakeet-tdt_ctc-1.1b | `nvidia/parakeet-tdt_ctc-1.1b` | `hybrid_tdt_ctc` | 1024 / 42 / 8 | 80 | batch_norm | false | `[0,1,2,3,4]` | 1024 |

The v1 `nvidia/parakeet-tdt-0.6b` is **not published** under that id (HF API 401 /
converter 404; the only public `nvidia` `parakeet-tdt` repos are `-0.6b-v2`,
`-0.6b-v3`, `-1.1b`, the `parakeet-tdt_ctc-*` hybrids, and `-tdt_ctc-0.6b-ja`).
It was superseded by `parakeet-tdt-0.6b-v2`, already validated at WER 0 in Phase 3.

### NeMo vs C++ transcripts + WER (`tests/fixtures/speech.wav`)

| Model | head | NeMo | C++ | WER |
| --- | --- | --- | --- | --- |
| `parakeet-tdt-1.1b` | TDT (`--decoder tdt`) | `well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait` | `well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait` | **0.0** |
| `parakeet-tdt_ctc-1.1b` | TDT (`--decoder tdt`) | `Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.` | `Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.` | **0.0** |
| `parakeet-tdt_ctc-1.1b` | CTC (`--decoder ctc`) | `Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.` | `Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.` | **0.0** |

All are **byte-for-byte identical** to NeMo (0 edits over 23 reference words). The
pure-TDT `parakeet-tdt-1.1b` emits lowercase/unpunctuated text (its own training);
the hybrid `parakeet-tdt_ctc-1.1b` emits cased/punctuated text on *both* of its
heads — and the C++ port reproduces each head exactly, including the second head
(CTC) of the hybrid via `--decoder ctc`. Reproduce:

```bash
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/parakeet-tdt_ctc-1.1b --output /tmp/val_tdtctc11.gguf
./build/examples/cli/parakeet-cli info /tmp/val_tdtctc11.gguf   # arch=hybrid_tdt_ctc
.venv/bin/python scripts/validate_vs_nemo.py \
    --model nvidia/parakeet-tdt_ctc-1.1b --gguf /tmp/val_tdtctc11.gguf \
    --audio tests/fixtures/speech.wav --head rnnt   # TDT head; --head ctc for the CTC head
# -> MODEL nvidia/parakeet-tdt_ctc-1.1b HEAD rnnt arch=hybrid_tdt_ctc xscaling=false WER 0.0000 ... PASS
```

## Test suite status

`ctest --test-dir build --output-on-failure` (with `PARAKEET_TEST_GGUF`,
`PARAKEET_TEST_BASELINE`, `PARAKEET_TEST_BASELINE_SPEECH` exported): all 21
runnable tests pass — the 15 Phase 0/1 tests, `test_prediction`, `test_joint`,
`test_transducer_core`, `test_prediction_step`, `test_tdt_greedy`,
`test_transcribe_tdt`. `test_transcribe_0_6b` skips (exit 77) unless
`PARAKEET_TEST_GGUF_06B` points at a converted 0.6B GGUF (a ~2.4GB download not
present in CI); with it set, it asserts the v2/v3 transcript matches NeMo.
