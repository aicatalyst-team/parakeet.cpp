#!/usr/bin/env python3
"""Acceptance check for the NeMo baseline dumper.

Runs ``scripts/gen_nemo_baseline.py`` on the committed fixture clip, re-opens the
produced ``baseline.gguf`` with ``gguf.GGUFReader``, and asserts that every
intermediate tensor Phase 1 parity tests rely on is present.

Exit codes (ctest convention): 0 = pass, 77 = skip (deps/model absent), 1 = fail.
"""
import os
import subprocess
import sys
import tempfile

# Skip cleanly if the GGUF reader itself is unavailable in this environment.
try:
    import gguf
except ImportError:
    print("check_baseline: 'gguf' not installed; skipping", file=sys.stderr)
    sys.exit(77)

MODEL = os.environ.get("PARAKEET_TEST_MODEL", "nvidia/parakeet-tdt_ctc-110m")
out = os.path.join(tempfile.gettempdir(), "pk_baseline.gguf")

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dumper = os.path.join(root, "scripts", "gen_nemo_baseline.py")
clip = os.path.join(root, "tests", "fixtures", "clip.wav")

r = subprocess.run(
    [sys.executable, dumper, "--model", MODEL, "--audio", clip, "--output", out],
    capture_output=True, text=True,
)
print(r.stdout, end="")
print(r.stderr, end="", file=sys.stderr)

# The dumper exits 2 (and prints a marker) when NeMo/gguf are not installed, or
# when the checkpoint cannot be obtained (no network, HF 403, etc.). CI without
# the reference venv or without model access must skip, not fail.
if (r.returncode == 2
        or "PARAKEET_CONVERT_DEPS_MISSING" in r.stderr
        or "PARAKEET_MODEL_UNAVAILABLE" in r.stderr):
    print("check_baseline: dumper dependencies or model unavailable; skipping",
          file=sys.stderr)
    sys.exit(77)
if r.returncode != 0:
    print("check_baseline: dumper failed", file=sys.stderr)
    sys.exit(1)

reader = gguf.GGUFReader(out)
names = {t.name for t in reader.tensors}
required = [
    "mel",
    "subsampling_out",
    "pos_emb",
    "enc_pre_layers",
    "l0_attn_out",
    "l0_conv_out",
    "enc_layer_0",
    "enc_layer_mid",
    "enc_layer_last",
    "encoder_out",
    "ctc_logits",
    "ctc_argmax_ids",
]
for req in required:
    assert req in names, f"missing baseline tensor {req}"

print("check_baseline OK:", sorted(names))
sys.exit(0)
