#!/usr/bin/env python3
"""Acceptance check for the NeMo -> GGUF converter.

Runs ``scripts/convert_parakeet_to_gguf.py`` on the anchor checkpoint, re-opens
the produced GGUF with ``gguf.GGUFReader``, and asserts the metadata-driven schema
(arch KV + verbatim NeMo tensor names + featurizer fb buffer + both decoder heads).

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
    print("check_convert: 'gguf' not installed; skipping", file=sys.stderr)
    sys.exit(77)

MODEL = os.environ.get("PARAKEET_TEST_MODEL", "nvidia/parakeet-tdt_ctc-110m")
out = os.path.join(tempfile.gettempdir(), "pk_check.gguf")

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
conv = os.path.join(root, "scripts", "convert_parakeet_to_gguf.py")

r = subprocess.run(
    [sys.executable, conv, "--model", MODEL, "--output", out],
    capture_output=True, text=True,
)
print(r.stdout, end="")
print(r.stderr, end="", file=sys.stderr)

# The converter exits 2 (and prints a marker) when NeMo/gguf are not installed.
# CI without the reference venv must skip, not fail.
if r.returncode == 2 or "PARAKEET_CONVERT_DEPS_MISSING" in r.stderr:
    print("check_convert: converter dependencies missing; skipping", file=sys.stderr)
    sys.exit(77)
if r.returncode != 0:
    print("check_convert: converter failed", file=sys.stderr)
    sys.exit(1)

reader = gguf.GGUFReader(out)
kv = {f.name: f for f in reader.fields.values()}

assert "general.architecture" in kv, "missing general.architecture"
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
