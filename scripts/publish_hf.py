#!/usr/bin/env python3
"""Publish parakeet.cpp GGUF models to HuggingFace Hub.

For a given source NeMo checkpoint (e.g. ``nvidia/parakeet-tdt_ctc-110m``),
this script:

1. Converts the checkpoint to F16 and Q8_0 variants via
   ``scripts/convert_parakeet_to_gguf.py``.
2. Re-quantizes the F32 intermediate to Q4_K via
   ``build/examples/cli/parakeet-cli quantize``.
3. Generates a per-model README (model card) with architecture details,
   variant sizes, validated WER (from ``docs/parity.md``), and a usage snippet.
4. Uploads the GGUFs + README to ``mudler/parakeet.cpp-<variant>`` on HF Hub.

**DRY-RUN BY DEFAULT** — without ``--upload`` it prints what it *would* upload
(repo id, files, sizes) but does NOT contact HuggingFace at all. Pass
``--upload`` to perform the real push.

Usage:
    .venv/bin/python scripts/publish_hf.py \\
        --model nvidia/parakeet-tdt_ctc-110m      # dry-run (safe default)

    .venv/bin/python scripts/publish_hf.py \\
        --model nvidia/parakeet-tdt_ctc-110m \\
        --upload                                  # actually push to HF
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONVERTER = SCRIPTS_DIR / "convert_parakeet_to_gguf.py"
# Prefer build/ CLI; fall back to build-shared/ (or whatever exists).
_CLI_CANDIDATES = [
    REPO_ROOT / "build" / "examples" / "cli" / "parakeet-cli",
    REPO_ROOT / "build-shared" / "examples" / "cli" / "parakeet-cli",
]
PYTHON = Path(sys.executable)

# ---------------------------------------------------------------------------
# Publish settings
# ---------------------------------------------------------------------------
HF_USER = "mudler"
DEFAULT_REPO_PREFIX = f"{HF_USER}/parakeet.cpp-"
DEFAULT_VARIANTS = ["f16", "q8_0", "q4_k"]

# Validated WER data sourced from docs/parity.md and docs/quantization.md.
# Structure: {hf_model_id: {variant: {"wer": float|str, "size_mb": float|str}}}
KNOWN_WER: dict = {
    "nvidia/parakeet-tdt_ctc-110m": {
        "f16":  {"wer": 0.0, "size_mb": 255.1},
        "q8_0": {"wer": 0.0, "size_mb": 169.6},
        "q4_k": {"wer": 0.0, "size_mb": 125.3},
    },
    "nvidia/parakeet-tdt-0.6b-v2": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": 862.0},
        "q4_k": {"wer": None, "size_mb": None},
    },
    "nvidia/parakeet-tdt-0.6b-v3": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": None},
        "q4_k": {"wer": None, "size_mb": None},
    },
    "nvidia/parakeet-tdt-1.1b": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": None},
        "q4_k": {"wer": None, "size_mb": None},
    },
    "nvidia/parakeet-tdt_ctc-1.1b": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": None},
        "q4_k": {"wer": None, "size_mb": None},
    },
    "nvidia/parakeet-ctc-0.6b": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": None},
        "q4_k": {"wer": None, "size_mb": None},
    },
    "nvidia/parakeet-ctc-1.1b": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": None},
        "q4_k": {"wer": None, "size_mb": None},
    },
    "nvidia/parakeet-rnnt-0.6b": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": None},
        "q4_k": {"wer": None, "size_mb": None},
    },
    "nvidia/parakeet-rnnt-1.1b": {
        "f16":  {"wer": 0.0, "size_mb": None},
        "q8_0": {"wer": 0.0, "size_mb": None},
        "q4_k": {"wer": None, "size_mb": None},
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_cli() -> Path:
    for p in _CLI_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "parakeet-cli not found in build/ or build-shared/. "
        "Run: cmake -B build -DPARAKEET_BUILD_CLI=ON && cmake --build build -j"
    )


def _model_slug(model_id: str) -> str:
    """``nvidia/parakeet-tdt_ctc-110m`` → ``parakeet-tdt_ctc-110m``."""
    return model_id.split("/")[-1]


def _gguf_name(model_id: str, variant: str) -> str:
    return f"{_model_slug(model_id)}-{variant}.gguf"


def _run(cmd: List[str], *, label: str) -> None:
    """Run *cmd* and propagate failures with a clear message."""
    print(f"  [{label}] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}"
        )


# ---------------------------------------------------------------------------
# Conversion + quantization
# ---------------------------------------------------------------------------

def convert_variant(model_id: str, variant: str, output_dir: Path, f32_path: Path) -> Path:
    """Convert/quantize ``model_id`` to *variant* and return the output path.

    * ``f16`` / ``q8_0``: call the Python converter with ``--dtype``.
    * ``q4_k``: call ``parakeet-cli quantize`` on the F32 GGUF.
    * ``f32``: call the Python converter with ``--dtype f32`` (intermediate).
    """
    out = output_dir / variant / _gguf_name(model_id, variant)
    out.parent.mkdir(parents=True, exist_ok=True)

    if variant in ("f16", "q8_0"):
        _run(
            [
                str(PYTHON), str(CONVERTER),
                "--model", model_id,
                "--output", str(out),
                "--dtype", variant,
            ],
            label=f"convert {variant}",
        )
    elif variant == "f32":
        _run(
            [
                str(PYTHON), str(CONVERTER),
                "--model", model_id,
                "--output", str(out),
                "--dtype", "f32",
            ],
            label="convert f32",
        )
    elif variant.startswith("q") and "_k" in variant:
        # K-quant: re-quantize from f32
        if not f32_path.is_file():
            raise FileNotFoundError(
                f"F32 intermediate not found at {f32_path}; "
                "cannot produce K-quant without it"
            )
        cli = _find_cli()
        _run(
            [str(cli), "quantize", str(f32_path), str(out), variant],
            label=f"quantize {variant}",
        )
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    if not out.is_file():
        raise RuntimeError(f"Expected output not produced: {out}")
    return out


def build_variants(
    model_id: str,
    variants: List[str],
    output_dir: Path,
) -> dict[str, Path]:
    """Build all requested variants; return {variant: path}."""
    slug = _model_slug(model_id)
    results: dict[str, Path] = {}

    # We may need F32 as an intermediate for K-quants.
    needs_f32 = any(v.endswith("_k") or "_k" in v for v in variants)
    f32_path = output_dir / "f32" / f"{slug}-f32.gguf"

    if needs_f32 and not f32_path.is_file():
        print(f"\n--- Converting {model_id} → f32 (intermediate for K-quants) ---")
        f32_path = convert_variant(model_id, "f32", output_dir, f32_path)

    for variant in variants:
        print(f"\n--- Converting {model_id} → {variant} ---")
        path = convert_variant(model_id, variant, output_dir, f32_path)
        results[variant] = path
        size_mb = path.stat().st_size / 1e6
        print(f"  produced: {path} ({size_mb:.1f} MB)")

    return results


# ---------------------------------------------------------------------------
# Model card generation
# ---------------------------------------------------------------------------

def _wer_str(wer) -> str:
    if wer is None:
        return "not measured"
    return f"{wer:.4f}"


def _size_str(size_mb, path: Optional[Path] = None) -> str:
    if path is not None and path.is_file():
        return f"{path.stat().st_size / 1e6:.1f} MB"
    if size_mb is not None:
        return f"{size_mb:.1f} MB"
    return "—"


def build_model_card(
    model_id: str,
    variants: List[str],
    variant_paths: dict[str, Path],
    repo_prefix: str,
) -> str:
    """Generate a Markdown model card for one model × all its variants."""
    slug = _model_slug(model_id)
    wer_data = KNOWN_WER.get(model_id, {})

    # Infer family / arch hint from model name
    name_lower = model_id.lower()
    if "tdt_ctc" in name_lower:
        arch_desc = "Hybrid TDT+CTC (FastConformer)"
        heads = "TDT + CTC"
    elif "tdt" in name_lower:
        arch_desc = "TDT transducer (FastConformer)"
        heads = "TDT"
    elif "rnnt" in name_lower:
        arch_desc = "RNNT transducer (FastConformer)"
        heads = "RNNT"
    elif "ctc" in name_lower:
        arch_desc = "CTC (FastConformer)"
        heads = "CTC"
    else:
        arch_desc = "Parakeet (FastConformer)"
        heads = "default"

    lines: List[str] = []

    # YAML frontmatter
    lines += [
        "---",
        "license: cc-by-4.0",
        "library_name: parakeet.cpp",
        "tags:",
        "  - automatic-speech-recognition",
        "  - asr",
        "  - parakeet",
        "  - gguf",
        "  - ggml",
        "  - cpp-inference",
        "  - nemo",
        "pipeline_tag: automatic-speech-recognition",
        f"base_model: {model_id}",
        "---",
        "",
    ]

    lines.append(f"# {slug} — GGUF for parakeet.cpp")
    lines.append("")
    lines.append(
        f"GGUF-format weights of [{model_id}](https://huggingface.co/{model_id}) "
        f"for use with [parakeet.cpp](https://github.com/mudler/parakeet.cpp), "
        f"a C++/ggml port of NVIDIA NeMo Parakeet that matches the upstream PyTorch "
        f"model on CPU."
    )
    lines.append("")
    lines.append(
        "This repo contains the quantized variants listed below. "
        "**F16 is the recommended default** — same accuracy as F32, ~1.7× smaller, "
        "and typically the fastest on modern CPUs via ggml's F32×F16 matmul fast path."
    )
    lines.append("")

    # Files table
    lines.append("## Available files")
    lines.append("")
    lines.append("| File | Variant | Size | WER vs NeMo |")
    lines.append("|---|---|---:|---:|")
    for v in variants:
        path = variant_paths.get(v)
        wd = wer_data.get(v, {})
        wer = _wer_str(wd.get("wer"))
        size = _size_str(wd.get("size_mb"), path)
        fname = _gguf_name(model_id, v)
        rec = " ← **recommended**" if v == "f16" else ""
        lines.append(f"| `{fname}`{rec} | {v.upper()} | {size} | {wer} |")
    lines.append("")

    lines.append(
        "> WER (word error rate) is computed against the upstream NeMo reference on "
        "`tests/fixtures/speech.wav` (LibriSpeech `2086-149220-0033`, ~7.4 s, English). "
        "0.0 = byte-for-byte identical transcript. "
        "See [parity.md](https://github.com/mudler/parakeet.cpp/blob/main/docs/parity.md) "
        "and [quantization.md](https://github.com/mudler/parakeet.cpp/blob/main/docs/quantization.md) "
        "for the full validation suite."
    )
    lines.append("")

    # Architecture
    lines.append("## Architecture")
    lines.append("")
    lines.append(f"- Source checkpoint: `{model_id}`")
    lines.append(f"- Architecture: {arch_desc}")
    lines.append(f"- Decoder head(s): {heads}")
    lines.append(f"- Upstream: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)")
    lines.append("")

    # Quantization notes
    lines.append("## Quantization notes")
    lines.append("")
    lines.append(
        "Quantization is applied **only** to the large linear weights fed directly "
        "into `ggml_mul_mat` (encoder FFN + attention projections, subsampling output "
        "projection, joint enc/pred projections). All other tensors (mel filterbank, "
        "LSTM prediction net, conv kernels, batch_norm stats, norms, biases, embeddings) "
        "stay F32."
    )
    lines.append("")
    lines.append("| Variant | What is quantized | Notes |")
    lines.append("|---|---|---|")
    lines.append(
        "| F16 | allowlisted linear weights → IEEE half | lossless; WER 0.0 on 110m |"
    )
    lines.append(
        "| Q8_0 | allowlisted linear weights → Q8_0 (8-bit, 32-element blocks) | WER 0.0 on 110m and 0.6B |"
    )
    lines.append(
        "| Q4_K | allowlisted linear weights → Q4_K (K-quant, 256-element superblocks) | "
        "`joint.pred.weight` stays F32 when ne[0] % 256 ≠ 0; WER 0.0 on 110m |"
    )
    lines.append("")

    # Usage
    lines.append("## Usage")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1. Clone + build parakeet.cpp")
    lines.append("git clone https://github.com/mudler/parakeet.cpp")
    lines.append("cd parakeet.cpp")
    lines.append("cmake -B build -DPARAKEET_BUILD_CLI=ON && cmake --build build -j")
    lines.append("")
    lines.append("# 2. Download a quant (F16 recommended)")
    repo_id_example = f"{repo_prefix}{slug}-f16"
    lines.append(
        f"huggingface-cli download {repo_id_example} {_gguf_name(model_id, 'f16')} "
        f"--local-dir models/"
    )
    lines.append("")
    lines.append("# 3. Transcribe")
    lines.append("build/examples/cli/parakeet-cli transcribe \\")
    lines.append(f"    --model models/{_gguf_name(model_id, 'f16')} \\")
    lines.append("    --input audio.wav")
    lines.append("```")
    lines.append("")

    # License
    lines.append("## License")
    lines.append("")
    lines.append(
        "The GGUF weights are derived from the NVIDIA NeMo Parakeet checkpoints, "
        "which are released under the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license. "
        "The parakeet.cpp runtime is MIT-licensed."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def _upload_with_retry(api, local_path: Path, remote_name: str, repo_id: str, max_retries: int = 3) -> None:
    """Upload one file with exponential backoff."""
    from huggingface_hub.utils import HfHubHTTPError  # type: ignore[attr-defined]

    size_mb = local_path.stat().st_size / 1e6
    print(f"  -> {remote_name} ({size_mb:.1f} MB)... ", end="", flush=True)
    delay = 2.0
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_name,
                repo_id=repo_id,
                repo_type="model",
            )
            dt = time.time() - t0
            mbps = size_mb / dt if dt > 0 else 0.0
            print(f"ok ({dt:.1f}s, {mbps:.1f} MB/s)")
            return
        except (HfHubHTTPError, OSError, ConnectionError) as e:
            last_err = e
            print(
                f"\n     attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(
        f"upload failed after {max_retries} attempts: {last_err}"
    ) from last_err


def publish_model(
    model_id: str,
    variants: List[str],
    variant_paths: dict[str, Path],
    *,
    repo_prefix: str,
    upload: bool,
) -> dict:
    """Publish one model to HF (or print what would happen in dry-run mode)."""
    slug = _model_slug(model_id)

    results = []
    for variant in variants:
        repo_id = f"{repo_prefix}{slug}-{variant}"
        path = variant_paths[variant]
        card = build_model_card(model_id, [variant], {variant: path}, repo_prefix)
        size_mb = path.stat().st_size / 1e6

        print(f"\n=== {repo_id} ===")
        print(f"  file: {path.name} ({size_mb:.1f} MB)")

        if not upload:
            print(f"  [dry-run] would create repo {repo_id} (public)")
            print(f"  [dry-run] would upload README.md ({len(card)} bytes)")
            print(f"  [dry-run] would upload {path.name} ({size_mb:.1f} MB)")
            results.append({
                "repo_id": repo_id,
                "url": f"https://huggingface.co/{repo_id}",
                "variant": variant,
                "file": path.name,
                "size_mb": size_mb,
                "dry_run": True,
            })
        else:
            from huggingface_hub import HfApi  # type: ignore[attr-defined]

            api = HfApi()
            api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
            print(f"  uploading README.md ({len(card)} bytes)")
            api.upload_file(
                path_or_fileobj=card.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add model card for {slug} {variant}",
            )
            _upload_with_retry(api, path, path.name, repo_id)
            results.append({
                "repo_id": repo_id,
                "url": f"https://huggingface.co/{repo_id}",
                "variant": variant,
                "file": path.name,
                "size_mb": size_mb,
                "dry_run": False,
            })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="HF_ID",
        help="Source HuggingFace model id, e.g. nvidia/parakeet-tdt_ctc-110m",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        metavar="VAR,...",
        help=(
            "Comma-separated list of quantization variants to produce. "
            f"Supported: f16, q8_0, q4_k. Default: {','.join(DEFAULT_VARIANTS)}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "models"),
        metavar="DIR",
        help="Directory to write GGUFs under (sub-dirs per variant). Default: models/",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help=(
            "Actually push to HuggingFace Hub. "
            "Without this flag the script is a dry run: it converts/quantizes "
            "the GGUFs locally but does NOT contact HuggingFace."
        ),
    )
    parser.add_argument(
        "--repo-prefix",
        default=DEFAULT_REPO_PREFIX,
        metavar="PREFIX",
        help=(
            f"HF repo-id prefix. Each variant is published to "
            f"<prefix><model-slug>-<variant>. Default: {DEFAULT_REPO_PREFIX}"
        ),
    )
    args = parser.parse_args()

    model_id: str = args.model
    variants: List[str] = [v.strip() for v in args.variants.split(",") if v.strip()]
    output_dir = Path(args.output_dir)
    upload: bool = args.upload
    repo_prefix: str = args.repo_prefix

    # Validate variants
    supported = {"f16", "q8_0", "q4_k", "q5_k", "q6_k", "q4_0", "q5_0", "q8_0"}
    bad = [v for v in variants if v not in supported]
    if bad:
        print(f"error: unknown variants: {bad}. Supported: {sorted(supported)}", file=sys.stderr)
        return 2

    # Sanity check: converter must exist
    if not CONVERTER.is_file():
        print(f"error: converter not found: {CONVERTER}", file=sys.stderr)
        return 2

    # Auth check (only when uploading)
    if upload:
        try:
            from huggingface_hub import HfApi  # type: ignore[attr-defined]

            api = HfApi()
            me = api.whoami()
            print(f"authenticated as: {me['name']}")
        except Exception as e:
            print(f"error: HF auth failed: {e}", file=sys.stderr)
            print(
                "hint: run `huggingface-cli login` or place a token in "
                "~/.cache/huggingface/token",
                file=sys.stderr,
            )
            return 2
    else:
        print("[dry-run mode] No HuggingFace calls will be made. Pass --upload to push.")
        print()

    # Step 1: Convert + quantize
    print(f"=== Building variants for {model_id} ===")
    t0 = time.time()
    try:
        variant_paths = build_variants(model_id, variants, output_dir)
    except Exception as e:
        print(f"error: conversion/quantization failed: {e}", file=sys.stderr)
        return 1

    # Step 2: Publish (or dry-run)
    try:
        all_results = publish_model(
            model_id,
            variants,
            variant_paths,
            repo_prefix=repo_prefix,
            upload=upload,
        )
    except Exception as e:
        print(f"error: publish failed: {e}", file=sys.stderr)
        return 1

    elapsed = time.time() - t0

    # Summary
    print()
    print("=" * 60)
    mode = "UPLOAD PLAN (dry-run)" if not upload else "UPLOAD SUMMARY"
    print(mode)
    print("=" * 60)
    print(f"{'Repo':55s} {'File':35s} {'Size':>10s}")
    for r in all_results:
        marker = "  [dry-run]" if r["dry_run"] else ""
        print(f"{r['repo_id']:55s} {r['file']:35s} {r['size_mb']:>8.1f} MB{marker}")
        print(f"  {'(would be at)' if r['dry_run'] else 'published at'}: {r['url']}")
    print()
    print(f"Elapsed: {elapsed:.1f}s")
    if not upload:
        print()
        print("NOTE: This was a dry run. Re-run with --upload to push to HuggingFace.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
