#!/usr/bin/env python3
"""Benchmark runner: NeMo (PyTorch CPU) vs our C++/ggml engine, per model.

For every requested model this:

1. Converts the HF checkpoint to GGUF for each requested dtype (f32, q8_0),
   recording the GGUF file size, and counts the checkpoint tensors.
2. Runs a **NeMo pass** in a subprocess under ``/usr/bin/time -v`` so we capture
   the peak RSS of the PyTorch process. The pass pins threads, selects the head
   that matches our engine's default, and times each ``transcribe`` (batch=1).
3. Runs an **ours pass** per dtype: ``parakeet-cli bench`` under ``/usr/bin/time
   -v`` (peak RSS), parsing the per-file proc_ms / text JSON.
4. Computes, per engine/dtype: RTFx (Σaudio_sec/Σproc_sec), median per-file
   latency (ms), WER-vs-ground-truth (over files WITH a ref, normalized),
   agreement WER (NeMo-vs-ours, per file, aggregated), peak RSS (MB), GGUF size
   (MB), and checkpoint tensor count.
5. Writes ``<out>/<model>.json`` with all metrics + run metadata.

Optionally (``--thread-sweep 1,2,4,8,20``) sweeps the thread count on ONE model
(default ``parakeet-tdt_ctc-110m``) over the diverse manifest, both engines, and
writes ``<out>/threads.json``.

The head we pick mirrors our converter/engine default: the transducer head
(``rnnt``) for tdt/rnnt/tdt_ctc/eou models, and ``ctc`` for the ctc-* models —
so NeMo and ours decode the *same* head and the agreement WER is a real fidelity
check.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

# Shared, dependency-free WER metric.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_metrics import wer  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CLI = REPO / "build" / "examples" / "cli" / "parakeet-cli"
DEFAULT_MODELS_DIR = REPO / "benchmarks" / "models"
CONVERTER = REPO / "scripts" / "convert_parakeet_to_gguf.py"
NEMO_PASS = REPO / "scripts" / "_nemo_bench_pass.py"
PYTHON = sys.executable  # the .venv python that invoked us

MANIFESTS = {
    "librispeech": REPO / "benchmarks" / "librispeech_manifest.tsv",
    "diverse": REPO / "benchmarks" / "diverse_manifest.tsv",
}

# Short model key -> (HF id, head). The head matches OUR engine's default:
#   * ctc-*           -> standalone CTC          -> "ctc"
#   * rnnt-*          -> transducer (RNNT)       -> "rnnt"
#   * tdt-*, tdt_ctc-*, realtime eou -> transducer (TDT) -> "rnnt"
# (Our CLI maps both rnnt and tdt heads to its `tdt` selector.)
MODELS: dict[str, tuple[str, str]] = {
    "ctc-0.6b": ("nvidia/parakeet-ctc-0.6b", "ctc"),
    "ctc-1.1b": ("nvidia/parakeet-ctc-1.1b", "ctc"),
    "rnnt-0.6b": ("nvidia/parakeet-rnnt-0.6b", "rnnt"),
    "rnnt-1.1b": ("nvidia/parakeet-rnnt-1.1b", "rnnt"),
    "tdt-0.6b-v2": ("nvidia/parakeet-tdt-0.6b-v2", "rnnt"),
    "tdt-0.6b-v3": ("nvidia/parakeet-tdt-0.6b-v3", "rnnt"),
    "tdt-1.1b": ("nvidia/parakeet-tdt-1.1b", "rnnt"),
    "tdt_ctc-110m": ("nvidia/parakeet-tdt_ctc-110m", "rnnt"),
    "tdt_ctc-1.1b": ("nvidia/parakeet-tdt_ctc-1.1b", "rnnt"),
    "realtime_eou_120m-v1": ("nvidia/parakeet_realtime_eou_120m-v1", "rnnt"),
}

# Our CLI's --decoder selector for a given head.
CLI_DECODER = {"ctc": "ctc", "rnnt": "tdt"}


# ---------------------------------------------------------------------------
# /usr/bin/time -v helpers
# ---------------------------------------------------------------------------

def _run_under_time(cmd: list[str]) -> tuple[int, str, str, float | None]:
    """Run ``cmd`` under ``/usr/bin/time -v``; return (rc, stdout, stderr, rss_mb).

    Peak RSS is parsed from the "Maximum resident set size (kbytes)" line that
    /usr/bin/time -v writes to *its* stderr (mixed with the child's stderr).
    """
    full = ["/usr/bin/time", "-v", *cmd]
    proc = subprocess.run(full, capture_output=True, text=True)
    rss_mb = None
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", proc.stderr)
    if m:
        rss_mb = int(m.group(1)) / 1024.0
    return proc.returncode, proc.stdout, proc.stderr, rss_mb


# ---------------------------------------------------------------------------
# Manifest + reference handling
# ---------------------------------------------------------------------------

def load_refs(manifest: Path) -> dict[str, str]:
    """path -> reference text (may be empty), skipping ``#``/blank lines."""
    refs: dict[str, str] = {}
    with open(manifest) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t", 1)
            path = parts[0].strip()
            ref = parts[1].strip() if len(parts) > 1 else ""
            if path:
                refs[path] = ref
    return refs


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

# dtypes the Python converter produces directly (applies its quant allowlist).
CONVERTER_DTYPES = {"f32", "f16", "q8_0"}
# dtypes produced by `parakeet-cli quantize <in> <out> <type>` from an f32 GGUF.
CLI_QUANT_DTYPES = {"q4_0", "q5_0", "q4_k", "q5_k", "q6_k"}


def convert(hf_id: str, dtype: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, str(CONVERTER),
        "--model", hf_id,
        "--dtype", dtype,
        "--output", str(out_path),
    ]
    print(f"  [convert] {hf_id} -> {out_path.name} ({dtype}) …", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"converter failed (rc={res.returncode})\ncmd: {' '.join(cmd)}\n"
            f"stderr:\n{res.stderr[-2000:]}"
        )


def quantize_cli(cli: Path, src_f32: Path, out_path: Path, dtype: str) -> None:
    """Produce a K-quant (q4_k/q5_k/q6_k/q4_0/q5_0) GGUF from an f32 GGUF via
    the CLI `quantize` subcommand (same allowlist as the converter's q8_0)."""
    cmd = [str(cli), "quantize", str(src_f32), str(out_path), dtype]
    print(f"  [quantize] {src_f32.name} -> {out_path.name} ({dtype}) …", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"quantize failed (rc={res.returncode})\ncmd: {' '.join(cmd)}\n"
            f"stderr:\n{res.stderr[-2000:]}"
        )


def ensure_gguf(hf_id: str, model_name: str, dtype: str, models_dir: Path,
                cli: Path) -> Path:
    """Ensure a GGUF for (model, dtype) exists; return its path. K-quants are
    built from the f32 GGUF (converted first if needed)."""
    out = models_dir / f"{model_name}.{dtype}.gguf"
    if out.exists():
        return out
    if dtype in CONVERTER_DTYPES:
        convert(hf_id, dtype, out)
    elif dtype in CLI_QUANT_DTYPES:
        f32 = models_dir / f"{model_name}.f32.gguf"
        if not f32.exists():
            convert(hf_id, "f32", f32)
        quantize_cli(cli, f32, out, dtype)
    else:
        raise ValueError(f"unknown dtype '{dtype}'")
    return out


def count_checkpoint_tensors(gguf_path: Path) -> int:
    """Number of tensors in the GGUF (a proxy for checkpoint tensor count)."""
    try:
        import gguf  # type: ignore

        reader = gguf.GGUFReader(str(gguf_path))
        return len(reader.tensors)
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------

def run_nemo(hf_id: str, head: str, manifest: Path, threads: int) -> dict:
    """Run the NeMo pass subprocess; return {files, peak_rss_mb, nemo_version,...}."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_json = Path(tf.name)
    try:
        cmd = [
            PYTHON, str(NEMO_PASS),
            "--model", hf_id,
            "--manifest", str(manifest),
            "--head", head,
            "--threads", str(threads),
            "--out", str(out_json),
        ]
        rc, _stdout, stderr, rss_mb = _run_under_time(cmd)
        if rc != 0:
            raise RuntimeError(
                f"NeMo pass failed (rc={rc}) for {hf_id}\nstderr tail:\n"
                f"{stderr[-2000:]}"
            )
        doc = json.loads(out_json.read_text())
        doc["peak_rss_mb"] = rss_mb
        return doc
    finally:
        out_json.unlink(missing_ok=True)


def run_ours(gguf: Path, head: str, manifest: Path, threads: int, cli: Path) -> dict:
    """Run ``parakeet-cli bench`` under time -v; return {files, peak_rss_mb,...}."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_json = Path(tf.name)
    try:
        cmd = [
            str(cli), "bench",
            "--model", str(gguf),
            "--manifest", str(manifest),
            "--decoder", CLI_DECODER[head],
            "--threads", str(threads),
            "--json", str(out_json),
        ]
        rc, _stdout, stderr, rss_mb = _run_under_time(cmd)
        if rc != 0:
            raise RuntimeError(
                f"ours pass failed (rc={rc}) for {gguf.name}\nstderr tail:\n"
                f"{stderr[-2000:]}"
            )
        doc = json.loads(out_json.read_text())
        # Normalize to seconds (CLI reports proc_ms / load_ms).
        for f in doc["files"]:
            f["proc_s"] = f["proc_ms"] / 1000.0
        doc["load_s"] = doc.get("load_ms", 0.0) / 1000.0
        doc["peak_rss_mb"] = rss_mb
        return doc
    finally:
        out_json.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(files: list[dict], refs: dict[str, str]) -> dict:
    """RTFx, median latency, WER-vs-truth (files with a ref), over a file list."""
    total_audio = sum(f["audio_sec"] for f in files)
    total_proc = sum(f["proc_s"] for f in files)
    rtfx = (total_audio / total_proc) if total_proc > 0 else None
    latencies_ms = sorted(f["proc_s"] * 1000.0 for f in files)
    median_latency_ms = statistics.median(latencies_ms) if latencies_ms else None

    # WER vs ground truth over files that HAVE a non-empty reference.
    scored = [
        (refs.get(f["path"], ""), f["text"])
        for f in files
        if refs.get(f["path"], "").strip()
    ]
    if scored:
        wer_truth = statistics.mean(wer(r, h) for r, h in scored)
        wer_truth_n = len(scored)
    else:
        wer_truth = None
        wer_truth_n = 0

    return {
        "total_audio_sec": total_audio,
        "total_proc_sec": total_proc,
        "rtfx": rtfx,
        "median_latency_ms": median_latency_ms,
        "wer_vs_truth": wer_truth,
        "wer_vs_truth_n_files": wer_truth_n,
    }


def agreement_wer(nemo_files: list[dict], our_files: list[dict]) -> float | None:
    """Mean per-file WER(nemo_text, our_text) over the files both engines saw."""
    nemo_by_path = {f["path"]: f["text"] for f in nemo_files}
    pairs = [
        (nemo_by_path[f["path"]], f["text"])
        for f in our_files
        if f["path"] in nemo_by_path
    ]
    if not pairs:
        return None
    return statistics.mean(wer(n, o) for n, o in pairs)


# ---------------------------------------------------------------------------
# Per-model benchmark
# ---------------------------------------------------------------------------

def benchmark_model(
    key: str,
    manifests: list[Path],
    dtypes: list[str],
    threads: int,
    cli: Path,
    models_dir: Path,
    keep_models: bool,
    skip_nemo: bool = False,
    out_dir: Path | None = None,
) -> dict:
    hf_id, head = MODELS[key]
    model_name = hf_id.split("/", 1)[1]  # e.g. parakeet-tdt_ctc-110m
    print(f"=== {model_name}  (head={head}, threads={threads}"
          f"{', skip-nemo' if skip_nemo else ''}) ===", flush=True)

    # When skipping NeMo, reuse the NeMo block (and n_files/total_audio_sec) from
    # the previously-committed result — NeMo is the unchanged reference, and its
    # CPU pass on the 1.1B models is the slow part. Only OUR engine is re-measured.
    prior: dict | None = None
    if skip_nemo:
        prior_path = (out_dir or Path("benchmarks/results")) / f"{model_name}.json"
        if not prior_path.exists():
            raise RuntimeError(
                f"--skip-nemo: no prior result to reuse NeMo from at {prior_path}")
        prior = json.loads(prior_path.read_text())

    # Convert each requested dtype once (independent of manifest). f32/f16/q8_0
    # come from the Python converter; q4_k/q5_k/q6_k from `cli quantize` off f32.
    ggufs: dict[str, Path] = {}
    gguf_size_mb: dict[str, float] = {}
    ckpt_tensors: dict[str, int] = {}
    for dtype in dtypes:
        gguf = ensure_gguf(hf_id, model_name, dtype, models_dir, cli)
        ggufs[dtype] = gguf
        gguf_size_mb[dtype] = gguf.stat().st_size / (1024.0 * 1024.0)
        ckpt_tensors[dtype] = count_checkpoint_tensors(gguf)

    result: dict = {
        "model": model_name,
        "hf_id": hf_id,
        "head": head,
        "cli_decoder": CLI_DECODER[head],
        "threads": threads,
        "dtypes": dtypes,
        "gguf_size_mb": gguf_size_mb,
        "checkpoint_tensors": ckpt_tensors,
        "manifests": {},
    }

    for manifest in manifests:
        man_key = next(k for k, v in MANIFESTS.items() if v == manifest)
        refs = load_refs(manifest)
        print(f"  -- manifest: {man_key} ({len(refs)} files) --", flush=True)

        if skip_nemo:
            # Reuse the committed NeMo block verbatim (unchanged reference).
            assert prior is not None
            prior_man = prior["manifests"][man_key]
            nemo_block = prior_man["nemo"]
            nemo_files = nemo_block["files"]
            n_files = prior_man["n_files"]
            total_audio_sec = prior_man["total_audio_sec"]
            nemo_version = nemo_block.get("nemo_version")
        else:
            # NeMo pass once per manifest (head is fixed).
            nemo = run_nemo(hf_id, head, manifest, threads)
            nemo_metrics = compute_metrics(nemo["files"], refs)
            nemo_files = nemo["files"]
            n_files = len(nemo["files"])
            total_audio_sec = nemo_metrics["total_audio_sec"]
            nemo_version = nemo.get("nemo_version")
            nemo_block = {
                **nemo_metrics,
                "peak_rss_mb": nemo["peak_rss_mb"],
                "load_s": nemo.get("load_s"),
                "nemo_version": nemo_version,
                "files": [
                    {"path": f["path"], "audio_sec": f["audio_sec"],
                     "proc_s": f["proc_s"], "text": f["text"]}
                    for f in nemo["files"]
                ],
            }

        per_dtype: dict[str, dict] = {}
        for dtype in dtypes:
            ours = run_ours(ggufs[dtype], head, manifest, threads, cli)
            our_metrics = compute_metrics(ours["files"], refs)
            agree = agreement_wer(nemo_files, ours["files"])
            per_dtype[dtype] = {
                **our_metrics,
                "peak_rss_mb": ours["peak_rss_mb"],
                "load_s": ours["load_s"],
                "gguf_size_mb": gguf_size_mb[dtype],
                "checkpoint_tensors": ckpt_tensors[dtype],
                "agreement_wer_vs_nemo": agree,
                "files": [
                    {"path": f["path"], "audio_sec": f["audio_sec"],
                     "proc_s": f["proc_s"], "text": f["text"]}
                    for f in ours["files"]
                ],
            }
            print(
                f"     [{dtype}] ours RTFx={_fmt(our_metrics['rtfx'])} "
                f"WERtruth={_fmt(our_metrics['wer_vs_truth'])} "
                f"agreement_WER={_fmt(agree)} RSS={_fmt(per_dtype[dtype]['peak_rss_mb'])}MB",
                flush=True,
            )

        result["manifests"][man_key] = {
            "n_files": n_files,
            "total_audio_sec": total_audio_sec,
            "nemo": nemo_block,
            "ours": per_dtype,
        }
        result.setdefault("nemo_version", nemo_version)
        print(
            f"     [nemo{'*reused' if skip_nemo else ''}] "
            f"RTFx={_fmt(nemo_block.get('rtfx'))} "
            f"WERtruth={_fmt(nemo_block.get('wer_vs_truth'))} "
            f"RSS={_fmt(nemo_block.get('peak_rss_mb'))}MB",
            flush=True,
        )

    # Prune GGUFs unless asked to keep them (HF cache stays — we have headroom).
    # Includes any f32 base built solely to seed K-quants (not in `ggufs`).
    if not keep_models:
        to_remove = set(ggufs.values())
        to_remove.add(models_dir / f"{model_name}.f32.gguf")
        n = 0
        for gguf in to_remove:
            if gguf.exists():
                gguf.unlink()
                n += 1
        print(f"  [prune] removed {n} GGUF(s) for {model_name}", flush=True)

    return result


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


# ---------------------------------------------------------------------------
# Thread sweep
# ---------------------------------------------------------------------------

def thread_sweep(
    key: str,
    sweep: list[int],
    manifest: Path,
    dtype: str,
    cli: Path,
    models_dir: Path,
    keep_models: bool,
) -> dict:
    hf_id, head = MODELS[key]
    model_name = hf_id.split("/", 1)[1]
    refs = load_refs(manifest)
    man_key = next(k for k, v in MANIFESTS.items() if v == manifest)
    print(f"=== thread-sweep {model_name} on {man_key}: {sweep} ===", flush=True)

    gguf = models_dir / f"{model_name}.{dtype}.gguf"
    if not gguf.exists():
        convert(hf_id, dtype, gguf)

    points = []
    for n in sweep:
        nemo = run_nemo(hf_id, head, manifest, n)
        nemo_m = compute_metrics(nemo["files"], refs)
        ours = run_ours(gguf, head, manifest, n, cli)
        ours_m = compute_metrics(ours["files"], refs)
        points.append({
            "threads": n,
            "nemo": {"rtfx": nemo_m["rtfx"],
                     "median_latency_ms": nemo_m["median_latency_ms"],
                     "peak_rss_mb": nemo["peak_rss_mb"]},
            "ours": {"rtfx": ours_m["rtfx"],
                     "median_latency_ms": ours_m["median_latency_ms"],
                     "peak_rss_mb": ours["peak_rss_mb"]},
        })
        print(
            f"  threads={n}: nemo RTFx={_fmt(nemo_m['rtfx'])} "
            f"ours RTFx={_fmt(ours_m['rtfx'])}",
            flush=True,
        )

    if not keep_models:
        gguf.unlink(missing_ok=True)

    return {
        "model": model_name,
        "hf_id": hf_id,
        "head": head,
        "dtype": dtype,
        "manifest": man_key,
        "n_files": len(refs),
        "sweep": points,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models", default="all",
        help="'all' or a CSV of short keys (e.g. tdt_ctc-110m,ctc-0.6b)",
    )
    ap.add_argument(
        "--manifest", choices=["librispeech", "diverse", "both"], default="both",
    )
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument(
        "--dtypes", default="f32,q8_0",
        help="CSV of dtypes to convert+bench (default: f32,q8_0)",
    )
    ap.add_argument(
        "--keep-models", action="store_true",
        help="do not prune the converted GGUFs after each model",
    )
    ap.add_argument(
        "--skip-nemo", action="store_true",
        help="reuse the NeMo block from the existing results JSON (unchanged "
             "reference) and only re-measure OUR engine. Requires a prior result "
             "per model in --out.",
    )
    ap.add_argument(
        "--thread-sweep", default="",
        help="CSV of thread counts (e.g. 1,2,4,8,20) to sweep on one model",
    )
    ap.add_argument(
        "--sweep-model", default="tdt_ctc-110m",
        help="short key of the model to thread-sweep (default: tdt_ctc-110m)",
    )
    ap.add_argument("--cli", default=str(DEFAULT_CLI))
    ap.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    args = ap.parse_args()

    cli = Path(args.cli)
    if not cli.exists():
        print(f"benchmark: parakeet-cli not found at {cli}", file=sys.stderr)
        return 1
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]

    if args.models == "all":
        keys = list(MODELS)
    else:
        keys = [k.strip() for k in args.models.split(",") if k.strip()]
    for k in keys:
        if k not in MODELS:
            print(f"benchmark: unknown model key '{k}' (known: {', '.join(MODELS)})",
                  file=sys.stderr)
            return 1

    if args.manifest == "both":
        manifests = [MANIFESTS["librispeech"], MANIFESTS["diverse"]]
    else:
        manifests = [MANIFESTS[args.manifest]]
    for m in manifests:
        if not m.exists():
            print(f"benchmark: manifest not found: {m}", file=sys.stderr)
            return 1

    # Per-model benchmark.
    for key in keys:
        result = benchmark_model(
            key, manifests, dtypes, args.threads, cli, models_dir, args.keep_models,
            skip_nemo=args.skip_nemo, out_dir=out_dir,
        )
        out_path = out_dir / f"{result['model']}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  -> wrote {out_path}", flush=True)

    # Optional thread sweep on one model over the diverse manifest.
    if args.thread_sweep:
        sweep = [int(x) for x in args.thread_sweep.split(",") if x.strip()]
        sweep_doc = thread_sweep(
            args.sweep_model, sweep, MANIFESTS["diverse"],
            dtypes[0], cli, models_dir, args.keep_models,
        )
        sweep_path = out_dir / "threads.json"
        sweep_path.write_text(json.dumps(sweep_doc, indent=2))
        print(f"  -> wrote {sweep_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
