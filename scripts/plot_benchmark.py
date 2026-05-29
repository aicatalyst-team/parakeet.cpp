#!/usr/bin/env python3
"""
plot_benchmark.py — Generate benchmark plots from parakeet.cpp JSON results.

Usage:
    python scripts/plot_benchmark.py --results benchmarks/results --out benchmarks/plots

Handles an arbitrary set of our dtypes (f32, f16, q8_0, q6_k, q5_k, q4_k, …):
the per-model comparison plots show NeMo + every dtype present, and dedicated
quantization plots show the size / speed / accuracy tradeoff across dtypes.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Style ──────────────────────────────────────────────────────────────────────

# Canonical dtype order (high precision -> low). Colours go warm->cool->warm to
# read as a precision gradient (f32 orange … q4_k red).
DTYPE_ORDER = ["f32", "f16", "q8_0", "q6_k", "q5_k", "q4_k", "q5_0", "q4_0"]

COLORS = {
    "nemo": "#4e79a7",  # blue  – NeMo / PyTorch reference
    "f32":  "#f28e2b",  # orange
    "f16":  "#edc948",  # yellow
    "q8_0": "#59a14f",  # green
    "q6_k": "#76b7b2",  # teal
    "q5_k": "#b07aa1",  # purple
    "q4_k": "#e15759",  # red
    "q5_0": "#9c755f",  # brown
    "q4_0": "#bab0ac",  # grey
}

HATCHES = {"nemo": "", "f32": "//", "q8_0": "xx", "q6_k": "..",
           "q5_k": "\\\\", "q4_k": "oo", "f16": "--", "q5_0": "++", "q4_0": "**"}

LABELS = {"nemo": "NeMo"}  # dtype keys fall back to their own name


def _label(key: str) -> str:
    return LABELS.get(key, key)


plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.4,
    "font.size": 10,
})


# ── Data loading ───────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> list[dict]:
    """Load all <model>.json files (skip threads.json) and return sorted list."""
    models = []
    for p in sorted(results_dir.glob("*.json")):
        if p.stem == "threads":
            continue
        with open(p) as f:
            models.append(json.load(f))
    return models


def load_threads(results_dir: Path) -> list[dict] | None:
    p = results_dir / "threads.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        return d["sweep"] if isinstance(d, dict) and "sweep" in d else d
    return None


def dtypes_present(models: list[dict]) -> list[str]:
    """Union of our dtypes across all models, in canonical order."""
    seen: set[str] = set()
    for m in models:
        seen.update(m["manifests"]["librispeech"]["ours"].keys())
    ordered = [d for d in DTYPE_ORDER if d in seen]
    ordered += sorted(d for d in seen if d not in DTYPE_ORDER)
    return ordered


def _short(name: str) -> str:
    return name.replace("parakeet-", "").replace("parakeet_realtime_eou_", "rt-eou-")


# ── Grouped-bar helper ───────────────────────────────────────────────────────────

def _grouped_bar(ax, labels, groups: dict[str, list[float]], ylabel: str, title: str):
    """groups = {color_key: [values per model]} in draw order."""
    n_models = len(labels)
    n_series = len(groups)
    width = 0.8 / max(n_series, 1)
    x = np.arange(n_models)
    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

    for (key, values), offset in zip(groups.items(), offsets):
        color = COLORS.get(key, "#888888")
        hatch = HATCHES.get(key, "")
        bars = ax.bar(
            x + offset, values, width,
            label=_label(key), color=color, hatch=hatch,
            edgecolor="white", linewidth=0.5,
        )
        if n_models <= 4:
            for bar, v in zip(bars, values):
                if v and not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.01, f"{v:.1f}",
                            ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([_short(l) for l in labels], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncol=max(1, (n_series + 2) // 3))


def _ls(m: dict) -> dict:
    return m["manifests"]["librispeech"]


def _figsize(n_models: int) -> tuple[float, float]:
    return (max(8, n_models * 1.6), 5)


# ── NeMo-vs-ours comparison plots (one bar per dtype) ────────────────────────────

def plot_rtfx(models, dtypes, out_dir: Path):
    labels = [m["model"] for m in models]
    groups = {"nemo": [_ls(m)["nemo"]["rtfx"] for m in models]}
    for d in dtypes:
        groups[d] = [_ls(m)["ours"][d]["rtfx"] for m in models]

    fig, ax = plt.subplots(figsize=_figsize(len(labels)))
    _grouped_bar(ax, labels, groups,
                 ylabel="RTFx  (audio_sec / proc_sec) — higher = faster",
                 title="Real-Time Factor × — NeMo vs parakeet.cpp  (LibriSpeech test-clean)")
    ax.axhline(1.0, color="red", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(out_dir / "rtfx.png"); plt.close(fig)
    print(f"  wrote rtfx.png  (models={len(labels)}, dtypes={dtypes})")


def plot_speedup(models, dtypes, out_dir: Path):
    labels = [m["model"] for m in models]
    groups = {}
    for d in dtypes:
        groups[d] = [_ls(m)["ours"][d]["rtfx"] / _ls(m)["nemo"]["rtfx"] for m in models]

    fig, ax = plt.subplots(figsize=_figsize(len(labels)))
    _grouped_bar(ax, labels, groups,
                 ylabel="Speedup  (ours RTFx / NeMo RTFx)",
                 title="Speedup vs NeMo  — >1 = parakeet.cpp is faster")
    ax.axhline(1.0, color="red", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.set_ylim(bottom=0)
    ax.text(0.01, 0.97, "<1 = slower than NeMo;  >1 = faster than NeMo",
            transform=ax.transAxes, fontsize=8, va="top", color="dimgray")
    fig.tight_layout(); fig.savefig(out_dir / "speedup.png"); plt.close(fig)
    print("  wrote speedup.png")


def plot_wer(models, dtypes, out_dir: Path):
    labels = [m["model"] for m in models]
    groups = {"nemo": [_ls(m)["nemo"]["wer_vs_truth"] * 100 for m in models]}
    for d in dtypes:
        groups[d] = [_ls(m)["ours"][d]["wer_vs_truth"] * 100 for m in models]

    fig, ax = plt.subplots(figsize=_figsize(len(labels)))
    _grouped_bar(ax, labels, groups,
                 ylabel="WER vs ground truth  (%)",
                 title="WER vs LibriSpeech test-clean ground truth  (lower = better)")
    ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(out_dir / "wer.png"); plt.close(fig)
    print("  wrote wer.png")


def plot_agreement(models, dtypes, out_dir: Path):
    labels = [m["model"] for m in models]
    groups = {}
    allv = []
    for d in dtypes:
        vals = [_ls(m)["ours"][d]["agreement_wer_vs_nemo"] * 100 for m in models]
        groups[d] = vals
        allv += [v for v in vals if v is not None]

    fig, ax = plt.subplots(figsize=_figsize(len(labels)))
    _grouped_bar(ax, labels, groups,
                 ylabel="Agreement WER (ours vs NeMo)  (%)",
                 title="Transcript Agreement vs NeMo  (lower = closer; quant adds divergence)")
    ax.set_ylim(bottom=0)
    ax.text(0.01, 0.97, "f32 ≈ 0% (byte-identical to NeMo); lower quants diverge more",
            transform=ax.transAxes, fontsize=8, va="top", color="dimgray")
    fig.tight_layout(); fig.savefig(out_dir / "agreement.png"); plt.close(fig)
    print("  wrote agreement.png")


def plot_memory(models, dtypes, out_dir: Path):
    labels = [m["model"] for m in models]
    groups = {"nemo": [_ls(m)["nemo"]["peak_rss_mb"] for m in models]}
    for d in dtypes:
        groups[d] = [_ls(m)["ours"][d]["peak_rss_mb"] for m in models]

    fig, ax = plt.subplots(figsize=_figsize(len(labels)))
    _grouped_bar(ax, labels, groups,
                 ylabel="Peak RSS  (MB)",
                 title="Peak Memory — NeMo vs parakeet.cpp  (LibriSpeech run)")
    ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(out_dir / "memory.png"); plt.close(fig)
    print("  wrote memory.png")


def plot_size(models, dtypes, out_dir: Path):
    labels = [m["model"] for m in models]
    groups = {d: [m["gguf_size_mb"][d] for m in models] for d in dtypes}

    fig, ax = plt.subplots(figsize=_figsize(len(labels)))
    _grouped_bar(ax, labels, groups,
                 ylabel="GGUF file size  (MB)",
                 title="GGUF Model Size by dtype  (lower = smaller on disk)")
    ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(out_dir / "size.png"); plt.close(fig)
    print("  wrote size.png")


def plot_latency_vs_len(models, dtypes, out_dir: Path):
    """Scatter proc_s vs audio_sec, pooled. Show NeMo, ours-f32, and lowest quant."""
    lowest = dtypes[-1] if dtypes else "f32"
    series = {"nemo": ([], []), "f32": ([], []), lowest: ([], [])}

    for m in models:
        for _mname, mdata in m["manifests"].items():
            for f in mdata["nemo"]["files"]:
                series["nemo"][0].append(f["audio_sec"]); series["nemo"][1].append(f["proc_s"])
            for key in ("f32", lowest):
                if key in mdata["ours"]:
                    for f in mdata["ours"][key]["files"]:
                        proc_s = f.get("proc_s") or f.get("proc_ms", 0) / 1000.0
                        series[key][0].append(f["audio_sec"]); series[key][1].append(proc_s)

    fig, ax = plt.subplots(figsize=(8, 5))
    kw = dict(alpha=0.5, s=18, edgecolors="none")
    allx = []
    for key, (xs, ys) in series.items():
        if xs:
            ax.scatter(xs, ys, color=COLORS.get(key, "#888"),
                       label=("NeMo" if key == "nemo" else f"ours {key}"), **kw)
            allx += xs
    if allx:
        xs = np.linspace(0, max(allx) * 1.05, 100)
        ax.plot(xs, xs, "r--", linewidth=1.2, alpha=0.7, label="real-time (1×)")

    ax.set_xlabel("Audio length  (s)")
    ax.set_ylabel("Processing time  (s)")
    ax.set_title("Latency vs Audio Length  (per file, all models & clips)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(out_dir / "latency_vs_len.png"); plt.close(fig)
    print("  wrote latency_vs_len.png")


# ── Quantization tradeoff plots ──────────────────────────────────────────────────

def plot_quant_tradeoff(models, dtypes, out_dir: Path):
    """Scatter: GGUF size (MB) vs agreement-WER (% vs NeMo), per (model, dtype).
    Shows the compression/accuracy frontier; colour = dtype."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for d in dtypes:
        xs, ys = [], []
        for m in models:
            size = m["gguf_size_mb"].get(d)
            agr = _ls(m)["ours"][d].get("agreement_wer_vs_nemo")
            if size is not None and agr is not None:
                xs.append(size); ys.append(agr * 100)
        if xs:
            ax.scatter(xs, ys, color=COLORS.get(d, "#888"), label=d, s=42,
                       edgecolors="white", linewidth=0.5, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("GGUF size  (MB, log scale)")
    ax.set_ylabel("Agreement WER vs NeMo  (%)  — accuracy cost")
    ax.set_title("Quantization Tradeoff — size vs accuracy  (each point = one model/dtype)")
    ax.legend(title="dtype", fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(out_dir / "quant_tradeoff.png"); plt.close(fig)
    print("  wrote quant_tradeoff.png")


def plot_quant_accuracy(models, dtypes, out_dir: Path):
    """Agreement-WER vs dtype, one line per model (degradation as precision drops)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(dtypes))
    cmap = plt.cm.tab10
    for i, m in enumerate(models):
        ys = [_ls(m)["ours"][d].get("agreement_wer_vs_nemo", float("nan")) * 100
              for d in dtypes]
        ax.plot(x, ys, "o-", color=cmap(i % 10), linewidth=1.6,
                markersize=4, label=_short(m["model"]))
    ax.set_xticks(x); ax.set_xticklabels(dtypes)
    ax.set_xlabel("dtype  (higher precision → lower)")
    ax.set_ylabel("Agreement WER vs NeMo  (%)")
    ax.set_title("Accuracy vs Quantization  (per model; f32 ≈ 0, lower quants diverge)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(out_dir / "quant_accuracy.png"); plt.close(fig)
    print("  wrote quant_accuracy.png")


def plot_threads(threads_data, out_dir: Path):
    threads = [row["threads"] for row in threads_data]
    nemo_rtfx = [row["nemo"]["rtfx"] for row in threads_data]
    ours_rtfx = [row["ours"]["rtfx"] for row in threads_data]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(threads, nemo_rtfx, "o-", color=COLORS["nemo"], label="NeMo (PyTorch CPU)", linewidth=2)
    ax.plot(threads, ours_rtfx, "s-", color=COLORS["f32"], label="ours", linewidth=2)
    ax.set_xlabel("Thread count"); ax.set_ylabel("RTFx")
    ax.set_title("Thread Scaling — RTFx vs Threads")
    ax.axhline(1.0, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.legend(fontsize=9); ax.set_xlim(left=0); ax.set_ylim(bottom=0); ax.set_xticks(threads)
    fig.tight_layout(); fig.savefig(out_dir / "threads.png"); plt.close(fig)
    print("  wrote threads.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate benchmark plots from parakeet.cpp results.")
    ap.add_argument("--results", default="benchmarks/results")
    ap.add_argument("--out", default="benchmarks/plots")
    args = ap.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = load_results(results_dir)
    if not models:
        print(f"ERROR: no model JSON files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    dtypes = dtypes_present(models)
    print(f"Loaded {len(models)} model(s); dtypes={dtypes}")
    print(f"Output → {out_dir}/")

    plot_rtfx(models, dtypes, out_dir)
    plot_speedup(models, dtypes, out_dir)
    plot_wer(models, dtypes, out_dir)
    plot_agreement(models, dtypes, out_dir)
    plot_memory(models, dtypes, out_dir)
    plot_size(models, dtypes, out_dir)
    plot_latency_vs_len(models, dtypes, out_dir)
    if len(dtypes) > 1:
        plot_quant_tradeoff(models, dtypes, out_dir)
        plot_quant_accuracy(models, dtypes, out_dir)

    threads_data = load_threads(results_dir)
    if threads_data:
        plot_threads(threads_data, out_dir)
    else:
        print("  skipped threads.png  (no threads.json found)")

    print("Done.")


if __name__ == "__main__":
    main()
