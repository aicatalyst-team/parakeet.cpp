#!/usr/bin/env python3
"""
bench_data.py  — assemble audio + manifests for the parakeet.cpp benchmark.

Usage:
    python scripts/bench_data.py --out benchmarks/audio --n 100
"""

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_wav_16k_mono(audio: np.ndarray, sr: int, out_path: Path) -> None:
    """Resample to 16 kHz, convert to mono, write 16-bit PCM WAV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Mix down to mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    # Resample if needed
    if sr != 16_000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
    # Clip and convert to int16
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    sf.write(str(out_path), audio_int16, 16_000, subtype="PCM_16")


# ---------------------------------------------------------------------------
# LibriSpeech
# ---------------------------------------------------------------------------

OPENSLR_TEST_CLEAN_URL = (
    "https://www.openslr.org/resources/12/test-clean.tar.gz"
)


def _librispeech_via_datasets(n: int, audio_dir: Path) -> list[tuple[Path, str]]:
    """Use HuggingFace datasets to get the first N utterances."""
    from datasets import load_dataset  # type: ignore

    print("[librispeech] Loading via datasets (HuggingFace) …")
    ds = load_dataset(
        "librispeech_asr", "clean", split="test"
    )
    ds = ds.select(range(min(n, len(ds))))

    results: list[tuple[Path, str]] = []
    for item in ds:
        utt_id = item["id"]  # e.g. "1089-134686-0000"
        ref = item["text"].upper()  # keep uppercase for consistency
        audio_data = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]

        out_path = audio_dir / f"{utt_id}.wav"
        write_wav_16k_mono(np.array(audio_data, dtype=np.float32), sr, out_path)
        results.append((out_path, ref))
        if len(results) % 10 == 0:
            print(f"  {len(results)}/{n} written …")

    return results


def _download_with_progress(url: str, dest: Path) -> None:
    print(f"[librispeech] Downloading {url} → {dest} …")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = downloaded * 100 / total_size
            mb = downloaded / 1_048_576
            print(f"\r  {pct:.1f}% ({mb:.0f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_report)
    print()


def _librispeech_via_openslr(n: int, audio_dir: Path, cache_dir: Path) -> list[tuple[Path, str]]:
    """Download the OpenSLR test-clean tar, extract, parse transcripts."""
    tar_path = cache_dir / "test-clean.tar.gz"
    extract_dir = cache_dir / "LibriSpeech"

    if not tar_path.exists():
        _download_with_progress(OPENSLR_TEST_CLEAN_URL, tar_path)
    else:
        print(f"[librispeech] Reusing cached tar {tar_path}")

    if not extract_dir.exists():
        print(f"[librispeech] Extracting to {extract_dir} …")
        with tarfile.open(str(tar_path), "r:gz") as tf:
            tf.extractall(str(cache_dir))
    else:
        print(f"[librispeech] Reusing extracted dir {extract_dir}")

    # Collect utterances: each *.trans.txt has lines: "<utt_id> UPPERCASE TEXT"
    trans_files = sorted(extract_dir.rglob("*.trans.txt"))
    utterances: list[tuple[str, str, Path]] = []  # (utt_id, ref, flac_path)

    for tf_path in trans_files:
        parent = tf_path.parent
        with open(tf_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                utt_id, _, ref = line.partition(" ")
                flac_path = parent / f"{utt_id}.flac"
                if flac_path.exists():
                    utterances.append((utt_id, ref, flac_path))
                if len(utterances) >= n:
                    break
        if len(utterances) >= n:
            break

    print(f"[librispeech] Found {len(utterances)} utterances, writing wavs …")
    results: list[tuple[Path, str]] = []
    for i, (utt_id, ref, flac_path) in enumerate(utterances[:n]):
        audio, sr = sf.read(str(flac_path), dtype="float32")
        out_path = audio_dir / f"{utt_id}.wav"
        write_wav_16k_mono(audio, sr, out_path)
        results.append((out_path, ref))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n} written …")

    return results


def prepare_librispeech(
    n: int,
    audio_dir: Path,
    manifest_path: Path,
    cache_dir: Path,
) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the OpenSLR tar (346 MB, self-contained, fast for a subset). Using
    # HF `datasets` would download the entire test-clean set just to take N
    # utterances, so it's only a fallback.
    try:
        results = _librispeech_via_openslr(n, audio_dir, cache_dir)
        method = "OpenSLR tar"
    except Exception as exc:
        print(f"[librispeech] OpenSLR tar failed ({exc}); falling back to HF datasets …")
        results = _librispeech_via_datasets(n, audio_dir)
        method = "HuggingFace datasets"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as fh:
        for wav_path, ref in results:
            fh.write(f"{wav_path.resolve()}\t{ref}\n")

    total_s = sum(
        sf.info(str(p)).frames / sf.info(str(p)).samplerate for p, _ in results
    )
    print(
        f"[librispeech] Done via {method}: {len(results)} utterances, "
        f"{total_s / 60:.1f} min audio → {manifest_path}"
    )


# ---------------------------------------------------------------------------
# Diverse clips
# ---------------------------------------------------------------------------

# Source-path, output stem, reference transcript (empty = none known)
DIVERSE_CLIPS = [
    (
        "jfk.wav",
        "jfk.wav",
        "And so my fellow Americans, ask not what your country can do for you, "
        "ask what you can do for your country.",
    ),
    (
        "I_have_a_dream.ogg",
        "i_have_a_dream.wav",
        "",  # sanity-only; reference transcript omitted for brevity
    ),
    (
        "antirez_speaking_italian_short.ogg",
        "antirez_italian.wav",
        "",  # Italian clip; exercises multilingual tdt-0.6b-v3; ref omitted
    ),
    (
        "test_speech.wav",
        "test_speech.wav",
        "",  # sanity-only
    ),
]

SAMPLES_DIR = Path("/home/mudler/_git/voxtral.c/samples")


def prepare_diverse(
    audio_dir: Path,
    manifest_path: Path,
) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[Path, str]] = []
    for src_name, dst_name, ref in DIVERSE_CLIPS:
        src_path = SAMPLES_DIR / src_name
        if not src_path.exists():
            print(f"[diverse] WARNING: source not found: {src_path} — skipping")
            continue

        dst_path = audio_dir / dst_name
        audio, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
        write_wav_16k_mono(audio, sr, dst_path)
        info = sf.info(str(dst_path))
        dur = info.frames / info.samplerate
        print(
            f"[diverse] {dst_name}: {dur:.1f}s @ {info.samplerate} Hz "
            f"{'(ref supplied)' if ref else '(no ref)'}"
        )
        results.append((dst_path, ref))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Header comment so it's self-documenting
    with open(manifest_path, "w") as fh:
        fh.write(
            "# wav_path\tref_text  "
            "# note: antirez_italian.wav is Italian — exercises multilingual tdt-0.6b-v3\n"
        )
        for wav_path, ref in results:
            fh.write(f"{wav_path.resolve()}\t{ref}\n")

    print(f"[diverse] Done: {len(results)} clips → {manifest_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble benchmark audio + manifests for parakeet.cpp."
    )
    parser.add_argument(
        "--out",
        default="benchmarks/audio",
        help="Root directory for output audio (default: benchmarks/audio)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of LibriSpeech test-clean utterances (default: 100)",
    )
    parser.add_argument(
        "--cache",
        default=".cache/bench_data",
        help="Cache dir for downloaded tars (default: .cache/bench_data)",
    )
    args = parser.parse_args()

    out_root = Path(args.out).resolve()
    cache_dir = Path(args.cache).resolve()

    libri_audio_dir = out_root / "librispeech"
    libri_manifest = out_root.parent / "librispeech_manifest.tsv"

    diverse_audio_dir = out_root / "diverse"
    diverse_manifest = out_root.parent / "diverse_manifest.tsv"

    print(f"=== Benchmark data prep: n={args.n}, out={out_root} ===\n")

    prepare_librispeech(args.n, libri_audio_dir, libri_manifest, cache_dir)
    print()
    prepare_diverse(diverse_audio_dir, diverse_manifest)

    print("\n=== All done ===")
    print(f"  LibriSpeech manifest : {libri_manifest}")
    print(f"  Diverse manifest     : {diverse_manifest}")


if __name__ == "__main__":
    main()
