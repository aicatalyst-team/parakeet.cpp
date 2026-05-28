#!/usr/bin/env python3
"""Convert a NeMo Parakeet checkpoint to GGUF (f32 / f16 / q8_0).

The GGUF is fully metadata-driven: all config lives in KV, and tensor names are
kept **verbatim** from the NeMo ``state_dict`` (no renaming) so the C++ port is a
1:1 mapping. The two featurizer buffers (``preprocessor.featurizer.fb`` and
``preprocessor.featurizer.window``) are lifted directly from the checkpoint so the
C++ side never re-derives the mel filterbank with librosa.

Quantization (``--dtype f16|q8_0``) is applied **only** to the large linear
weights that the C++ engine consumes directly via ``ggml_mul_mat`` (the encoder
FFN + attention projections, the subsampling output projection, and the joint
enc/pred projections). ggml dequantizes those on the fly inside the compute
graph. Everything the hand-rolled C++ reads as raw F32 (the mel filterbank /
window, the LSTM prediction net, the joint output projection, batch_norm running
stats, conv kernels, embeddings, all norms and biases, pos_bias) stays F32 -- see
``should_quantize`` and ``docs/quantization.md``.

See ``docs/conversion.md`` for the full schema.
"""
import argparse
import pathlib
import re
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np

try:
    import gguf
except ImportError as e:  # pragma: no cover - env guard
    print(f"converter: missing dependency 'gguf': {e}", file=sys.stderr)
    print("PARAKEET_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)

try:
    from nemo.collections.asr.models import ASRModel
except ImportError as e:  # pragma: no cover - env guard
    print(f"converter: missing dependency 'nemo_toolkit[asr]': {e}", file=sys.stderr)
    print("PARAKEET_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)


def _get(cfg, key, default=None):
    """Read ``key`` from an OmegaConf node or plain object, tolerating both."""
    try:
        return cfg[key]
    except Exception:
        return getattr(cfg, key, default)


def detect_arch(m):
    """Map a NeMo model to one of ctc/rnnt/tdt/hybrid_rnnt_ctc/hybrid_tdt_ctc."""
    cfg = m.cfg
    if _get(cfg, "aux_ctc") is not None:
        loss = _get(_get(cfg, "loss", {}) or {}, "loss_name", "")
        durs = _get(_get(cfg, "decoding", {}) or {}, "durations")
        return "hybrid_tdt_ctc" if (loss == "tdt" or durs) else "hybrid_rnnt_ctc"
    if _get(cfg, "joint") is not None:
        durs = _get(_get(cfg, "decoding", {}) or {}, "durations")
        nxo = _get(_get(cfg, "joint", {}) or {}, "num_extra_outputs", 0)
        return "tdt" if (durs or (nxo and nxo > 0)) else "rnnt"
    return "ctc"


# ---------------------------------------------------------------------------
# Quantization policy.
#
# The C++ engine only tolerates a non-F32 weight when that weight is fed
# *directly* into ``ggml_mul_mat`` (ggml dequantizes f16/q8_0 src0 on the fly).
# Every other weight is read by hand-rolled C++ as a raw ``float*`` (mel
# filterbank/window, LSTM prediction net, joint output projection, batch_norm
# stats, embeddings), or is reshaped/transposed before the matmul in a way that
# does not survive block-quantized storage (the CTC head is stored [1, d, V] and
# squeezed in-graph; conv pointwise weights are reshaped from [1, in, out]).
# Those MUST stay F32 or the engine produces garbage.
#
# Allowlist of weights that are passed verbatim to ggml_mul_mat (see the audit in
# docs/quantization.md). Names are matched after the verbatim NeMo state_dict
# name; "N" is any layer index.
_QUANTIZABLE_PATTERNS = [
    # Conformer feed-forward modules: linear1 (d->ff) and linear2 (ff->d).
    r"^encoder\.layers\.\d+\.feed_forward[12]\.linear[12]\.weight$",
    # Conformer self-attention projections q/k/v/out/pos.
    r"^encoder\.layers\.\d+\.self_attn\.linear_(q|k|v|out|pos)\.weight$",
    # Subsampling output projection (Linear C*F' -> d_model), fed straight to
    # ggml_mul_mat in subsampling.cpp with no reshape.
    r"^encoder\.pre_encode\.out\.weight$",
    # Joint enc/pred projections (ggml_mul_mat in joint.cpp). NOTE: the joint
    # OUTPUT projection joint.joint_net.2.weight is read as a raw float* and
    # stays F32 -- it is intentionally NOT in this allowlist.
    r"^joint\.enc\.weight$",
    r"^joint\.pred\.weight$",
]
_QUANTIZABLE_RE = [re.compile(p) for p in _QUANTIZABLE_PATTERNS]


def should_quantize(name, shape, dtype):
    """Return the ggml quantization type for ``name`` given the requested dtype.

    ``shape`` is the ggml ``ne`` (reverse of the torch shape), so ``shape[0]`` is
    the contraction / leading dimension -- the axis q8_0 blocks along (block
    size 32). Returns ``None`` (keep F32) unless the tensor is on the linear-
    weight allowlist, is at least 2-D with both dims >= 32, and (for q8_0) has a
    leading dimension divisible by the 32-element block size.
    """
    if dtype == "f32":
        return None
    if not any(rx.match(name) for rx in _QUANTIZABLE_RE):
        return None
    if len(shape) < 2 or shape[0] < 32 or shape[1] < 32:
        return None
    if dtype == "f16":
        return gguf.GGMLQuantizationType.F16
    if dtype == "q8_0":
        if shape[0] % 32 != 0:
            return None  # leading dim not block-aligned -> keep F32
        return gguf.GGMLQuantizationType.Q8_0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local .nemo")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--dtype",
        choices=["f32", "f16", "q8_0"],
        default="f32",
        help="quantization for allowlisted linear weights (everything else f32)",
    )
    args = ap.parse_args()

    is_local = pathlib.Path(args.model).exists()
    try:
        if is_local:
            m = ASRModel.restore_from(args.model, map_location="cpu")
        else:
            m = ASRModel.from_pretrained(args.model, map_location="cpu")
    except Exception as e:  # pragma: no cover - network/cache guard
        print(f"PARAKEET_MODEL_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(2)
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
    w.add_string("parakeet.encoder.conv_norm_type",
                 str(_get(enc, "conv_norm_type", "batch_norm")))
    w.add_uint32("parakeet.encoder.subsampling_factor",
                 int(_get(enc, "subsampling_factor")))
    w.add_uint32("parakeet.encoder.subsampling_conv_channels",
                 int(_get(enc, "subsampling_conv_channels")))
    w.add_bool("parakeet.encoder.xscaling", bool(_get(enc, "xscaling", True)))
    w.add_uint32("parakeet.encoder.pos_emb_max_len",
                 int(_get(enc, "pos_emb_max_len", 5000)))

    # preprocessor (effective values off the featurizer object)
    w.add_uint32("parakeet.preprocessor.sample_rate",
                 int(getattr(feat, "sample_rate", 16000)))
    w.add_uint32("parakeet.preprocessor.n_mels", int(getattr(feat, "nfilt")))
    w.add_uint32("parakeet.preprocessor.n_fft", int(getattr(feat, "n_fft")))
    w.add_uint32("parakeet.preprocessor.win_length", int(getattr(feat, "win_length")))
    w.add_uint32("parakeet.preprocessor.hop_length", int(getattr(feat, "hop_length")))
    pre = getattr(feat, "preemph", None)
    w.add_float32("parakeet.preprocessor.preemph", float(pre) if pre is not None else 0.0)
    w.add_float32("parakeet.preprocessor.mag_power",
                  float(getattr(feat, "mag_power", 2.0)))
    w.add_string("parakeet.preprocessor.normalize",
                 str(getattr(feat, "normalize", "per_feature")))
    lzg = getattr(feat, "log_zero_guard_value", None)
    w.add_float32("parakeet.preprocessor.log_zero_guard",
                  float(lzg) if isinstance(lzg, (int, float)) else 2 ** -24)

    # vocab / tokenizer
    vocab = int(m.tokenizer.vocab_size)
    w.add_uint32("parakeet.vocab_size", vocab)
    w.add_uint32("parakeet.blank_id", vocab)  # blank always == vocab_size
    pieces = [m.tokenizer.ids_to_tokens([i])[0] for i in range(vocab)]
    w.add_array("parakeet.tokenizer.pieces", [str(p) for p in pieces])

    # transducer config
    if arch in ("rnnt", "tdt", "hybrid_rnnt_ctc", "hybrid_tdt_ctc"):
        prednet = _get(cfg.decoder, "prednet", {}) or {}
        w.add_uint32("parakeet.decoder.pred_hidden", int(_get(prednet, "pred_hidden")))
        w.add_uint32("parakeet.decoder.pred_rnn_layers",
                     int(_get(prednet, "pred_rnn_layers", 1)))
        jn = _get(cfg.joint, "jointnet", {}) or {}
        w.add_uint32("parakeet.joint.joint_hidden", int(_get(jn, "joint_hidden")))
        w.add_string("parakeet.joint.activation", str(_get(jn, "activation", "relu")))
    if arch in ("tdt", "hybrid_tdt_ctc"):
        durs = (_get(_get(cfg, "decoding", {}) or {}, "durations")
                or _get(_get(cfg, "model_defaults", {}) or {}, "tdt_durations"))
        if not durs:
            raise ValueError(
                f"arch={arch} requires TDT durations but none found in "
                "cfg.decoding.durations or cfg.model_defaults.tdt_durations"
            )
        w.add_array("parakeet.tdt.durations", [int(d) for d in durs])

    # tensors: verbatim names. Allowlisted linear weights are quantized per
    # --dtype (ggml dequantizes them on the fly inside ggml_mul_mat); everything
    # else stays f32. Include featurizer buffers explicitly.
    sd = m.state_dict()
    written = 0
    quantized = 0
    keep_buffers = {"preprocessor.featurizer.fb", "preprocessor.featurizer.window"}
    for name, t in sd.items():
        if name.startswith("preprocessor.") and name not in keep_buffers:
            continue  # skip preprocessor internals except fb/window
        if not hasattr(t, "detach"):
            continue
        arr = t.detach().cpu().float().numpy()
        if arr.ndim == 0:
            continue  # skip scalar bookkeeping (e.g. num_batches_tracked)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        # ggml ne is the reverse of the numpy/torch shape; ne[0] is the leading
        # (contraction) axis q8_0 blocks along.
        ggml_ne = list(arr.shape[::-1])
        qtype = should_quantize(name, ggml_ne, args.dtype)
        if qtype is None:
            w.add_tensor(name, arr)
        else:
            raw = gguf.quantize(arr, qtype)
            # gguf expects raw_shape to be the *byte* shape of the quantized
            # buffer; it derives the element shape from it via raw_dtype.
            w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=qtype)
            quantized += 1
        written += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {args.output}: arch={arch} vocab={vocab} tensors={written} "
        f"dtype={args.dtype} quantized={quantized}"
    )


if __name__ == "__main__":
    main()
