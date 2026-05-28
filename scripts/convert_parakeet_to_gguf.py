#!/usr/bin/env python3
"""Convert a NeMo Parakeet checkpoint to GGUF (f32).

The GGUF is fully metadata-driven: all config lives in KV, and tensor names are
kept **verbatim** from the NeMo ``state_dict`` (no renaming) so the C++ port is a
1:1 mapping. The two featurizer buffers (``preprocessor.featurizer.fb`` and
``preprocessor.featurizer.window``) are lifted directly from the checkpoint so the
C++ side never re-derives the mel filterbank with librosa.

See ``docs/conversion.md`` for the full schema.
"""
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local .nemo")
    ap.add_argument("--output", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="error on tensors that look unexpected (reserved)")
    args = ap.parse_args()

    is_hf_id = "/" in args.model and not args.model.endswith(".nemo")
    if is_hf_id:
        m = ASRModel.from_pretrained(args.model, map_location="cpu")
    else:
        m = ASRModel.restore_from(args.model, map_location="cpu")
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
            continue  # skip scalar bookkeeping (e.g. num_batches_tracked)
        w.add_tensor(name, np.ascontiguousarray(arr))
        written += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}: arch={arch} vocab={vocab} tensors={written}")


if __name__ == "__main__":
    main()
