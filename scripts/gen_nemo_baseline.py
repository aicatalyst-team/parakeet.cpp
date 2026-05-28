#!/usr/bin/env python3
"""Dump NeMo Parakeet intermediate tensors to ``baseline.gguf`` for C++ parity.

Phase 1 validates each C++ stage (mel -> subsampling -> conformer layers ->
encoder -> CTC) by diffing against the exact tensors this script captures from
the reference NeMo model. Correctness and determinism are therefore paramount:

* ``dither`` is forced to 0.0 so the mel spectrogram is reproducible (the C++
  side skips dither too).
* The forward runs under ``torch.no_grad()`` with the model in ``eval()``.
* The CTC logits are produced by running the encoder + CTC head **explicitly**
  rather than via ``transcribe``. ``parakeet-tdt_ctc-110m`` is a hybrid
  (TDT + CTC) model whose default ``transcribe`` uses the RNNT/TDT head and
  never calls ``ctc_decoder`` — a forward hook on it would never fire. Driving
  ``m.ctc_decoder`` directly guarantees the CTC logits are real.

Stored tensors (squeezed; f32 except the int32 ids). Axis order documented in
``docs/conversion.md`` ("Baseline intermediates"):

* ``mel``             ``[n_mels, T]``      output of ``m.preprocessor``
* ``subsampling_out`` ``[T', d_model]``    output of ``m.encoder.pre_encode``
* ``pos_emb``         ``[2*T'-1, d_model]`` rel pos enc (``m.encoder.pos_enc`` out[1])
* ``enc_pre_layers``  ``[T', d_model]``    tensor fed INTO conformer ``layers[0]``
* ``l0_attn_in``      ``[T', d_model]``    self_attn input = ``norm_self_att(residual)``
                                           (NOTE: residual = enc_pre_layers + 0.5*FFN1(..),
                                           so this is NOT norm_self_att(enc_pre_layers))
* ``l0_attn_out``     ``[T', d_model]``    output of ``layers[0].self_attn`` (RelPosMHA)
* ``l0_conv_out``     ``[T', d_model]``    output of ``layers[0].conv`` (ConformerConvolution)
* ``enc_layer_0``     ``[T', d_model]``    output of conformer ``layers[0]``
* ``enc_layer_mid``   ``[T', d_model]``    output of conformer ``layers[n//2]``
* ``enc_layer_last``  ``[T', d_model]``    output of conformer ``layers[n-1]``
* ``encoder_out``     ``[d_model, T']``    output of ``m.encoder`` (transposed)
* ``ctc_logits``      ``[T', V+1]``        log-probs from ``m.ctc_decoder``
* ``ctc_argmax_ids``  ``[T']`` int32       argmax over the vocab axis of logits

Transducer-core ground truth (Phase 2; ``m.decoder`` = ``RNNTDecoder``
prediction net, ``m.joint`` = ``RNNTJoint`` TDT joint):

* ``pred_input_ids``  ``[U]`` int32        fixed label sequence fed to the
                                           prediction net (``[120, 7, 300, 42]``,
                                           all non-blank; blank=1024).
* ``pred_out``        ``[U+1, pred_hidden=640]``  output ``g`` of
                                           ``m.decoder.predict(y, add_sos=True)``,
                                           squeezed. **add_sos=True** so a zero
                                           "start-of-sequence" embedding step is
                                           PREPENDED: with U=4 input ids the
                                           output length is U+1 = 5. The SOS step
                                           uses the zero embedding row
                                           (``embed.weight[1024]`` is all-zero,
                                           ``padding_idx=blank=1024``). The C++
                                           prediction net MUST also prepend the
                                           SOS step to match.
* ``joint_out``       ``[N, U+1, 1030]``   RAW logits from
                                           ``m.joint.joint(enc_slice, pred_out)``
                                           with N = ``joint_enc_frames`` = 4
                                           leading encoder frames. **RAW logits,
                                           NOT log_softmax**: ``m.joint.joint`` on
                                           CPU defaults to applying
                                           ``log_softmax`` over all 1030 entries
                                           (``log_softmax`` cfg is ``None``), but
                                           the TDT greedy decoder consumes raw
                                           logits and splits them — token logits
                                           ``[:1025]`` (1024 vocab + blank) and
                                           duration logits ``[1025:]`` (the 5 TDT
                                           durations [0,1,2,3,4]) get SEPARATE
                                           log_softmaxes. So we force
                                           ``m.joint.log_softmax = False`` to dump
                                           the raw output that the C++ joint
                                           (plain ReLU + linear) produces.
* ``joint_enc_frames`` ``[1]`` int32       N = number of leading encoder frames
                                           used for ``joint_out`` (= 4). The C++
                                           joint test feeds ``encoder_out[:, :N]``.

TDT greedy ground truth (Phase 3; ``m`` with the transducer/TDT head selected):

* ``tdt_token_ids``    ``[L]`` int32        NeMo's TDT-head greedy decoded
                                            token-id sequence (``hyp.y_sequence``)
                                            for the clip. Captured by selecting
                                            the transducer head
                                            (``change_decoding_strategy(decoder_type='rnnt')``
                                            — for this hybrid the transducer head
                                            IS the TDT decoder) and running
                                            ``m.transcribe([audio])``. May be
                                            length-0 for a silent / tone clip.

String KVs:

* ``baseline.detok_text``  detokenized text for the ``detok_ids`` fixture
* ``baseline.ctc_text``    authoritative NeMo CTC greedy transcript of the clip
* ``baseline.tdt_text``    authoritative NeMo TDT-head greedy transcript of the clip

Exit codes (ctest convention): 0 = ok, 2 = deps/model unavailable, 1 = fail.
"""
import argparse
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np

try:
    import gguf
except ImportError as e:  # pragma: no cover - env guard
    print(f"baseline: missing dependency 'gguf': {e}", file=sys.stderr)
    print("PARAKEET_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)

try:
    import torch
    import soundfile as sf
    from nemo.collections.asr.models import ASRModel
except ImportError as e:  # pragma: no cover - env guard
    print(f"baseline: missing dependency: {e}", file=sys.stderr)
    print("PARAKEET_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)


def _squeeze(arr):
    """np.squeeze but never collapse to a 0-d scalar."""
    out = np.squeeze(np.asarray(arr))
    if out.ndim == 0:
        out = out.reshape(1)
    return np.ascontiguousarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet-tdt_ctc-110m",
                    help="HF id or local .nemo")
    ap.add_argument("--audio", required=True, help="16k mono wav clip")
    ap.add_argument("--output", required=True)
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
    # Determinism: zero the spectrogram dither so the mel is reproducible.
    m.preprocessor.featurizer.dither = 0.0

    # Per-layer / module captures via forward hooks. The preprocessor and
    # encoder return (tensor, length) tuples; conformer layers return a bare
    # tensor (no cache) — handle both.
    cap = {}

    def save(name):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor):
                cap[name] = t.detach().cpu().float().numpy()
        return fn

    def save_pos_emb(name):
        # RelPositionalEncoding.forward returns (dropout(x), pos_emb); we want the
        # SECOND element (the [1, 2*T'-1, d_model] relative positional encoding),
        # NOT element 0 (the scaled input embeddings).
        def fn(mod, inp, out):
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                raise RuntimeError(
                    f"baseline: expected pos_enc to return a (x, pos_emb) tuple, "
                    f"got {type(out).__name__} (len={len(out) if hasattr(out, '__len__') else 'n/a'})"
                )
            t = out[1]
            if not isinstance(t, torch.Tensor):
                raise RuntimeError(
                    f"baseline: pos_enc out[1] is not a tensor (got {type(t).__name__})"
                )
            cap[name] = t.detach().cpu().float().numpy()
        return fn

    def save_layer_input(name):
        # Forward PRE-hook on layers[0]: capture the tensor fed INTO the layer
        # (encoder embeddings after subsampling + xscaling + dropout). The encoder
        # calls the layer with x as a KEYWORD arg, so prefer kwargs['x'] and fall
        # back to the first positional arg.
        def fn(mod, args, kwargs):
            t = kwargs.get("x") if "x" in kwargs else (args[0] if args else None)
            if not isinstance(t, torch.Tensor):
                raise RuntimeError(
                    f"baseline: could not capture {name}: layer input was not a "
                    f"tensor (args={len(args)}, kwargs={sorted(kwargs)})"
                )
            cap[name] = t.detach().cpu().float().numpy()
        return fn

    def save_attn_in(name):
        # Forward PRE-hook on layers[0].self_attn: capture the `query` argument,
        # i.e. the normalized attention input (== `key` == `value`). The conformer
        # layer calls self_attn with query/key/value as KEYWORD args.
        def fn(mod, args, kwargs):
            t = kwargs.get("query") if "query" in kwargs else (args[0] if args else None)
            if not isinstance(t, torch.Tensor):
                raise RuntimeError(
                    f"baseline: could not capture {name}: self_attn query was not a "
                    f"tensor (args={len(args)}, kwargs={sorted(kwargs)})"
                )
            cap[name] = t.detach().cpu().float().numpy()
        return fn

    def submodule(path):
        """Resolve a dotted attribute path under ``m``; clear error if missing."""
        obj = m
        for attr in path.split("."):
            if attr.endswith("]") and "[" in attr:
                base, idx = attr[:-1].split("[")
                obj = getattr(obj, base)[int(idx)]
            else:
                obj = getattr(obj, attr, None)
            if obj is None:
                raise RuntimeError(
                    f"baseline: submodule '{path}' not found on the model "
                    f"(failed at '{attr}'). The NeMo module layout may differ."
                )
        return obj

    n = len(m.encoder.layers)
    handles = [
        m.preprocessor.register_forward_hook(save("mel")),
        m.encoder.pre_encode.register_forward_hook(save("subsampling_out")),
        # Fine-grained encoder captures for relpos-attention / conformer parity.
        submodule("encoder.pos_enc").register_forward_hook(save_pos_emb("pos_emb")),
        submodule("encoder.layers[0]").register_forward_pre_hook(
            save_layer_input("enc_pre_layers"), with_kwargs=True
        ),
        # self_attn input = norm_self_att(residual) where residual already includes
        # FFN1. Capturing it directly lets the relpos-attention parity test feed the
        # exact normalized input without re-implementing FFN1 (that's the next task).
        submodule("encoder.layers[0].self_attn").register_forward_pre_hook(
            save_attn_in("l0_attn_in"), with_kwargs=True
        ),
        submodule("encoder.layers[0].self_attn").register_forward_hook(
            save("l0_attn_out")
        ),
        submodule("encoder.layers[0].conv").register_forward_hook(save("l0_conv_out")),
        m.encoder.layers[0].register_forward_hook(save("enc_layer_0")),
        m.encoder.layers[n // 2].register_forward_hook(save("enc_layer_mid")),
        m.encoder.layers[n - 1].register_forward_hook(save("enc_layer_last")),
        m.encoder.register_forward_hook(save("encoder_out")),
    ]

    # Load the clip as float32 mono [1, num_samples].
    wav, sr = sf.read(args.audio, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        print(f"PARAKEET_BASELINE_BAD_AUDIO: expected 16k mono, got sr={sr}",
              file=sys.stderr)
        sys.exit(1)
    wav_t = torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0)  # [1, S]
    len_t = torch.tensor([wav_t.shape[1]], dtype=torch.int64)

    # Explicit forward path (NOT transcribe) so the CTC head actually runs.
    #   preprocessor.forward(input_signal, length) -> (feats[B,n_mels,T], feat_len)
    #   encoder.forward(audio_signal, length)      -> (enc[B,d_model,T'], enc_len)
    #   ctc_decoder.forward(encoder_output)        -> log-probs [B, T', V+1]
    with torch.no_grad():
        feats, feat_len = m.preprocessor(input_signal=wav_t, length=len_t)
        enc, enc_len = m.encoder(audio_signal=feats, length=feat_len)
        ctc_log = m.ctc_decoder(encoder_output=enc)

    for h in handles:
        h.remove()

    cap["ctc_logits"] = ctc_log.detach().cpu().float().numpy()  # [B, T', V+1]

    # ctc_argmax_ids: greedy CTC ids = argmax over the vocab axis (last) of the
    # squeezed [T', V+1] logits. Deterministic end-to-end CTC check for Phase 1.
    logits = _squeeze(cap["ctc_logits"])  # [T', V+1]
    ids = (logits.argmax(-1) if logits.ndim == 2 else logits.argmax(0)).astype(np.int32)

    # ---- Transducer-core ground truth (Phase 2): prediction net + joint ----
    # Done BEFORE change_decoding_strategy() (which only swaps the decoding
    # strategy, not module weights, but we keep the model untouched to be safe).
    #
    # Prediction net: m.decoder.predict(y, state, add_sos, batch_size) returns
    # (g, hidden). With add_sos=True (the predict() default and what the RNNT/TDT
    # decoders use to prime the network), a zero "start-of-sequence" embedding
    # step is PREPENDED, so g is [B, U+1, pred_hidden]. blank=1024 is the
    # padding_idx so embed.weight[1024] is all-zero (== the SOS embedding).
    pred_input_ids = np.array([120, 7, 300, 42], dtype=np.int32)  # all non-blank
    add_sos = True
    y = torch.from_numpy(pred_input_ids.astype(np.int64)).unsqueeze(0)  # [1, U]
    with torch.no_grad():
        g, _hid = m.decoder.predict(y, state=None, add_sos=add_sos, batch_size=None)
    cap["pred_out"] = g.detach().cpu().float().numpy()  # [1, U+1, pred_hidden]

    # Sanity: SOS step (g row 0) uses the zero embedding -> confirm embed[blank]
    # is all-zero so the C++ side can reproduce the SOS step exactly.
    embed_w = m.decoder.prediction["embed"].weight.detach()
    blank_idx = int(getattr(m.decoder, "blank_idx", embed_w.shape[0] - 1))
    sos_embed_is_zero = bool(torch.all(embed_w[blank_idx] == 0).item())
    if not sos_embed_is_zero:  # pragma: no cover - layout guard
        print(
            f"PARAKEET_BASELINE_WARN: embed.weight[{blank_idx}] (SOS/padding) is "
            f"NOT all-zero (absmax={float(embed_w[blank_idx].abs().max())}); the "
            f"C++ SOS step must use this exact embedding row.",
            file=sys.stderr,
        )

    # Joint: m.joint.joint(f, g) with f=[B,T,enc_hidden], g=[B,U,pred_hidden] ->
    # [B, T, U, vocab+1+num_durations]=[B,T,U,1030]. `enc` from the encoder above
    # is [B, d_model, T'] so transpose to [B, T', d_model] first. Slice to the
    # first N=joint_enc_frames encoder frames to keep the dump small.
    #
    # RAW LOGITS: m.joint.joint() on CPU applies log_softmax over all 1030 entries
    # by default (joint.log_softmax cfg is None). But the TDT greedy decoder uses
    # RAW logits and splits them — token logits [:1025] and the 5 duration logits
    # [1025:] get SEPARATE log_softmaxes. The C++ joint emits raw logits, so we
    # force log_softmax=False here to dump the matching raw output.
    joint_enc_frames = 4
    enc_btd = enc.transpose(1, 2)  # [B, T', d_model]
    n_frames = min(joint_enc_frames, enc_btd.shape[1])
    enc_slice = enc_btd[:, :n_frames, :].contiguous()  # [1, N, enc_hidden]
    saved_log_softmax = m.joint.log_softmax
    try:
        m.joint.log_softmax = False  # raw logits, matching the C++ joint
        with torch.no_grad():
            jout = m.joint.joint(enc_slice, g)  # [1, N, U+1, 1030] raw logits
    finally:
        m.joint.log_softmax = saved_log_softmax
    cap["joint_out"] = jout.detach().cpu().float().numpy()  # [1, N, U+1, 1030]
    joint_enc_frames_arr = np.array([n_frames], dtype=np.int32)

    # ---- TDT greedy ground truth (Phase 3) ----
    # Select the transducer head (for this hybrid the transducer head IS the TDT
    # decoder) and run the model's own greedy decode to capture the reference
    # token-id sequence. change_decoding_strategy(decoder_type='rnnt') sets
    # cur_decoder='rnnt'; transcribe() then runs the TDT greedy path. We capture
    # hyp.y_sequence (the emitted token ids) — this is the EXACT sequence the C++
    # pk::tdt_greedy loop must reproduce. For a tone/silent clip the sequence may
    # be empty (length 0), which is fine.
    m.change_decoding_strategy(decoder_type="rnnt")
    assert m.cur_decoder == "rnnt", f"expected RNNT/TDT decoder, got {m.cur_decoder}"
    with torch.no_grad():
        tdt_out = m.transcribe([args.audio], batch_size=1, return_hypotheses=True)
    tdt_hyps = tdt_out[0] if isinstance(tdt_out, tuple) else tdt_out
    tdt_first = tdt_hyps[0]
    y_seq = tdt_first.y_sequence
    if isinstance(y_seq, torch.Tensor):
        y_seq = y_seq.cpu().tolist()
    tdt_token_ids = np.array(list(y_seq), dtype=np.int32)
    tdt_text = tdt_first.text if hasattr(tdt_first, "text") else str(tdt_first)

    # Tokenizer detok fixture: a small hand-picked set of regular BPE ids
    # (avoid id=0 <unk> and blank_id=1024 which is beyond vocab).
    detok_ids = np.array([10, 25, 100, 3, 7], dtype=np.int32)
    detok_text = m.tokenizer.ids_to_text(detok_ids.tolist())

    # Authoritative CTC transcript of the clip from NeMo's own greedy CTC
    # decoder. parakeet-tdt_ctc-110m is a hybrid; its default transcribe() uses
    # the RNNT/TDT head, so we explicitly switch to the CTC head first. This is
    # the end-to-end ground truth for pk::transcribe (Task 11). transcribe()
    # returns a tuple (list_of_results, optional_beam_results); with
    # return_hypotheses=False each result is a plain text string, but be
    # defensive and extract .text from Hypothesis objects if present.
    m.change_decoding_strategy(decoder_type="ctc")
    assert m.cur_decoder == "ctc", f"expected CTC decoder, got {m.cur_decoder}"
    with torch.no_grad():
        out = m.transcribe([args.audio], batch_size=1, return_hypotheses=False)
    hyps = out[0] if isinstance(out, tuple) else out
    first = hyps[0]
    ctc_text = first.text if hasattr(first, "text") else str(first)

    w = gguf.GGUFWriter(args.output, "parakeet-baseline")
    for k, v in cap.items():
        w.add_tensor(k, _squeeze(v))
    w.add_tensor("ctc_argmax_ids", np.ascontiguousarray(ids))
    w.add_tensor("detok_ids", detok_ids)
    w.add_tensor("pred_input_ids", np.ascontiguousarray(pred_input_ids))
    w.add_tensor("joint_enc_frames", np.ascontiguousarray(joint_enc_frames_arr))
    # tdt_token_ids: NeMo's TDT greedy reference token ids. gguf cannot store a
    # zero-length tensor, so only emit the tensor when non-empty; always record
    # the length as a KV so consumers can distinguish "empty" from "missing".
    w.add_uint32("baseline.tdt_token_count", int(tdt_token_ids.shape[0]))
    if tdt_token_ids.shape[0] > 0:
        w.add_tensor("tdt_token_ids", np.ascontiguousarray(tdt_token_ids))
    w.add_string("baseline.detok_text", detok_text)
    w.add_string("baseline.ctc_text", ctc_text)
    w.add_string("baseline.tdt_text", tdt_text)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    shapes = {k: tuple(_squeeze(v).shape) for k, v in cap.items()}
    shapes["ctc_argmax_ids"] = tuple(ids.shape)
    shapes["detok_ids"] = tuple(detok_ids.shape)
    shapes["pred_input_ids"] = tuple(pred_input_ids.shape)
    shapes["joint_enc_frames"] = tuple(joint_enc_frames_arr.shape)
    if tdt_token_ids.shape[0] > 0:
        shapes["tdt_token_ids"] = tuple(tdt_token_ids.shape)
    print("baseline tensors:", shapes)
    print(f"baseline.detok_text: {repr(detok_text)}")
    print(f"baseline.ctc_text: {repr(ctc_text)}")
    print(f"baseline.tdt_text: {repr(tdt_text)}")
    print(f"tdt_token_ids ({tdt_token_ids.shape[0]}): {tdt_token_ids.tolist()}")
    print(
        f"transducer: pred_input_ids={pred_input_ids.tolist()} add_sos={add_sos} "
        f"U+1={shapes['pred_out'][0]} sos_embed_zero={sos_embed_is_zero} "
        f"joint raw_logits=True joint_enc_frames={n_frames}"
    )
    print(f"wrote {args.output}: tensors={len(shapes)} (dither=0.0, explicit forward)")


if __name__ == "__main__":
    main()
