#include "parakeet.h"
#include "model_loader.hpp"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int cmd_info(const char* path) {
    pk::ModelLoader ml;
    if (!ml.load(path)) { std::fprintf(stderr, "failed to load %s\n", path); return 1; }
    const pk::ParakeetConfig& c = ml.config();
    std::printf("parakeet.cpp %s\n", parakeet_version());
    std::printf("model: %s\n", path);
    std::printf("  arch            : %s\n", c.arch.c_str());
    std::printf("  d_model/layers/heads: %u / %u / %u\n", c.d_model, c.n_layers, c.n_heads);
    std::printf("  conv_kernel/norm: %u / %s\n", c.conv_kernel, c.conv_norm_type.c_str());
    std::printf("  xscaling        : %s\n", c.xscaling ? "true" : "false");
    std::printf("  subsampling     : x%u (ch=%u)\n", c.subsampling_factor, c.subsampling_conv_channels);
    std::printf("  mel/n_fft/win/hop: %u / %u / %u / %u\n", c.n_mels, c.n_fft, c.win_length, c.hop_length);
    std::printf("  vocab/blank     : %u / %u\n", c.vocab_size, c.blank_id);
    if (!c.tdt_durations.empty()) {
        std::printf("  tdt durations   : [");
        for (size_t i=0;i<c.tdt_durations.size();++i) std::printf("%s%d", i?",":"", c.tdt_durations[i]);
        std::printf("]\n");
    }
    // Cache-aware streaming / causal config (Phase 5). Only meaningful for
    // streaming models; offline models report style "regular" with full context.
    std::printf("  att_context     : [%d,%d] %s\n",
                c.att_context_left, c.att_context_right, c.att_context_style.c_str());
    std::printf("  causal ds/conv  : %s / %s\n",
                c.causal_downsampling ? "true" : "false",
                c.conv_causal ? "true" : "false");
    if (c.streaming.present) {
        const pk::StreamingCfg& s = c.streaming;
        auto print_ivec = [](const char* label, const std::vector<int32_t>& v) {
            std::printf("  %-15s : [", label);
            for (size_t i = 0; i < v.size(); ++i) std::printf("%s%d", i ? "," : "", v[i]);
            std::printf("]\n");
        };
        std::printf("  streaming       : enabled\n");
        print_ivec("  chunk_size", s.chunk_size);
        print_ivec("  shift_size", s.shift_size);
        print_ivec("  pre_enc_cache", s.pre_encode_cache_size);
        std::printf("    cache_drop     : %d\n", s.cache_drop_size);
        std::printf("    last_ch_cache  : %d\n", s.last_channel_cache_size);
        std::printf("    valid_out_len  : %d\n", s.valid_out_len);
        std::printf("    drop_extra_pre : %d\n", s.drop_extra_pre_encoded);
    }
    return 0;
}

// parakeet-cli transcribe --model <m.gguf> --input <wav> [--decoder ctc|tdt]
// Prints the transcript. Default decoder is chosen by arch (TDT for transducer
// archs, CTC for ctc arch — matching NeMo's cur_decoder default).
static int cmd_transcribe(int argc, char** argv) {
    std::string model, input, decoder_str;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model = argv[++i];
        } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input = argv[++i];
        } else if (std::strcmp(argv[i], "--decoder") == 0 && i + 1 < argc) {
            decoder_str = argv[++i];
        }
    }
    if (model.empty() || input.empty()) {
        std::fprintf(stderr,
            "usage: parakeet-cli transcribe --model <m.gguf> --input <wav> "
            "[--decoder ctc|tdt]\n");
        return 2;
    }

    // Resolve the decoder selector.
    pk::Decoder dec = pk::Decoder::kDefault;
    if (!decoder_str.empty()) {
        if (decoder_str == "ctc") {
            dec = pk::Decoder::kCTC;
        } else if (decoder_str == "tdt") {
            dec = pk::Decoder::kTDT;
        } else {
            std::fprintf(stderr, "parakeet-cli: unknown --decoder '%s' (want ctc|tdt)\n",
                         decoder_str.c_str());
            return 2;
        }
    }

    std::string text;
    try {
        text = pk::transcribe(model, input, dec);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "transcribe failed: %s\n", e.what());
        return 1;
    }
    std::printf("%s\n", text.c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// quantize: re-quantize the SAME allowlisted linear weights as the converter's
// f16/q8_0 path (docs/quantization.md) to a target ggml type -- including the
// K-quants (Q4_K/Q5_K/Q6_K) that the Python gguf writer can't produce. Every
// non-allowlisted tensor (and any allowlisted tensor that isn't shape-eligible)
// is copied verbatim in its stored type. All KV metadata is copied unchanged.
// ---------------------------------------------------------------------------

// Returns true if `name` is on the Task-2 quantization allowlist (mul_mat src0
// weights only). Mirrors _QUANTIZABLE_PATTERNS in convert_parakeet_to_gguf.py.
//   ^encoder\.layers\.\d+\.feed_forward[12]\.linear[12]\.weight$
//   ^encoder\.layers\.\d+\.self_attn\.linear_(q|k|v|out|pos)\.weight$
//   ^encoder\.pre_encode\.out\.weight$
//   ^joint\.enc\.weight$
//   ^joint\.pred\.weight$
static bool is_quantizable_name(const std::string& n) {
    if (n == "encoder.pre_encode.out.weight") return true;
    if (n == "joint.enc.weight") return true;
    if (n == "joint.pred.weight") return true;
    // encoder.layers.<d+>.<rest>
    const char* prefix = "encoder.layers.";
    if (n.rfind(prefix, 0) != 0) return false;
    size_t i = std::strlen(prefix);
    size_t start = i;
    while (i < n.size() && std::isdigit(static_cast<unsigned char>(n[i]))) ++i;
    if (i == start || i >= n.size() || n[i] != '.') return false;  // need digits then '.'
    std::string rest = n.substr(i + 1);
    // feed_forward{1,2}.linear{1,2}.weight
    if ((rest == "feed_forward1.linear1.weight") ||
        (rest == "feed_forward1.linear2.weight") ||
        (rest == "feed_forward2.linear1.weight") ||
        (rest == "feed_forward2.linear2.weight")) return true;
    // self_attn.linear_{q,k,v,out,pos}.weight
    if ((rest == "self_attn.linear_q.weight")   ||
        (rest == "self_attn.linear_k.weight")   ||
        (rest == "self_attn.linear_v.weight")   ||
        (rest == "self_attn.linear_out.weight") ||
        (rest == "self_attn.linear_pos.weight")) return true;
    return false;
}

static bool parse_quant_type(const std::string& s, ggml_type& out) {
    std::string t = s;
    std::transform(t.begin(), t.end(), t.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    if (t == "q4_0") { out = GGML_TYPE_Q4_0; return true; }
    if (t == "q5_0") { out = GGML_TYPE_Q5_0; return true; }
    if (t == "q8_0") { out = GGML_TYPE_Q8_0; return true; }
    if (t == "q4_k") { out = GGML_TYPE_Q4_K; return true; }
    if (t == "q5_k") { out = GGML_TYPE_Q5_K; return true; }
    if (t == "q6_k") { out = GGML_TYPE_Q6_K; return true; }
    return false;
}

static bool is_k_quant(ggml_type t) {
    return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K;
}

static int cmd_quantize(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: parakeet-cli quantize <in.gguf> <out.gguf> "
            "<q4_0|q5_0|q8_0|q4_k|q5_k|q6_k>\n");
        return 2;
    }
    const std::string in_path  = argv[0];
    const std::string out_path = argv[1];
    ggml_type qtype;
    if (!parse_quant_type(argv[2], qtype)) {
        std::fprintf(stderr,
            "parakeet-cli quantize: unknown type '%s' "
            "(want q4_0|q5_0|q8_0|q4_k|q5_k|q6_k)\n", argv[2]);
        return 2;
    }

    // Load the source GGUF the same way ModelLoader does: gguf_init_from_file
    // with a backing ggml_context so the tensor data is read into memory.
    struct ggml_context* src_ctx = nullptr;
    struct gguf_init_params p{ /*no_alloc*/ false, /*ctx*/ &src_ctx };
    struct gguf_context* src = gguf_init_from_file(in_path.c_str(), p);
    if (!src) {
        std::fprintf(stderr, "failed to open %s\n", in_path.c_str());
        return 1;
    }

    const int64_t block = ggml_blck_size(qtype);  // 32 for q*_0, 256 for K-quants

    // Destination: empty gguf, copy all KV verbatim, then add tensors.
    struct gguf_context* dst = gguf_init_empty();
    gguf_set_kv(dst, src);  // copies every KV pair unchanged

    const int64_t nt = gguf_get_n_tensors(src);
    // Holds quantized buffers alive until gguf_write_to_file copies them out.
    std::vector<std::vector<uint8_t>> quant_bufs;
    quant_bufs.reserve(static_cast<size_t>(nt));
    // We must reset each quantized tensor's type/data on the in-memory ggml
    // tensor so gguf_add_tensor records the new type and points at our buffer.
    struct ggml_init_params meta_p{ /*mem_size*/ ggml_tensor_overhead() * (nt + 1),
                                    /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    struct ggml_context* meta = ggml_init(meta_p);

    int n_quant = 0, n_kept = 0, n_skipped = 0;
    for (int64_t i = 0; i < nt; ++i) {
        const char* name = gguf_get_tensor_name(src, i);
        struct ggml_tensor* t = ggml_get_tensor(src_ctx, name);

        bool do_quant = false;
        if (is_quantizable_name(name) && t->type == GGML_TYPE_F32) {
            const bool two_d   = (t->ne[2] == 1 && t->ne[3] == 1 &&
                                  t->ne[0] >= 32 && t->ne[1] >= 32);
            const bool blk_ok  = (t->ne[0] % block == 0);
            if (two_d && blk_ok) {
                do_quant = true;
            } else if (two_d && !blk_ok) {
                std::fprintf(stderr,
                    "  keep F32: %-48s ne0=%lld not divisible by %s block %lld\n",
                    name, (long long)t->ne[0],
                    is_k_quant(qtype) ? "K-quant superblock" : "block",
                    (long long)block);
                ++n_skipped;
            }
        }

        if (do_quant) {
            const int64_t n_per_row = t->ne[0];
            const int64_t nrows     = t->ne[1];
            const size_t  out_bytes = ggml_row_size(qtype, n_per_row) * (size_t)nrows;
            quant_bufs.emplace_back(out_bytes);
            const float* fsrc = (const float*)t->data;
            // imatrix = NULL: none of q4_0/q5_0/q8_0/q4_k/q5_k/q6_k require one.
            ggml_quantize_chunk(qtype, fsrc, quant_bufs.back().data(),
                                /*start*/ 0, nrows, n_per_row, /*imatrix*/ nullptr);
            // Build a fresh meta tensor in the new type pointing at our buffer.
            struct ggml_tensor* q = ggml_new_tensor_2d(meta, qtype, n_per_row, nrows);
            ggml_set_name(q, name);
            q->data = quant_bufs.back().data();
            gguf_add_tensor(dst, q);
            ++n_quant;
        } else {
            // Copy verbatim in its stored type/data.
            gguf_add_tensor(dst, t);
            ++n_kept;
        }
    }

    const bool ok = gguf_write_to_file(dst, out_path.c_str(), /*only_meta*/ false);

    std::printf("quantize: %s -> %s [%s]\n", in_path.c_str(), out_path.c_str(), argv[2]);
    std::printf("  quantized %d tensor(s), copied %d verbatim, %d allowlisted kept F32 (block).\n",
                n_quant, n_kept, n_skipped);

    ggml_quantize_free();
    ggml_free(meta);
    gguf_free(dst);
    gguf_free(src);
    ggml_free(src_ctx);

    if (!ok) {
        std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "info") == 0) return cmd_info(argv[2]);
    if (argc >= 2 && std::strcmp(argv[1], "transcribe") == 0)
        return cmd_transcribe(argc - 2, argv + 2);
    if (argc >= 2 && std::strcmp(argv[1], "quantize") == 0)
        return cmd_quantize(argc - 2, argv + 2);
    std::fprintf(stderr,
        "usage:\n"
        "  parakeet-cli info <model.gguf>\n"
        "  parakeet-cli transcribe --model <model.gguf> --input <wav> "
        "[--decoder ctc|tdt]\n"
        "  parakeet-cli quantize <in.gguf> <out.gguf> "
        "<q4_0|q5_0|q8_0|q4_k|q5_k|q6_k>\n");
    return 2;
}
