#include "parakeet.h"
#include "parakeet_capi.h"
#include "model.hpp"
#include "model_loader.hpp"
#include "audio_io.hpp"
#include "streaming.hpp"
#include "transcription.hpp"
#include "ggml_graph.hpp"   // pk::set_num_threads
#include "ggml.h"
#include "gguf.h"
#include <chrono>
#include <fstream>
#include <memory>
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

// Cache-aware streaming transcription for the EOU streaming model. Feeds the WAV
// to a pk::StreamingSession in the model's exact chunk schedule, printing partial
// text incrementally and `[EOU @ <t>s]` / `[EOB @ <t>s]` markers when events
// fire, then the finalize() tail. Returns 0/1.
//
// When `timestamps` is set, also prints one line per finalized word
// (`<start>-<end>  <word>  (<conf>)`) after the running text/EOU line.
static int cmd_transcribe_stream(const std::string& model, const std::string& input,
                                 bool timestamps) {
    pk::ModelLoader ml;
    if (!ml.load(model)) {
        std::fprintf(stderr, "parakeet-cli: failed to load model %s\n", model.c_str());
        return 1;
    }
    if (!ml.config().streaming.present) {
        std::fprintf(stderr,
            "parakeet-cli: --stream requires a cache-aware streaming model "
            "(e.g. parakeet_realtime_eou_120m-v1); %s is not one\n", model.c_str());
        return 1;
    }
    pk::Audio audio;
    if (!pk::load_audio_16k_mono(input, audio)) {
        std::fprintf(stderr, "parakeet-cli: failed to load audio %s\n", input.c_str());
        return 1;
    }

    try {
        pk::StreamingSession sess(ml);
        std::vector<pk::Word> all_words;  // collected for the --timestamps recap
        std::printf("[stream] ");
        std::fflush(stdout);
        pk::run_stream_over_pcm(
            sess, ml, audio.samples,
            [&](const std::string& new_text, const std::vector<pk::EouEvent>& evs,
                const std::vector<pk::Word>& words) {
                if (!new_text.empty()) {
                    std::printf("%s", new_text.c_str());
                    std::fflush(stdout);
                }
                for (const pk::EouEvent& e : evs) {
                    std::printf(" [%s @ %.2fs]", e.is_eob ? "EOB" : "EOU", e.time_sec);
                    std::fflush(stdout);
                }
                if (timestamps)
                    all_words.insert(all_words.end(), words.begin(), words.end());
            });
        // Flush the end-of-stream tail (no extra <EOU> is fabricated if NeMo's
        // streaming would not emit one for this clip).
        std::string tail = sess.finalize();
        if (!tail.empty()) std::printf("%s", tail.c_str());
        if (timestamps) {
            // The trailing open word finalizes only at finalize(); drain it now.
            std::vector<pk::Word> last = sess.drain_words();
            all_words.insert(all_words.end(), last.begin(), last.end());
        }
        std::printf("\n");
        // Also print the full transcript on its own line for easy capture.
        std::printf("[stream:final] %s\n", sess.text().c_str());
        if (timestamps) {
            for (const pk::Word& w : all_words)
                std::printf("%.2f-%.2f  %s  (%.2f)\n", w.start, w.end,
                            w.text.c_str(), w.conf);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "parakeet-cli: streaming failed: %s\n", e.what());
        return 1;
    }
    return 0;
}

// parakeet-cli transcribe --model <m.gguf> --input <wav> [--decoder ctc|tdt]
//                         [--stream]
// Prints the transcript. Default decoder is chosen by arch (TDT for transducer
// archs, CTC for ctc arch — matching NeMo's cur_decoder default). --stream uses
// the cache-aware streaming path (EOU streaming model only).
static int cmd_transcribe(int argc, char** argv) {
    std::string model, input, decoder_str;
    bool stream = false;
    bool timestamps = false;
    bool json = false;
    int threads = 0;  // 0 == unset -> use the persistent-backend default
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model = argv[++i];
        } else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input = argv[++i];
        } else if (std::strcmp(argv[i], "--decoder") == 0 && i + 1 < argc) {
            decoder_str = argv[++i];
        } else if (std::strcmp(argv[i], "--stream") == 0) {
            stream = true;
        } else if (std::strcmp(argv[i], "--timestamps") == 0) {
            timestamps = true;
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--json") == 0) {
            json = true;
        }
    }
    if (model.empty() || input.empty()) {
        std::fprintf(stderr,
            "usage: parakeet-cli transcribe --model <m.gguf> --input <wav> "
            "[--decoder ctc|tdt] [--stream] [--timestamps] [--threads N] [--json]\n");
        return 2;
    }
    // Apply the thread override (offline + streaming graph compute). When unset
    // the persistent-backend default (kDefaultThreads) is used.
    if (threads > 0) pk::set_num_threads(threads);

    if (stream) {
        if (!decoder_str.empty()) {
            std::fprintf(stderr,
                "parakeet-cli: --stream is RNN-T only; --decoder is ignored\n");
        }
        if (json) {
            std::fprintf(stderr,
                "parakeet-cli: --json is not supported with --stream\n");
            return 2;
        }
        return cmd_transcribe_stream(model, input, timestamps);
    }

    // Resolve the decoder selector.
    pk::Decoder dec = pk::Decoder::kDefault;
    int dec_int = 0;  // C-API decoder selector (0 default, 1 ctc, 2 tdt)
    if (!decoder_str.empty()) {
        if (decoder_str == "ctc") {
            dec = pk::Decoder::kCTC;
            dec_int = 1;
        } else if (decoder_str == "tdt") {
            dec = pk::Decoder::kTDT;
            dec_int = 2;
        } else {
            std::fprintf(stderr, "parakeet-cli: unknown --decoder '%s' (want ctc|tdt)\n",
                         decoder_str.c_str());
            return 2;
        }
    }

    // --json: emit the C-API JSON document (text + word/token timestamps + conf).
    if (json) {
        parakeet_ctx* ctx = parakeet_capi_load(model.c_str());
        if (!ctx) {
            std::fprintf(stderr, "parakeet-cli: failed to load model %s\n", model.c_str());
            return 1;
        }
        char* j = parakeet_capi_transcribe_path_json(ctx, input.c_str(), dec_int);
        if (!j) {
            std::fprintf(stderr, "parakeet-cli: transcribe failed: %s\n",
                         parakeet_capi_last_error(ctx));
            parakeet_capi_free(ctx);
            return 1;
        }
        std::printf("%s\n", j);
        parakeet_capi_free_string(j);
        parakeet_capi_free(ctx);
        return 0;
    }

    // --timestamps: print one line per word `<start>-<end>  <word>  (<conf>)`.
    if (timestamps) {
        try {
            std::unique_ptr<pk::Model> m = pk::Model::load(model);
            if (!m) {
                std::fprintf(stderr, "parakeet-cli: failed to load model %s\n", model.c_str());
                return 1;
            }
            pk::Transcription tr = m->transcribe_path_with_timestamps(input, dec);
            for (const pk::Word& w : tr.words)
                std::printf("%.2f-%.2f  %s  (%.2f)\n", w.start, w.end,
                            w.text.c_str(), w.conf);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "transcribe failed: %s\n", e.what());
            return 1;
        }
        return 0;
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

// ---------------------------------------------------------------------------
// bench: clean per-file transcription timing for the NeMo-vs-ours benchmark.
//
// Loads the model ONCE (timed -> load_ms), then for each audio path in a
// manifest: loads the WAV, computes audio_sec = samples/16000, and times ONLY
// pk::Model::transcribe_path (steady_clock, ms). Emits a single JSON document
// so the Python runner can compute RTFx (audio_sec / proc_sec) fairly without
// the one-time model-load cost polluting the per-file numbers.
//
// --threads N controls the ggml compute threads for EVERY graph computation
// (via pk::set_num_threads, read inside pk::run_graph). When unset we leave the
// process-global override clear so the components' built-in default is used.
// ---------------------------------------------------------------------------

// Append `s` to `out` as a JSON string literal (quoted), escaping per RFC 8259.
// Mirrors append_json_string in src/parakeet_capi.cpp (UTF-8 >= 0x80 passes
// through verbatim).
static void bench_json_string(std::string& out, const std::string& s) {
    out += '"';
    char esc[8];
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    std::snprintf(esc, sizeof(esc), "\\u%04x", (unsigned)c);
                    out += esc;
                } else {
                    out += (char)c;
                }
        }
    }
    out += '"';
}

// Reads one audio path per line from the manifest. Blank lines and lines whose
// first non-space char is '#' are ignored. A tab-separated `path\tref` line is
// accepted -- only the first field (the path) is taken. Leading/trailing
// whitespace on the path field is trimmed.
static std::vector<std::string> read_manifest(const std::string& path, bool& ok) {
    std::vector<std::string> out;
    std::ifstream f(path);
    if (!f) { ok = false; return out; }
    ok = true;
    std::string line;
    while (std::getline(f, line)) {
        // Strip a trailing CR (CRLF manifests).
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // First field only (tab-separated path\tref is allowed).
        size_t tab = line.find('\t');
        std::string p = (tab == std::string::npos) ? line : line.substr(0, tab);
        // Trim surrounding whitespace.
        size_t b = p.find_first_not_of(" \t");
        if (b == std::string::npos) continue;                 // blank
        size_t e = p.find_last_not_of(" \t");
        p = p.substr(b, e - b + 1);
        if (p.empty() || p[0] == '#') continue;               // blank / comment
        out.push_back(p);
    }
    return out;
}

static int cmd_bench(int argc, char** argv) {
    std::string model, manifest, decoder_str, json_out;
    int threads = 0;  // 0 == unset -> use the components' built-in default
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model = argv[++i];
        } else if (std::strcmp(argv[i], "--manifest") == 0 && i + 1 < argc) {
            manifest = argv[++i];
        } else if (std::strcmp(argv[i], "--decoder") == 0 && i + 1 < argc) {
            decoder_str = argv[++i];
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_out = argv[++i];
        }
    }
    if (model.empty() || manifest.empty()) {
        std::fprintf(stderr,
            "usage: parakeet-cli bench --model <m.gguf> --manifest <file> "
            "[--decoder ctc|tdt] [--threads N] [--json <out>]\n");
        return 2;
    }

    // Resolve the decoder selector (matches `transcribe`).
    pk::Decoder dec = pk::Decoder::kDefault;
    if (!decoder_str.empty()) {
        if (decoder_str == "ctc") {
            dec = pk::Decoder::kCTC;
        } else if (decoder_str == "tdt") {
            dec = pk::Decoder::kTDT;
        } else {
            std::fprintf(stderr,
                "parakeet-cli bench: unknown --decoder '%s' (want ctc|tdt)\n",
                decoder_str.c_str());
            return 2;
        }
    }

    // Apply the thread count to EVERY ggml graph computation. When --threads is
    // omitted we report the components' built-in default in the JSON so the
    // runner records the thread count that was actually used.
    int reported_threads = threads;
    if (threads > 0) {
        pk::set_num_threads(threads);
    } else {
        reported_threads = 8;  // the persistent-backend default (kDefaultThreads)
    }

    bool man_ok = false;
    std::vector<std::string> paths = read_manifest(manifest, man_ok);
    if (!man_ok) {
        std::fprintf(stderr, "parakeet-cli bench: failed to read manifest %s\n",
                     manifest.c_str());
        return 1;
    }
    if (paths.empty()) {
        std::fprintf(stderr, "parakeet-cli bench: manifest %s has no audio paths\n",
                     manifest.c_str());
        return 1;
    }

    using clock = std::chrono::steady_clock;
    auto ms_since = [](clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };

    // Load the model ONCE -- timed, and excluded from per-file proc_ms.
    auto t_load = clock::now();
    std::unique_ptr<pk::Model> m = pk::Model::load(model);
    double load_ms = ms_since(t_load);
    if (!m) {
        std::fprintf(stderr, "parakeet-cli bench: failed to load model %s\n",
                     model.c_str());
        return 1;
    }

    struct FileResult { std::string path; double audio_sec; double proc_ms; std::string text; };
    std::vector<FileResult> results;
    results.reserve(paths.size());

    for (const std::string& p : paths) {
        pk::Audio audio;
        if (!pk::load_audio_16k_mono(p, audio)) {
            std::fprintf(stderr, "parakeet-cli bench: failed to load audio %s\n",
                         p.c_str());
            return 1;
        }
        // audio_sec from the decoded 16 kHz sample count.
        double audio_sec = (double)audio.samples.size() / 16000.0;

        // Time ONLY the transcription (model already loaded).
        auto t_proc = clock::now();
        std::string text;
        try {
            text = m->transcribe_pcm(audio.samples, 16000, dec);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "parakeet-cli bench: transcribe failed on %s: %s\n",
                         p.c_str(), e.what());
            return 1;
        }
        double proc_ms = ms_since(t_proc);
        results.push_back({p, audio_sec, proc_ms, text});
    }

    // Hand-roll the JSON document.
    std::string out;
    out.reserve(256 + results.size() * 128);
    out += "{\"model\":";
    bench_json_string(out, model);
    char numbuf[64];
    std::snprintf(numbuf, sizeof(numbuf), ",\"threads\":%d", reported_threads);
    out += numbuf;
    std::snprintf(numbuf, sizeof(numbuf), ",\"load_ms\":%.3f", load_ms);
    out += numbuf;
    out += ",\"files\":[";
    for (size_t i = 0; i < results.size(); ++i) {
        if (i) out += ',';
        out += "{\"path\":";
        bench_json_string(out, results[i].path);
        std::snprintf(numbuf, sizeof(numbuf), ",\"audio_sec\":%.6f", results[i].audio_sec);
        out += numbuf;
        std::snprintf(numbuf, sizeof(numbuf), ",\"proc_ms\":%.3f", results[i].proc_ms);
        out += numbuf;
        out += ",\"text\":";
        bench_json_string(out, results[i].text);
        out += '}';
    }
    out += "]}";

    if (!json_out.empty()) {
        std::ofstream of(json_out, std::ios::binary | std::ios::trunc);
        if (!of) {
            std::fprintf(stderr, "parakeet-cli bench: failed to write %s\n",
                         json_out.c_str());
            return 1;
        }
        of << out << '\n';
    } else {
        std::printf("%s\n", out.c_str());
    }
    return 0;
}

// Run a subcommand, then free the process-global backend while the GPU driver is
// still alive (the subcommand's local Model is already destroyed by the time it
// returns, releasing its device weight buffer). Avoids the CUDA "driver shutting
// down" abort caused by static-destruction teardown ordering.
static int run_and_shutdown(int (*fn)(int, char**), int argc, char** argv) {
    int rc = fn(argc, argv);
    pk::shutdown_backend();
    return rc;
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "info") == 0)
        return run_and_shutdown([](int, char** a) { return cmd_info(a[0]); }, 1, argv + 2);
    if (argc >= 2 && std::strcmp(argv[1], "transcribe") == 0)
        return run_and_shutdown(cmd_transcribe, argc - 2, argv + 2);
    if (argc >= 2 && std::strcmp(argv[1], "quantize") == 0)
        return run_and_shutdown(cmd_quantize, argc - 2, argv + 2);
    if (argc >= 2 && std::strcmp(argv[1], "bench") == 0)
        return run_and_shutdown(cmd_bench, argc - 2, argv + 2);
    std::fprintf(stderr,
        "usage:\n"
        "  parakeet-cli info <model.gguf>\n"
        "  parakeet-cli transcribe --model <model.gguf> --input <wav> "
        "[--decoder ctc|tdt] [--stream] [--timestamps] [--threads N] [--json]\n"
        "  parakeet-cli quantize <in.gguf> <out.gguf> "
        "<q4_0|q5_0|q8_0|q4_k|q5_k|q6_k>\n"
        "  parakeet-cli bench --model <model.gguf> --manifest <file> "
        "[--decoder ctc|tdt] [--threads N] [--json <out>]\n");
    return 2;
}
