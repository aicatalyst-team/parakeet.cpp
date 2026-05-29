#include "encoder.hpp"
#include "subsampling.hpp"
#include "conformer.hpp"
#include "pos_enc.hpp"
#include "ggml_graph.hpp"
#include "backend.hpp"
#include "graph_builder.hpp"
#include "ggml.h"
#include <cassert>
#include <cmath>
#include <vector>

namespace pk {

Encoder::Encoder(const ModelLoader& ml)
    : ml_(ml) {
    d_model_  = (int)ml.config().d_model;
    n_layers_ = (int)ml.config().n_layers;
    xscaling_ = ml.config().xscaling;
    assert(n_layers_ > 0 && d_model_ > 0);
}

void Encoder::forward(const std::vector<float>& mel, int n_mels, int T,
                      std::vector<float>& enc_out, int& d_model, int& Tout) const {
    std::vector<int> none;
    std::vector<std::vector<float>> ignored;
    forward_capture(mel, n_mels, T, enc_out, d_model, Tout, none, ignored);
}

void Encoder::forward_capture(const std::vector<float>& mel, int n_mels, int T,
                              std::vector<float>& enc_out, int& d_model, int& Tout,
                              const std::vector<int>& capture_layers,
                              std::vector<std::vector<float>>& layer_outs) const {
    // The WHOLE encoder is ONE ggml graph: subsampling -> xscaling -> N conformer
    // layers (each FFN1+attn+conv+FFN2+norm_out in-graph) -> final transpose,
    // computed in a SINGLE Backend::compute. (Mel stays plain C++, transposed and
    // fed as the input inside the subsampling builder.) Versus the old ~85
    // per-utterance graphs, this lets the CPU threadpool parallelise across the
    // entire encoder. The per-component graph-BUILDERS (Subsampling/RelPos/
    // ConformerLayer::build_graph) are reused verbatim by the unit tests.
    layer_outs.assign(capture_layers.size(), {});

    // Pos_emb is computed host-side once T' is known (deterministic from T, but
    // we read it from the subsampling builder). Storage lives in the pool so it
    // outlives Backend::compute. capture_layers map: layer idx -> output slot.
    GraphInputPool pool;
    Subsampling sub(ml_);
    int Tp = 0, valid_len = 0, dm_out = 0;

    bool ok = pk::run_graph(/*mem_bytes*/0, /*n_threads*/0,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // ---- 1. Subsampling: mel [n_mels,T] -> x [d_model, T'] (+ valid). ----
            ggml_tensor* x = sub.build_graph(ctx, mel, n_mels, T, pool, Tp,
                                             valid_len);
            dm_out = d_model_;
            assert((int)x->ne[0] == d_model_);

            // ---- 2. xscaling (gated; off for this model). NeMo scales x. ----
            if (xscaling_) {
                x = ggml_scale(ctx, x, std::sqrt((float)d_model_));
            }

            // ---- 3. Relative positional encoding pos_emb [d_model, 2T'-1]. ----
            const int pos_len = 2 * Tp - 1;
            std::vector<float>& pe_host = pool.alloc_f32();
            rel_pos_encoding(Tp, d_model_, pe_host); // row-major [pos_len, d_model]
            int64_t pe_ne[2] = {d_model_, pos_len};
            ggml_tensor* pe = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, pe_ne,
                                  pe_host.data(), pe_host.size() * sizeof(float));

            // ---- 4. Conformer layer stack (all in-graph). ----
            for (int i = 0; i < n_layers_; ++i) {
                ConformerLayer layer(ml_, i);
                x = layer.build_graph(ctx, x, Tp, pe, pos_len, valid_len, pool);
                // Capture requested layer outputs from the SAME graph (row-major
                // [T', d_model], matching the layer output orientation).
                for (size_t c = 0; c < capture_layers.size(); ++c) {
                    if (capture_layers[c] == i)
                        pk::capture_graph_output(x, &layer_outs[c]);
                }
            }

            // ---- 5. Final transpose [d_model, T'] -> [T', d_model] (channels-
            //         first): ggml transpose+cont gives ne0=T', ne1=d_model ->
            //         row-major [d_model, T'] = enc_out[c*T' + t]. ----
            x = ggml_cont(ctx, ggml_transpose(ctx, x));
            return x; // ne [T', d_model] -> row-major [d_model, T']
        }, enc_out);

    assert(ok && "encoder graph failed");
    (void)ok;

    d_model = dm_out;
    Tout = Tp;
}

} // namespace pk
