#include "ggml_graph.hpp"
#include "common.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdlib>
#include <cstring>

namespace pk {

bool run_graph(size_t mem_bytes, int n_threads,
               const std::function<ggml_tensor*(ggml_context*)>& build,
               std::vector<float>& out) {
    // Initialise a fixed-size context with inline tensor storage (no_alloc=false).
    struct ggml_init_params params = {
        /* .mem_size   = */ mem_bytes,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        PK_LOG("ggml_init failed (mem_bytes=%zu)", mem_bytes);
        return false;
    }

    // Let the caller build the graph and produce the output tensor.
    struct ggml_tensor* output = build(ctx);
    if (!output) {
        PK_LOG("build() returned null output tensor");
        ggml_free(ctx);
        return false;
    }

    // Build the forward computation graph.
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);

    // Compute on CPU using ggml_graph_compute_with_ctx (allocates work buffer
    // from the same context).  This is the simplest CPU path exposed by
    // third_party/ggml/include/ggml-cpu.h in this v0.13.0 tree.
    enum ggml_status status = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    if (status != GGML_STATUS_SUCCESS) {
        PK_LOG("ggml_graph_compute_with_ctx failed (status=%d)", (int)status);
        ggml_free(ctx);
        return false;
    }

    // Copy output data into the caller's vector.
    size_t n = (size_t)ggml_nelements(output);
    out.resize(n);
    const float* src = ggml_get_data_f32(output);
    std::memcpy(out.data(), src, n * sizeof(float));

    ggml_free(ctx);
    return true;
}

} // namespace pk
