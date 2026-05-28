#pragma once
#include <functional>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace pk {

// One-shot CPU graph runner.
//
// Allocates a ggml context of `mem_bytes` (no_alloc=false, so tensor data is
// stored inline in the context buffer). Calls `build(ctx)` which must create
// input tensors, write their data, build the computation graph, and return the
// output tensor. Then the forward graph for that output is built, executed on
// CPU with `n_threads` threads, and `ggml_nelements(output)` f32 values are
// copied into `out`. Returns true on success, false on any failure.
bool run_graph(size_t mem_bytes, int n_threads,
               const std::function<ggml_tensor*(ggml_context*)>& build,
               std::vector<float>& out);

} // namespace pk
