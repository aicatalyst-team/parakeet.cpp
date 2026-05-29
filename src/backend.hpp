#pragma once
#include <cstddef>
#include <functional>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace pk {

class ModelLoader;

// Persistent CPU compute backend + reusable graph allocator.
//
// parakeet's inference path builds hundreds of tiny ggml graphs per utterance
// (272 on the 110m for a 10 s clip). The old `run_graph` did a full
// `ggml_init` -> build -> `ggml_graph_compute_with_ctx` -> `ggml_free` on every
// call, so ~40% of wall time was allocator churn (init/alloc + free of the
// per-call compute buffer), and the disposable threadpool was respawned each
// time. This mirrors the wall rt-detr.cpp hit; the fix is the same: a
// persistent `ggml_backend_t` (CPU) + a persistent `ggml_gallocr_t` reused
// across every graph computation.
//
// Backend::compute() builds the graph in a `no_alloc=true` context (metadata
// only), allocates it via the persistent gallocr (which packs intermediates
// with lifetime-aware reuse AND keeps the underlying compute buffer alive
// across calls, so the steady-state loop sees no mmap/munmap traffic), pushes
// the host input data AFTER alloc, runs it on the persistent backend, and reads
// the output back.
//
// CORRECTNESS-CRITICAL ordering: with `no_alloc=true`, a tensor's `->data` is
// NULL until `ggml_gallocr_alloc_graph`. So input tensor data MUST be written
// AFTER alloc (via `ggml_backend_tensor_set`), never inline in the build
// lambda. The build lambda registers each host-backed input by calling
// `pk::add_graph_input(...)`; Backend defers the copy until after alloc.
class Backend {
public:
    // Construct a CPU backend with `n_threads` worker threads (<=0 -> 1). The
    // gallocr is created lazily on the first compute and reused afterwards.
    explicit Backend(int n_threads);
    ~Backend();

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    // Update the worker-thread count (cheap; just calls
    // ggml_backend_cpu_set_n_threads). <=0 is clamped to 1.
    void set_n_threads(int n_threads);
    int  n_threads() const { return n_threads_; }

    // Build + run a single graph on the persistent backend/gallocr and copy the
    // output tensor's f32 contents into `out`.
    //
    //   build(ctx): create the graph in `ctx` (a no_alloc=true context).
    //     - Input tensors: `ggml_new_tensor*` then register their host data via
    //       `pk::add_graph_input(t, host_ptr, nbytes)`. Do NOT write `t->data`
    //       (it is NULL until alloc). add_graph_input marks the tensor as a
    //       graph input so the gallocr keeps it.
    //     - Weights: reference loader tensors directly (their `->data` is set by
    //       the mmap'd loader context, so the gallocr treats them as already
    //       allocated and never touches them) OR build small host-computed
    //       constants as inputs via add_graph_input.
    //     - Returns the output tensor (Backend marks it as an output).
    //
    // Returns true on success.
    bool compute(const std::function<ggml_tensor*(ggml_context*)>& build,
                 std::vector<float>& out);

    // Backend-internal hook used by add_graph_input(). Registers a host->device
    // input copy to be performed after ggml_gallocr_alloc_graph. Public so the
    // free helper can reach it, but callers should use add_graph_input().
    void register_input(ggml_tensor* t, const void* host, size_t nbytes);

private:
    struct Impl;
    Impl* impl_;
    int   n_threads_ = 1;
};

// Register a host-backed graph input for the currently-active Backend::compute
// build phase. Marks `t` as a ggml graph input and records that `nbytes` from
// `host` must be copied into `t` AFTER the graph is allocated. Must be called
// from inside a build lambda passed to Backend::compute (it routes to the
// Backend driving the current compute via a thread-local pointer).
//
// `host` must stay valid until Backend::compute returns (it does for all
// callers: the data is the caller's input vector, alive across the call).
void add_graph_input(ggml_tensor* t, const void* host, size_t nbytes);

// Create a graph input tensor of the given type and ne[] inside `ctx`, mark it
// as a graph input, and register a host->device copy of `nbytes` from `host`
// to be performed after the gallocr allocates it. The data is NOT written
// inline (the tensor's ->data is NULL until alloc); use this instead of
// `ggml_new_tensor*` + memcpy in build lambdas. Returns the new tensor.
ggml_tensor* graph_input_tensor(ggml_context* ctx, int type, int n_dims,
                                const int64_t* ne, const void* host,
                                size_t nbytes);

// Bring a weight tensor from the loader into the graph as an input. The loader
// stores weights in an mmap'd ggml context (no backend buffer), so we cannot
// reference them directly as graph leaves (a reshape/view of them would fail
// ggml_backend_view_init, which requires the source to live in a backend
// buffer). Instead we materialize a same-type, same-ne input tensor and copy
// the loader's raw bytes into it after alloc. Allowlisted linears may be
// f16/q8_0; the raw quantized bytes are copied through unchanged and
// ggml_mul_mat dequantizes src0 on the fly. Returns nullptr iff `name` is
// absent (for optional weights); asserts present otherwise via clone_weight.
ggml_tensor* clone_weight(ggml_context* ctx, const ModelLoader& ml,
                          const char* name);
ggml_tensor* clone_weight_opt(ggml_context* ctx, const ModelLoader& ml,
                              const char* name);

} // namespace pk
