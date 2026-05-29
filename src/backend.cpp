#include "backend.hpp"
#include "common.hpp"
#include "model_loader.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cstring>
#include <vector>

namespace pk {

namespace {
// Number of graph nodes the metadata context must hold. The biggest single
// graph today is a streaming conformer layer (~150 nodes); leave generous head
// room for Task 2's fused encoder (~85 layers worth of ops in one graph).
constexpr size_t kGraphSize = 16384;

struct PendingInput {
    ggml_tensor* tensor;
    const void*  host;
    size_t       nbytes;
};
} // namespace

struct Backend::Impl {
    ggml_backend_t    backend  = nullptr;
    ggml_gallocr_t    galloc   = nullptr;   // created lazily, reused across calls
    // Inputs registered by the build lambda for the IN-FLIGHT compute. Copied
    // into the gallocr-allocated tensors after ggml_gallocr_alloc_graph, then
    // cleared. Never overlaps across calls (compute is not re-entrant).
    std::vector<PendingInput> pending;
};

// Thread-local pointer to the Backend whose compute() build lambda is currently
// executing, so the free helper add_graph_input() can route registrations
// without threading the Backend through every component's build lambda. compute
// is not re-entrant on a single thread (a build lambda never calls compute), so
// a single pointer is sufficient.
static thread_local Backend* t_active = nullptr;

Backend::Backend(int n_threads) : impl_(new Impl()) {
    impl_->backend = ggml_backend_cpu_init();
    if (!impl_->backend) {
        PK_LOG("ggml_backend_cpu_init returned null");
        return;
    }
    set_n_threads(n_threads);
}

Backend::~Backend() {
    if (impl_) {
        // Free the gallocr BEFORE the backend: the gallocr owns the compute
        // scratch buffer (allocated via the backend's buffer_type). Matches
        // rt-detr's teardown order.
        if (impl_->galloc) ggml_gallocr_free(impl_->galloc);
        if (impl_->backend) ggml_backend_free(impl_->backend);
        delete impl_;
        impl_ = nullptr;
    }
}

void Backend::set_n_threads(int n_threads) {
    n_threads_ = n_threads > 0 ? n_threads : 1;
    if (impl_ && impl_->backend) {
        ggml_backend_cpu_set_n_threads(impl_->backend, n_threads_);
    }
}

void Backend::register_input(ggml_tensor* t, const void* host, size_t nbytes) {
    impl_->pending.push_back({t, host, nbytes});
}

bool Backend::compute(const std::function<ggml_tensor*(ggml_context*)>& build,
                      std::vector<float>& out) {
    if (!impl_ || !impl_->backend) {
        PK_LOG("Backend::compute called on an uninitialised backend");
        return false;
    }

    // Metadata-only context: holds graph + tensor structs, no tensor data
    // (no_alloc=true). Tensor data lives in the gallocr's persistent buffer.
    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * kGraphSize +
                            ggml_graph_overhead_custom(kGraphSize, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        PK_LOG("Backend::compute: ggml_init failed");
        return false;
    }

    // Drive add_graph_input() registrations to this Backend for the build call.
    impl_->pending.clear();
    Backend* prev_active = t_active;
    t_active = this;
    struct ggml_tensor* output = build(ctx);
    t_active = prev_active;

    if (!output) {
        PK_LOG("Backend::compute: build() returned null output tensor");
        impl_->pending.clear();
        ggml_free(ctx);
        return false;
    }
    // Mark the output so the gallocr does not recycle its storage before we read
    // it back, then expand the forward graph.
    ggml_set_output(output);

    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);
    ggml_build_forward_expand(gf, output);

    // Lazily create the persistent gallocr (reused on every subsequent call; it
    // only reallocates the underlying buffer when the graph grows beyond the
    // current high-water mark).
    if (!impl_->galloc) {
        impl_->galloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(impl_->backend));
        if (!impl_->galloc) {
            PK_LOG("Backend::compute: ggml_gallocr_new failed");
            impl_->pending.clear();
            ggml_free(ctx);
            return false;
        }
    }
    if (!ggml_gallocr_alloc_graph(impl_->galloc, gf)) {
        PK_LOG("Backend::compute: ggml_gallocr_alloc_graph failed");
        impl_->pending.clear();
        ggml_free(ctx);
        return false;
    }

    // Inputs are allocated now (->buffer/->data set): push host data in.
    for (const PendingInput& pi : impl_->pending) {
        ggml_backend_tensor_set(pi.tensor, pi.host, 0, pi.nbytes);
    }
    impl_->pending.clear();

    enum ggml_status status = ggml_backend_graph_compute(impl_->backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        PK_LOG("Backend::compute: ggml_backend_graph_compute failed (status=%d)",
               (int)status);
        ggml_free(ctx);
        return false;
    }

    size_t n = (size_t)ggml_nelements(output);
    out.resize(n);
    ggml_backend_tensor_get(output, out.data(), 0, n * sizeof(float));

    ggml_free(ctx);
    return true;
}

void add_graph_input(ggml_tensor* t, const void* host, size_t nbytes) {
    GGML_ASSERT(t_active != nullptr &&
                "add_graph_input called outside a Backend::compute build lambda");
    ggml_set_input(t);
    t_active->register_input(t, host, nbytes);
}

ggml_tensor* graph_input_tensor(ggml_context* ctx, int type, int n_dims,
                                const int64_t* ne, const void* host,
                                size_t nbytes) {
    ggml_tensor* t = ggml_new_tensor(ctx, (ggml_type)type, n_dims, ne);
    add_graph_input(t, host, nbytes);
    return t;
}

ggml_tensor* clone_weight(ggml_context* ctx, const ModelLoader& ml,
                          const char* name) {
    ggml_tensor* src = ml.tensor(name);
    assert(src && "missing tensor");
    const int nd = ggml_n_dims(src);
    int64_t ne[4] = {1, 1, 1, 1};
    for (int i = 0; i < nd; ++i) ne[i] = src->ne[i];
    return graph_input_tensor(ctx, (int)src->type, nd, ne, src->data,
                              ggml_nbytes(src));
}

ggml_tensor* clone_weight_opt(ggml_context* ctx, const ModelLoader& ml,
                              const char* name) {
    if (!ml.tensor(name)) return nullptr;
    return clone_weight(ctx, ml, name);
}

} // namespace pk
