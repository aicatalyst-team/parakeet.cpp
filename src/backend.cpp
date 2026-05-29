#include "backend.hpp"
#include "common.hpp"
#include "ggml_graph.hpp"   // pk::global_backend()
#include "model_loader.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
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
// Extra graph tensors to read back after compute (besides the final output).
// Used by the fused encoder's forward_capture to pull per-layer outputs out of
// the SAME single graph (vs computing each layer in its own graph). `dst` is the
// caller's vector, alive across the compute call.
struct PendingCapture {
    ggml_tensor*        tensor;
    std::vector<float>* dst;
};
} // namespace

struct Backend::Impl {
    ggml_backend_t    backend  = nullptr;
    ggml_gallocr_t    galloc   = nullptr;   // created lazily, reused across calls
    // Inputs registered by the build lambda for the IN-FLIGHT compute. Copied
    // into the gallocr-allocated tensors after ggml_gallocr_alloc_graph, then
    // cleared. Never overlaps across calls (compute is not re-entrant).
    std::vector<PendingInput> pending;
    // Extra tensors to read back after compute (registered via capture_output).
    std::vector<PendingCapture> captures;
};

// Thread-local pointer to the Backend whose compute() build lambda is currently
// executing, so the free helper add_graph_input() can route registrations
// without threading the Backend through every component's build lambda. compute
// is not re-entrant on a single thread (a build lambda never calls compute), so
// a single pointer is sufficient.
static thread_local Backend* t_active = nullptr;

Backend::Backend(int n_threads) : impl_(new Impl()) {
    // Optional override: PARAKEET_DEVICE=cpu forces the CPU backend (used to take
    // a CPU baseline on a GPU box without rebuilding).
    const char* force = std::getenv("PARAKEET_DEVICE");
    const bool force_cpu = force && std::string(force) == "cpu";

    if (!force_cpu) {
        // Pick the first GPU device the registry reports. Whatever backend was
        // compiled in (CUDA/Metal/Vulkan/HIP/SYCL) registers itself here, so this
        // single path covers them all with no backend-specific includes.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                impl_->backend = ggml_backend_dev_init(dev, nullptr);
                if (impl_->backend) {
                    device_name_ = ggml_backend_dev_name(dev);
                    PK_LOG("pk::Backend using GPU device: %s", device_name_.c_str());
                    break;
                }
            }
        }
    }
    if (!impl_->backend) {              // CPU fallback (or CPU-only build)
        impl_->backend = ggml_backend_cpu_init();
        device_name_ = "cpu";
    }
    if (!impl_->backend) {
        PK_LOG("backend init returned null");
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
    if (impl_ && impl_->backend && ggml_backend_is_cpu(impl_->backend)) {
        ggml_backend_cpu_set_n_threads(impl_->backend, n_threads_);
    }
}

ggml_backend_t Backend::handle() const {
    return impl_ ? impl_->backend : nullptr;
}

void Backend::register_input(ggml_tensor* t, const void* host, size_t nbytes) {
    impl_->pending.push_back({t, host, nbytes});
}

void Backend::register_capture(ggml_tensor* t, std::vector<float>* dst) {
    impl_->captures.push_back({t, dst});
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

    // Drive add_graph_input()/capture registrations to this Backend for the
    // build call.
    impl_->pending.clear();
    impl_->captures.clear();
    Backend* prev_active = t_active;
    t_active = this;
    struct ggml_tensor* output = build(ctx);
    t_active = prev_active;

    if (!output) {
        PK_LOG("Backend::compute: build() returned null output tensor");
        impl_->pending.clear();
        impl_->captures.clear();
        ggml_free(ctx);
        return false;
    }
    // Mark the output (and any captured tensors) so the gallocr does not recycle
    // their storage before we read them back, then expand the forward graph.
    ggml_set_output(output);
    for (const PendingCapture& pc : impl_->captures) ggml_set_output(pc.tensor);

    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);
    // Expand captures FIRST so they are present in the graph even if the final
    // output's subgraph does not reach them (it does here, but be robust).
    for (const PendingCapture& pc : impl_->captures)
        ggml_build_forward_expand(gf, pc.tensor);
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
            impl_->captures.clear();
            ggml_free(ctx);
            return false;
        }
    }
    if (!ggml_gallocr_alloc_graph(impl_->galloc, gf)) {
        PK_LOG("Backend::compute: ggml_gallocr_alloc_graph failed");
        impl_->pending.clear();
        impl_->captures.clear();
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
        impl_->captures.clear();
        ggml_free(ctx);
        return false;
    }

    // Read back any captured intermediates (per-layer outputs), then the final
    // output.
    for (const PendingCapture& pc : impl_->captures) {
        size_t cn = (size_t)ggml_nelements(pc.tensor);
        pc.dst->resize(cn);
        ggml_backend_tensor_get(pc.tensor, pc.dst->data(), 0, cn * sizeof(float));
    }
    impl_->captures.clear();

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

void capture_graph_output(ggml_tensor* t, std::vector<float>* dst) {
    GGML_ASSERT(t_active != nullptr &&
                "capture_graph_output called outside a Backend::compute build lambda");
    t_active->register_capture(t, dst);
}

void ensure_weights_realized(const ModelLoader& ml) {
    if (ml.weights_realized()) return;
    // realize_weights mutates tensor->buffer; the ModelLoader is held by `const`
    // ref throughout the inference path (the components are read-only views), but
    // realizing the backend buffer is a one-time, semantically-const setup of the
    // weight storage. Cast away const for that single call.
    ModelLoader& mut = const_cast<ModelLoader&>(ml);
    mut.realize_weights(global_backend().handle());
}

ggml_tensor* clone_weight(ggml_context* /*ctx*/, const ModelLoader& ml,
                          const char* name) {
    // Zero-copy: ensure the loader's weights have a backend buffer once, then
    // return the loader tensor DIRECTLY. It has ->data + ->buffer set, so the
    // gallocr treats it as already-allocated (never copies it) and the CPU
    // backend reads its bytes in place. Downstream reshapes/views of the
    // returned tensor resolve their data pointer at build time (the src has
    // data), so no per-call weight copy ever happens.
    ensure_weights_realized(ml);
    ggml_tensor* src = ml.tensor(name);
    assert(src && "missing tensor");
    return src;
}

ggml_tensor* clone_weight_opt(ggml_context* ctx, const ModelLoader& ml,
                              const char* name) {
    if (!ml.tensor(name)) return nullptr;
    return clone_weight(ctx, ml, name);
}

void weight_to_host_f32(const ModelLoader& ml, const char* name, std::vector<float>& out) {
    ensure_weights_realized(ml);
    ggml_tensor* t = ml.tensor(name);
    GGML_ASSERT(t && "weight_to_host_f32: missing tensor");
    GGML_ASSERT(t->type == GGML_TYPE_F32 && "weight_to_host_f32: tensor not f32");
    out.resize((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
}

} // namespace pk
