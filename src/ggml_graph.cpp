#include "ggml_graph.hpp"
#include "backend.hpp"
#include "common.hpp"
#include "ggml.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>

namespace pk {

// --- Temporary profiling (env PARAKEET_GRAPH_PROFILE=1): quantify per-call
// init/compute churn vs actual compute. With the persistent backend the
// "init+alloc" and "free" buckets should collapse (we no longer ggml_init a
// data context nor free a per-call compute buffer); only metadata-context
// init/alloc and graph build remain in the "init" bucket. Not committed long
// term (Task 5 removes it). ---
namespace {
std::atomic<long long> g_calls{0}, g_init_ns{0}, g_compute_ns{0}, g_free_ns{0};
struct ProfileDumper {
    ~ProfileDumper() {
        if (!std::getenv("PARAKEET_GRAPH_PROFILE")) return;
        long long c = g_calls.load(), in = g_init_ns.load(),
                  co = g_compute_ns.load(), fr = g_free_ns.load();
        if (c == 0) return;
        double tot = (in + co + fr) / 1e6;
        std::fprintf(stderr,
            "[graph-profile] run_graph calls=%lld  total=%.1fms  "
            "init+build=%.1fms(%.0f%%)  compute=%.1fms(%.0f%%)  readback=%.1fms(%.0f%%)\n",
            c, tot, in/1e6, 100.0*in/(in+co+fr), co/1e6, 100.0*co/(in+co+fr),
            fr/1e6, 100.0*fr/(in+co+fr));
    }
} g_profile_dumper;
inline bool prof() { static bool on = std::getenv("PARAKEET_GRAPH_PROFILE") != nullptr; return on; }
using clk = std::chrono::steady_clock;
inline long long ns(clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}
} // namespace

// Process-global compute-thread override. 0 == unset (use the default thread
// count). Atomic so it is safe to set from one thread and read from the compute
// path; the benchmark CLI sets it once before any inference.
static std::atomic<int> g_num_threads{0};

void set_num_threads(int n) { g_num_threads.store(n < 0 ? 0 : n, std::memory_order_relaxed); }
int  num_threads()          { return g_num_threads.load(std::memory_order_relaxed); }

namespace {
// Process-global persistent backend. Created on first use with the configured
// thread count and reused across every graph computation. Holding it here (vs.
// per-Model) keeps the migration minimal: the components call pk::run_graph,
// which routes through this single Backend, so the per-call ggml_init/ggml_free
// churn is eliminated without threading a Backend handle through every layer.
// Task 2+ will fuse graphs on top of the same Backend.
std::once_flag g_backend_once;
std::unique_ptr<Backend> g_backend;
int g_backend_threads = 0;
// Serializes access to the process-global Backend. The old run_graph allocated
// a fresh per-call ggml context (so concurrent transcribes were independent);
// the shared Backend (one gallocr + pending-input list, not re-entrant) is not,
// so we serialize compute() across threads. In practice inference is driven
// from a single thread per process (the parallelism is inside the graph's
// worker threads), so this lock is uncontended; it only guards against a caller
// that drives transcribe() concurrently from multiple threads.
std::mutex g_backend_mutex;
} // namespace

Backend& global_backend() {
    std::call_once(g_backend_once, [] {
        const int g = g_num_threads.load(std::memory_order_relaxed);
        const int n = g > 0 ? g : 4;  // 4 matches the historical per-call default
        g_backend = std::make_unique<Backend>(n);
        g_backend_threads = n;
    });
    // Honor a late --threads override: keep the live backend's thread count in
    // sync with the global override (the override is set once before inference).
    const int g = g_num_threads.load(std::memory_order_relaxed);
    if (g > 0 && g != g_backend_threads) {
        g_backend->set_n_threads(g);
        g_backend_threads = g;
    }
    return *g_backend;
}

bool run_graph(size_t /*mem_bytes*/, int n_threads,
               const std::function<ggml_tensor*(ggml_context*)>& build,
               std::vector<float>& out) {
    std::lock_guard<std::mutex> lock(g_backend_mutex);
    Backend& be = global_backend();
    // When no global override is set, honor the caller's per-call n_threads (the
    // historical behavior, used by the unit tests). A positive global override
    // already pinned the backend's thread count in global_backend().
    const int g = g_num_threads.load(std::memory_order_relaxed);
    if (g <= 0 && n_threads > 0 && n_threads != g_backend_threads) {
        be.set_n_threads(n_threads);
        g_backend_threads = n_threads;
    }

    const bool p = prof();
    clk::time_point t0 = p ? clk::now() : clk::time_point{};
    // Backend::compute builds the graph in a no_alloc context, allocates via the
    // persistent gallocr, pushes inputs AFTER alloc, computes, and reads back.
    bool ok = be.compute(build, out);
    if (p) {
        clk::time_point t1 = clk::now();
        g_calls.fetch_add(1, std::memory_order_relaxed);
        // We no longer split init/compute/free inside compute(); attribute the
        // whole call to "compute" since the per-call init/free churn is gone.
        g_compute_ns.fetch_add(ns(t0, t1), std::memory_order_relaxed);
    }
    return ok;
}

} // namespace pk
