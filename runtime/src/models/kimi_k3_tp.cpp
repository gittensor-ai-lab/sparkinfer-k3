// Tensor-parallel Kimi K3 decode. See kimi_k3_tp.h for the scope and the reasoning.

#include "sparkinfer/models/kimi_k3_tp.h"

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/tp/shard.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <thread>
#include <utility>

namespace k3k = sparkinfer::kernels::k3;

namespace sparkinfer {

namespace {

// ---------------------------------------------------------------------------
// Host-issue accounting (SPARKINFER_K3_ISSUE_PROFILE=1)
// ---------------------------------------------------------------------------
// Why this exists, and why it is shaped this way:
//
// A single host thread issues every rank's kernels, rank 0 first and rank 7
// last, and 92 all-reduces per token act as hard barriers. If issuing a layer
// costs the host a meaningful fraction of what the layer costs the GPU, then
// rank 7 arrives at every barrier structurally late and the collective's cost
// is not the transfer at all — it is the host that has not finished asking yet.
//
// Everything measured below is ASYNCHRONOUS work-submission. That is the whole
// point: no cudaEventRecord, no cudaEventSynchronize, no extra stream sync.
// Instrumenting this path with CUDA events would sync the very thing being
// measured (SPARKINFER_K3_PROFILE=1 does exactly that and eliminates the fast
// mode on this box), so this uses steady_clock only — ~20 ns per sample against
// microsecond-scale intervals, and it cannot change the GPU's schedule.
//
// t_issue  host time submitting the per-rank layer phases (async enqueue)
// t_coll   host time inside the group all-reduce call (async enqueue)
// t_sync   host time blocked in the ONE end-of-token sync -> GPU catch-up
// t_total  wall time of the whole forward_token
//
// t_issue >> 0 and t_sync ~ 0 means the GPU is starved and the host is the
// bottleneck. The reverse means the host stays ahead and the GPU is the wall.
struct IssueProfile {
    bool on = false;
    long long n_tokens = 0;
    double t_issue = 0, t_coll = 0, t_sync = 0, t_total = 0;
    long long n_phase_calls = 0, n_setdev = 0;
};

IssueProfile& issue_profile() {
    static IssueProfile p = [] {
        IssueProfile q;
        const char* e = std::getenv("SPARKINFER_K3_ISSUE_PROFILE");
        q.on = e && e[0] == '1';
        return q;
    }();
    return p;
}

using IClock = std::chrono::steady_clock;
inline double secs_since(IClock::time_point t0) {
    return std::chrono::duration<double>(IClock::now() - t0).count();
}

// Bounded spin. Pause for cache-friendly waiting, then yield so a box that is
// oversubscribed (or a rank that died) degrades to scheduling instead of
// burning a core forever.
inline void spin_step(int& n) {
    if (++n < 4096) {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
    } else {
        std::this_thread::yield();
        n = 4096;
    }
}

// Token row gather. Same contract as the single-GPU path: token_embd is replicated,
// so every rank does this identically and no broadcast is needed.
bool embed_token(const KimiK3Weights& w, const KimiK3Config& cfg, int token_id,
                 float* x, cudaStream_t stream) {
    if (!w.token_embd.ok()) return false;
    long row_bytes = 0;
    if (w.token_embd.type == 0)      row_bytes = (long)cfg.hidden * sizeof(float);
    else if (w.token_embd.type == 8) row_bytes = (long)(cfg.hidden / 32) * 34;
    else return false;
    const char* base = (const char*)w.token_embd.data + (size_t)token_id * row_bytes;
    return k3k::dequant_f32_by_type(x, base, cfg.hidden, w.token_embd.type, stream);
}

}  // namespace

// ---------------------------------------------------------------------------
// K3IssuePool
// ---------------------------------------------------------------------------

void K3IssuePool::start(const std::vector<int>& devices) {
    if (!workers_.empty()) return;
    devices_ = devices;
    stop_.store(false, std::memory_order_relaxed);
    epoch_.store(0, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);

    const int n = (int)devices_.size();
    workers_.reserve((size_t)n);
    for (int r = 0; r < n; ++r) {
        workers_.emplace_back([this, r] {
            // ONCE per thread, not once per launch. The current device is
            // thread-local, so this is the whole reason the 1496 per-token
            // cudaSetDevice calls disappear rather than merely moving.
            cudaSetDevice(devices_[(size_t)r]);
            unsigned seen = 0;
            for (;;) {
                int spins = 0;
                while (epoch_.load(std::memory_order_acquire) == seen) {
                    if (stop_.load(std::memory_order_acquire)) return;
                    spin_step(spins);
                }
                seen = epoch_.load(std::memory_order_acquire);
                if (stop_.load(std::memory_order_acquire)) return;

                // A throwing job must not strand the barrier: the done_ count
                // is what the submitter waits on, so it is incremented on every
                // path out of the call.
                bool ok = false;
                try {
                    ok = (*job_)(r);
                } catch (...) {
                    ok = false;
                }
                if (!ok) failed_.store(true, std::memory_order_relaxed);
                done_.fetch_add(1, std::memory_order_release);
            }
        });
    }
}

bool K3IssuePool::run(const std::function<bool(int)>& job) {
    const int n = (int)workers_.size();
    if (n == 0) return false;
    job_ = &job;
    failed_.store(false, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    epoch_.fetch_add(1, std::memory_order_release);

    int spins = 0;
    while (done_.load(std::memory_order_acquire) < n) spin_step(spins);
    job_ = nullptr;
    return !failed_.load(std::memory_order_relaxed);
}

void K3IssuePool::shutdown() {
    if (workers_.empty()) return;
    stop_.store(true, std::memory_order_release);
    epoch_.fetch_add(1, std::memory_order_release);   // wake the parked spinners
    for (auto& t : workers_) if (t.joinable()) t.join();
    workers_.clear();
}

bool kimi_k3_tp_init(const GGUF& g, const KimiK3Config& cfg, const K3PlanOptions& opt,
                    const std::vector<int>& devices, int max_ctx, KimiK3TP& out) {
    const int tp_size = (int)devices.size();
    if (tp_size < 1) return false;
    if (tp_size > 1 && (cfg.n_experts % tp_size) != 0) {
        std::fprintf(stderr, "[k3-tp] %d experts do not divide %d ranks — a ragged "
                             "expert band would give every rank a different launch "
                             "geometry\n", cfg.n_experts, tp_size);
        return false;
    }

    out.cfg = cfg;
    out.opt = opt;
    out.ranks.clear();
    out.ranks.resize((size_t)tp_size);
    out.streams.assign((size_t)tp_size, nullptr);
    out.reduce_bufs.assign((size_t)tp_size, nullptr);
    out.n_collectives = 0;

    // LOAD THE RANKS IN PARALLEL.
    //
    // This loop used to be strictly serial, and the load is the dominant cost of every run:
    // ~150 s of a decode benchmark that then measures 8 tokens. Under ExpertsOnly each rank
    // takes all the replicated tensors plus its own 1/8 expert band -- about 124 GiB on this
    // model -- so eight ranks push ~992 GiB host-to-device, one after another, through
    // synchronous cudaMemcpy from the pageable mmap. That stages through the driver at a
    // few GB/s, which is exactly the 125-150 s observed.
    //
    // The ranks are independent: GGUF is a const mmap read by everyone, each rank writes its
    // own pre-sized slot, and every device has its own PCIe path. cudaSetDevice is per-THREAD
    // state in CUDA, so a thread per rank needs no other coordination -- which is also the
    // trap: the device must be set INSIDE the worker, never inherited from the caller.
    //
    // Deliberately not touched here: the copies are still synchronous and still from pageable
    // memory. Pinned staging would buy another 2-4x on top, and is a separate change with its
    // own failure mode (pinning tens of GiB can starve the host).
    std::vector<std::string> rank_err((size_t)tp_size);
    std::vector<std::string> rank_log((size_t)tp_size);

    auto load_rank = [&](int r) -> bool {
        KimiK3TPRank& R = out.ranks[(size_t)r];
        R.device = devices[(size_t)r];
        R.rank = r;
        if (cudaSetDevice(R.device) != cudaSuccess) {
            rank_err[(size_t)r] = "cudaSetDevice failed";
            return false;
        }
        if (cudaStreamCreate(&R.stream) != cudaSuccess) {
            rank_err[(size_t)r] = "cudaStreamCreate failed";
            return false;
        }
        out.streams[(size_t)r] = R.stream;

        // Which attention band to shard, and why it is MLA rather than KDA.
        //
        // Measured on the shipped build (nsys, ctx 131072), as a share of GPU
        // kernel time: MLA attention 36.9% over 24 layers, KDA 6.6% over 69.
        // Both cost one all-reduce per layer at ~0.088% each, so:
        //
        //     shard KDA:  +5.8% compute, -6.1% collectives  =  -0.3%  REGRESSION
        //     shard MLA: +32.3% compute, -2.1% collectives  = +30.2%
        //
        // Sharding the 69 KDA layers costs more in collectives than it saves in
        // compute. SPARKINFER_K3_SHARD_KDA=1 keeps that branch reachable because
        // the comparison is the evidence for this choice, but it stays OFF.
        static const bool shard_kda = [] {
            const char* e = std::getenv("SPARKINFER_K3_SHARD_KDA");
            return e && e[0] == '1';
        }();
        static const bool shard_mla = [] {
            const char* e = std::getenv("SPARKINFER_K3_SHARD_MLA");
            return !(e && e[0] == '0');          // ON unless explicitly disabled
        }();
        R.weights.policy =
            (shard_kda && tp_size > 1) ? KimiK3Weights::ShardPolicy::ExpertsAndKda :
            (shard_mla && tp_size > 1) ? KimiK3Weights::ShardPolicy::ExpertsAndMla :
                                         KimiK3Weights::ShardPolicy::ExpertsOnly;
        R.weights.shard.tp_size = tp_size;
        R.weights.shard.rank = r;
        R.weights.shard.hidden = cfg.hidden;
        R.weights.shard.n_experts_total = cfg.n_experts;
        R.weights.shard.experts_sharded = tp_size > 1;
        R.weights.shard.n_experts = cfg.n_experts / tp_size;
        R.weights.shard.expert_band = tp_size > 1
            ? tp::even_band(cfg.n_experts, tp_size, r)
            : tp::Band{0, cfg.n_experts};

        if (!kimi_k3_load_weights(g, cfg, opt, R.weights, 0, cfg.n_layers - 1)) {
            rank_err[(size_t)r] = "weight load failed";
            return false;
        }
        // Sized to the slice the weights were just cut with. Passing the weights'
        // own KdaShardDims rather than re-deriving it is what makes it impossible
        // for the state and the projections to disagree about which heads this rank
        // owns -- the failure that runs at full speed and emits wrong tokens.
        if (!kimi_k3_alloc_state(cfg, max_ctx, R.state, -1, -1, &R.weights.kda)) {
            rank_err[(size_t)r] = "alloc_state failed";
            return false;
        }

        R.fwd.cfg = &out.cfg;
        R.fwd.w = &R.weights;
        R.fwd.state = &R.state;
        R.fwd.opt = out.opt;
        R.fwd.stream = R.stream;
        if (!kimi_k3_forward_alloc_scratch(cfg, R.fwd)) {
            rank_err[(size_t)r] = "forward_alloc_scratch failed";
            return false;
        }

        if (cudaMalloc(&R.x, (size_t)cfg.hidden * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&R.x_next, (size_t)cfg.hidden * sizeof(float)) != cudaSuccess) {
            rank_err[(size_t)r] = "cudaMalloc x/x_next failed";
            return false;
        }
        if (r == 0 &&
            cudaMalloc(&R.logits, (size_t)cfg.vocab * sizeof(float)) != cudaSuccess) {
            rank_err[(size_t)r] = "cudaMalloc logits failed";
            return false;
        }

        // Buffered, not printed: eight ranks writing stderr concurrently interleaves
        // mid-line. Emitted in rank order after the join so the log reads as before.
        char line[160];
        std::snprintf(line, sizeof(line), "[k3-tp] rank %d: device %d, experts [%d,%d)\n",
                      r, R.device, R.weights.shard.expert_band.offset,
                      R.weights.shard.expert_band.end());
        rank_log[(size_t)r] = line;
        return true;
    };

    // SPARKINFER_TP_SERIAL_LOAD forces the old path. A concurrency bug here looks like a
    // corrupt weight tensor rather than a crash, so being able to bisect it without a
    // rebuild is worth one branch.
    const bool serial_load = std::getenv("SPARKINFER_TP_SERIAL_LOAD") != nullptr || tp_size == 1;
    if (serial_load) {
        for (int r = 0; r < tp_size; ++r) {
            if (!load_rank(r)) {
                std::fprintf(stderr, "[k3-tp] rank %d: %s\n", r, rank_err[(size_t)r].c_str());
                return false;
            }
        }
    } else {
        std::vector<std::thread> workers;
        workers.reserve((size_t)tp_size);
        for (int r = 0; r < tp_size; ++r) workers.emplace_back(load_rank, r);
        for (auto& t : workers) t.join();
    }
    for (int r = 0; r < tp_size; ++r) {
        if (!rank_log[(size_t)r].empty()) std::fputs(rank_log[(size_t)r].c_str(), stderr);
    }
    // Report EVERY failed rank, not just the first. With eight concurrent loads the
    // interesting case is "ranks 4-7 ran out of memory", and stopping at the first hides
    // whether one device is sick or the whole box is short.
    bool any_failed = false;
    for (int r = 0; r < tp_size; ++r) {
        if (!rank_err[(size_t)r].empty()) {
            std::fprintf(stderr, "[k3-tp] rank %d: %s\n", r, rank_err[(size_t)r].c_str());
            any_failed = true;
        }
    }
    if (any_failed) return false;

    // The caller's device is thread-local and the workers set their own, so restore an
    // explicit one before the collective and everything after it.
    if (cudaSetDevice(devices[0]) != cudaSuccess) return false;

    // The collective. need_f32 because K3's residual stream is f32; max_count is the
    // widest payload the forward will reduce, which is the expert_latent partial.
    std::string requested_env;
    if (const char* e = std::getenv("SPARKINFER_TP_BACKEND")) requested_env = e;
    std::string why;
    const tp::Backend requested = tp::backend_from_string(requested_env, &why);
    if (!why.empty()) std::fprintf(stderr, "[k3-tp] %s\n", why.c_str());

    std::string err;
    out.coll = tp::make_collective(devices, requested, &err,
                                   // hidden (7168) > expert_latent (3584): the KDA
                                   // reduce is the wider payload, and a collective
                                   // sized to the narrower one would truncate it.
                                   /*max_count=*/(size_t)(cfg.hidden > cfg.expert_latent
                                                          ? cfg.hidden : cfg.expert_latent),
                                   /*need_f32=*/true);
    if (!out.coll) {
        std::fprintf(stderr, "[k3-tp] no collective: %s\n", err.c_str());
        return false;
    }
    if (tp_size > 1 && out.coll->backend() == tp::Backend::None) {
        // A no-op collective at tp>1 leaves every rank holding its own expert band's
        // partial sum and calls it the answer. Refuse rather than emit fluent noise.
        std::fprintf(stderr, "[k3-tp] FATAL: fell back to the no-op collective at "
                             "tp_size %d: %s\n", tp_size, err.c_str());
        return false;
    }
    if (!out.coll->supports_f32()) {
        std::fprintf(stderr, "[k3-tp] FATAL: backend %s cannot reduce f32\n",
                     tp::backend_name(out.coll->backend()));
        return false;
    }
    if (out.coll->owns_buffers()) {
        // Mode B needs the caller to stage through reduce_in/reduce_out. The forward
        // reduces a buffer the executor owns, so that path would need a copy in and
        // out per collective — not wired, and silently staging it would erase the
        // reason to use those backends. need_f32 already steers away from them.
        std::fprintf(stderr, "[k3-tp] FATAL: %s uses collective-owned buffers, which "
                             "this forward does not stage through\n",
                     tp::backend_name(out.coll->backend()));
        return false;
    }
    return true;
}

bool kimi_k3_tp_forward_token(KimiK3TP& p, int token_id, float* out_logits) {
    if (p.ranks.empty() || !p.coll) return false;
    const KimiK3Config& cfg = p.cfg;
    const int H = cfg.hidden;
    const int tp_size = (int)p.ranks.size();

    IssueProfile& ip = issue_profile();
    const IClock::time_point t_tok0 = ip.on ? IClock::now() : IClock::time_point{};

    // Parallel submission. Off at tp_size 1 (nothing to parallelise, and the
    // single-rank path must stay identical) and off under SPARKINFER_K3_SERIAL_ISSUE=1,
    // which is the A/B control: same binary, same kernels, only the submission
    // order changes. Any measured delta is therefore the submission, not a rebuild.
    static const bool serial_issue = [] {
        const char* e = std::getenv("SPARKINFER_K3_SERIAL_ISSUE");
        return e && e[0] == '1';
    }();
    const bool parallel_issue = (tp_size > 1) && !serial_issue;
    if (parallel_issue && !p.issue.started()) {
        std::vector<int> devs;
        devs.reserve(p.ranks.size());
        for (auto& R : p.ranks) devs.push_back(R.device);
        p.issue.start(devs);
    }

    // Submit `job` on every rank — concurrently when the pool is up, otherwise
    // in rank order exactly as before. The serial arm keeps its per-call
    // cudaSetDevice; the parallel arm does not need one, because each worker
    // pinned its device once at thread start.
    auto issue_all = [&](const std::function<bool(int)>& job) -> bool {
        if (parallel_issue) return p.issue.run(job);
        for (int r = 0; r < tp_size; ++r) {
            if (cudaSetDevice(p.ranks[(size_t)r].device) != cudaSuccess) return false;
            if (!job(r)) return false;
        }
        return true;
    };

    // The cross-layer residual bank is PER TOKEN on every rank — same lifetime rule
    // as the single-GPU path, and forgetting it here fails on token 2 rather than
    // token 1, because max_ckpt is exactly one token's worth of checkpoints.
    for (auto& R : p.ranks) R.state.n_ckpt = 0;

    {
        const IClock::time_point t0 = ip.on ? IClock::now() : IClock::time_point{};
        if (!issue_all([&](int r) {
                KimiK3TPRank& R = p.ranks[(size_t)r];
                return embed_token(R.weights, cfg, token_id, R.x, R.stream);
            })) return false;
        if (ip.on) {
            ip.t_issue += secs_since(t0);
            if (!parallel_issue) ip.n_setdev += tp_size;
        }
    }

    for (int layer = 0; layer < cfg.n_layers; ++layer) {
        const bool is_moe = layer >= cfg.leading_dense;

        // --- phase 1 + 2 on every rank -------------------------------------
        // Under ExpertsOnly, attention is replicated, so no collective separates
        // these two phases. They are still issued as distinct calls because the
        // phase boundary is where a fully-sharded build WOULD reduce, and keeping
        // the call sites is what makes that a one-line change rather than a
        // re-split of the forward.
        // Under ExpertsAndKda a KDA layer's attn_output is COL-sharded, so every rank
        // holds a full-width partial sum of the attention output and the two phases
        // are no longer separable -- the reduce has to land between them. This is the
        // boundary the forward already exposes; kimi_k3_partial_buffer(Attn) returns
        // s.attn_out at cfg.hidden, which is exactly the partial.
        //
        // When the KDA attention is NOT sharded the two phases are still issued as
        // one job, so the replicated path keeps its single barrier per layer and
        // pays nothing for a split it does not need.
        // Same seam, the other band: under ExpertsAndMla an MLA layer's
        // attn_output is COL-sharded, so its partial is full-width at hidden and
        // has to be summed before ffn_norm — rms_norm is not linear and cannot be
        // applied to a partial sum. Exactly one of these can be true, because the
        // two policies are mutually exclusive and a layer is either KDA or MLA.
        const bool kda_reduce = tp_size > 1 && cfg.is_kda_layer(layer) &&
            p.ranks[0].weights.policy == KimiK3Weights::ShardPolicy::ExpertsAndKda;
        const bool mla_reduce = tp_size > 1 && !cfg.is_kda_layer(layer) &&
            p.ranks[0].weights.policy == KimiK3Weights::ShardPolicy::ExpertsAndMla;
        const bool attn_reduce = kda_reduce || mla_reduce;

        const IClock::time_point t_p12 = ip.on ? IClock::now() : IClock::time_point{};
        if (!issue_all([&](int r) {
                KimiK3TPRank& R = p.ranks[(size_t)r];
                if (!kimi_k3_forward_layer_phase(R.fwd, layer, K3LayerPhase::Attn,
                                                 R.x, R.x_next)) return false;
                if (attn_reduce) return true;  // FfnPartial waits for the reduce
                return kimi_k3_forward_layer_phase(R.fwd, layer, K3LayerPhase::FfnPartial,
                                                   R.x, R.x_next);
            })) return false;
        // Closed HERE so t_issue never spans the reduce below; the second issue
        // adds its own span. Letting one bracket cover both would book collective
        // time as submission time and quietly inflate exactly the number this
        // instrumentation exists to keep honest.
        if (ip.on) {
            ip.t_issue += secs_since(t_p12);
            ip.n_phase_calls += (attn_reduce ? 1 : 2) * tp_size;
            if (!parallel_issue) ip.n_setdev += tp_size;
        }

        if (attn_reduce) {
            int count = 0;
            for (int r = 0; r < tp_size; ++r) {
                int n = 0;
                float* buf = kimi_k3_partial_buffer(p.ranks[(size_t)r].fwd, layer,
                                                    K3LayerPhase::Attn, &n);
                if (!buf || n <= 0) return false;
                if (r == 0) count = n;
                else if (n != count) return false;
                p.reduce_bufs[(size_t)r] = buf;
            }
            const IClock::time_point tk = ip.on ? IClock::now() : IClock::time_point{};
            const bool okk = p.coll->allreduce_f32_group(p.reduce_bufs, (size_t)count,
                                                         p.streams);
            if (ip.on) ip.t_coll += secs_since(tk);
            if (!okk) {
                std::fprintf(stderr, "[k3-tp] attention all-reduce failed at layer %d\n", layer);
                return false;
            }
            ++p.n_collectives;

            const IClock::time_point t_fp = ip.on ? IClock::now() : IClock::time_point{};
            if (!issue_all([&](int r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    return kimi_k3_forward_layer_phase(R.fwd, layer,
                                                       K3LayerPhase::FfnPartial,
                                                       R.x, R.x_next);
                })) return false;
            if (ip.on) {
                ip.t_issue += secs_since(t_fp);
                ip.n_phase_calls += tp_size;
                if (!parallel_issue) ip.n_setdev += tp_size;
            }
        }

        // --- the collective -------------------------------------------------
        // MoE layers only. The leading dense layer's FFN is replicated under this
        // policy, so it holds a complete result already and reducing it would
        // multiply it by tp_size.
        if (is_moe && tp_size > 1) {
            int count = 0;
            for (int r = 0; r < tp_size; ++r) {
                int n = 0;
                float* buf = kimi_k3_partial_buffer(p.ranks[(size_t)r].fwd, layer,
                                                    K3LayerPhase::FfnPartial, &n);
                if (!buf || n <= 0) return false;
                if (r == 0) count = n;
                else if (n != count) return false;   // ranks must agree on the width
                p.reduce_bufs[(size_t)r] = buf;
            }
            // Enqueued on each rank's own stream, after that rank's dispatch, so the
            // ordering is stream-ordered — no host barrier, and no cudaDeviceSynchronize
            // on the critical path. The GROUP form is mandatory: a single host thread
            // looping per-rank allreduce deadlocks (see collective.h).
            // SPARKINFER_K3_TP_HOST_REDUCE=1 replaces the collective with an explicit
            // device->host->sum->device round trip. Slow and only for bisecting: it
            // isolates "the reduce is wrong" from "the reduce is right but something
            // around it is", which are otherwise indistinguishable from the logits.
            static const bool host_reduce = [] {
                const char* e = std::getenv("SPARKINFER_K3_TP_HOST_REDUCE");
                return e && e[0] == '1';
            }();
            if (host_reduce) {
                std::vector<float> sum((size_t)count, 0.0f), tmp((size_t)count);
                for (int r = 0; r < tp_size; ++r) {
                    cudaSetDevice(p.ranks[(size_t)r].device);
                    cudaStreamSynchronize(p.ranks[(size_t)r].stream);
                    cudaMemcpy(tmp.data(), p.reduce_bufs[(size_t)r],
                               (size_t)count * sizeof(float), cudaMemcpyDeviceToHost);
                    for (int i = 0; i < count; ++i) sum[(size_t)i] += tmp[(size_t)i];
                }
                for (int r = 0; r < tp_size; ++r) {
                    cudaSetDevice(p.ranks[(size_t)r].device);
                    cudaMemcpy(p.reduce_bufs[(size_t)r], sum.data(),
                               (size_t)count * sizeof(float), cudaMemcpyHostToDevice);
                }
            } else {
                const IClock::time_point tc = ip.on ? IClock::now() : IClock::time_point{};
                const bool okc = p.coll->allreduce_f32_group(p.reduce_bufs, (size_t)count,
                                                             p.streams);
                if (ip.on) ip.t_coll += secs_since(tc);
                if (!okc) {
                    std::fprintf(stderr, "[k3-tp] all-reduce failed at layer %d\n", layer);
                    return false;
                }
            }
            ++p.n_collectives;
        }

        // --- phase 3 on every rank ------------------------------------------
        const IClock::time_point t_p3 = ip.on ? IClock::now() : IClock::time_point{};
        if (!issue_all([&](int r) {
                KimiK3TPRank& R = p.ranks[(size_t)r];
                if (!kimi_k3_forward_layer_phase(R.fwd, layer, K3LayerPhase::FfnFinish,
                                                 R.x, R.x_next)) return false;
                // Each worker swaps only its OWN rank's pair; no rank reads
                // another's x, so this needs no synchronisation beyond the
                // barrier that ends the phase.
                std::swap(R.x, R.x_next);
                return true;
            })) return false;
        if (ip.on) {
            ip.t_issue += secs_since(t_p3);
            ip.n_phase_calls += tp_size;
            if (!parallel_issue) ip.n_setdev += tp_size;
        }
    }

    // --- head: every rank holds the identical hidden state, so rank 0 suffices ---
    KimiK3TPRank& R0 = p.ranks[0];
    if (cudaSetDevice(R0.device) != cudaSuccess) return false;
    const KimiK3Weights& w = R0.weights;

    if (cfg.attn_res_block_size > 0) {
        if (!w.has_output_res_score || !w.output_res_score.ok()) return false;
        k3k::attn_res_mix_f32(R0.x_next, R0.state.res_bank, R0.x,
                              (const float*)w.output_res_score.data, H,
                              R0.state.n_ckpt, cfg.rms_eps, R0.stream);
        std::swap(R0.x, R0.x_next);
    }
    if (!w.output_norm.ok()) return false;
    k3k::rms_norm_f32(R0.x_next, R0.x, (const float*)w.output_norm.data, H,
                      cfg.rms_eps, R0.stream);
    if (!w.output.ok()) return false;
    if (!k3k::k3_proj_f32(R0.logits, R0.x_next, w.output.data, w.output.type,
                          cfg.vocab, H, R0.stream))
        return false;

    const IClock::time_point t_s0 = ip.on ? IClock::now() : IClock::time_point{};
    if (cudaStreamSynchronize(R0.stream) != cudaSuccess) return false;
    if (ip.on) ip.t_sync += secs_since(t_s0);
    if (cudaMemcpy(out_logits, R0.logits, (size_t)cfg.vocab * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;

    if (ip.on) {
        ip.t_total += secs_since(t_tok0);
        ++ip.n_tokens;
        // Report every token: the run is 32 tokens and the per-token variation is
        // itself the signal (a host-bound loop is steady, a GPU-bound one is not).
        const double n = (double)ip.n_tokens;
        std::fprintf(stderr,
            "[k3-issue] tok=%lld  total=%.2f ms  issue=%.2f ms (%.1f%%)  "
            "coll=%.2f ms (%.1f%%)  sync=%.2f ms (%.1f%%)  "
            "| phase_calls/tok=%.0f setdev/tok=%.0f\n",
            ip.n_tokens,
            1e3 * ip.t_total / n,
            1e3 * ip.t_issue / n, 100.0 * ip.t_issue / ip.t_total,
            1e3 * ip.t_coll  / n, 100.0 * ip.t_coll  / ip.t_total,
            1e3 * ip.t_sync  / n, 100.0 * ip.t_sync  / ip.t_total,
            (double)ip.n_phase_calls / n, (double)ip.n_setdev / n);
    }

    // Advance every rank's position together. They run the same attention, so a rank
    // whose position drifted would index a different KV row for the same token.
    for (auto& R : p.ranks) ++R.state.position;
    return true;
}

void kimi_k3_tp_free(KimiK3TP& p) {
    // Join the submission threads BEFORE any rank buffer is freed. They only
    // spin while parked, but a worker that is still inside a phase call would
    // be launching against memory this loop is about to release.
    p.issue.shutdown();
    for (auto& R : p.ranks) {
        cudaSetDevice(R.device);
        if (R.x) cudaFree(R.x);
        if (R.x_next) cudaFree(R.x_next);
        if (R.logits) cudaFree(R.logits);
        kimi_k3_forward_free_scratch(R.fwd);
        kimi_k3_free_state(R.state);
        kimi_k3_free_weights(R.weights);
        if (R.stream) cudaStreamDestroy(R.stream);
    }
    p.ranks.clear();
    p.streams.clear();
    p.reduce_bufs.clear();
    p.coll.reset();
}

}  // namespace sparkinfer
