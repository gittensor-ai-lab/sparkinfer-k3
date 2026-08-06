// Tensor-parallel Kimi K3 decode. See kimi_k3_tp.h for the scope and the reasoning.

#include "sparkinfer/models/kimi_k3_tp.h"

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/models/k3_head_band.h"
#include "sparkinfer/models/kimi_k3_prefill.h"
#include "sparkinfer/tp/k3_coll_1bar.h"
#include "sparkinfer/tp/shard.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
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

// The chunk size for kimi_k3_tp_forward_prompt. Read HERE rather than at the use
// site because kimi_k3_tp_init has to size the collective's owned buffers against
// it: a chunk reduces B partials in one call, and a collective sized for one
// token's 10752 floats fails the very first reduce — after a two-minute weight
// load, which is the most expensive possible place to discover it.
int prefill_batch_env() {
    static const int v = [] {
        const char* e = std::getenv("SPARKINFER_K3_PREFILL_BATCH");
        const int x = e ? std::atoi(e) : 1;
        return x > 0 ? x : 1;
    }();
    return v;
}

// The tile width, read here for the SAME reason as the chunk size above: the tile
// driver now reduces a whole tile's partials in one call, so the collective has to be
// sized for T tokens at init or the first reduce fails after the weight load.
//
// Default 4 mirrors the use site. The two are read independently and the collective is
// sized to whichever is larger, because one binary serves both drivers.
int prefill_tile_env() {
    static const int v = [] {
        const char* e = std::getenv("SPARKINFER_K3_PREFILL_TILE");
        const int x = e ? std::atoi(e) : 4;
        return x > 0 ? x : 4;
    }();
    return v;
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
    // Why a rank fell back from the 2-D MoE default to whole-expert sharding. Buffered
    // like rank_log so eight ranks do not interleave mid-line, and reported rather than
    // swallowed: a silent fallback is a silent perf cliff.
    std::vector<std::string> moe_2d_note((size_t)tp_size);

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

        // Which attention bands to shard.
        //
        // MLA first, because it is where the time is: profiled at ctx 131072,
        // MLA attention is 36.9% of GPU kernel time over 24 layers against KDA's
        // 6.6% over 69 — 16x more per layer.
        //
        // KDA WAS OFF HERE, ON AN ARITHMETIC THAT WAS WRONG. The estimate priced
        // a collective at ~0.088% of GPU time and concluded:
        //
        //     shard KDA: +5.8% compute, -6.1% collectives = -0.3%, a regression
        //
        // That per-collective figure came from a profile whose NCCL time was
        // inflated by rank-arrival skew — 35 ms max against a 71 us mean — i.e.
        // it priced the stall, not the reduce. With submission parallelised the
        // stall is gone and a reduce costs a fraction of that, so the 5.8% is
        // worth its 69 extra collectives.
        //
        // What settled it was measuring the competitor rather than arguing with
        // them: PR #64 shards all 93 layers and runs 63.6 ms/token on this box,
        // against 70.3 for the MLA-only build here. Sharding both bands is the
        // better trade once the skew that made collectives look expensive is
        // fixed — and this tree is the one that fixed it.
        static const bool shard_kda = [] {
            const char* e = std::getenv("SPARKINFER_K3_SHARD_KDA");
            return !(e && e[0] == '0');          // ON unless explicitly disabled
        }();
        static const bool shard_mla = [] {
            const char* e = std::getenv("SPARKINFER_K3_SHARD_MLA");
            return !(e && e[0] == '0');          // ON unless explicitly disabled
        }();
        const bool k = shard_kda && tp_size > 1;
        const bool m = shard_mla && tp_size > 1;
        R.weights.policy =
            (k && m) ? KimiK3Weights::ShardPolicy::ExpertsAndAttn :
            k        ? KimiK3Weights::ShardPolicy::ExpertsAndKda  :
            m        ? KimiK3Weights::ShardPolicy::ExpertsAndMla  :
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
        R.weights.shard.moe_ffn_total = cfg.moe_ffn;

        // 2-D MoE: SPARKINFER_K3_MOE_2D = number of EXPERT GROUPS. Read once per rank
        // here so the weights, the state and the kernels all derive their geometry
        // from this single ShardDims — see moe_2d_dims.
        //
        // ON BY DEFAULT, AND THAT IS THE POINT. As an opt-in env flag this was
        // unmeasurable: the eval builds the tree and runs the bench, it does not set
        // anything, so it scored the whole-expert path and reported the change as
        // 0.1% over frontier. A perf change that the harness cannot reach is not a
        // perf change. Set the variable to 0 or 1 to get the old sharding back.
        bool moe_2d_explicit = false;
        int  moe_2d_groups   = tp::k3_default_moe_expert_groups(tp_size);
        if (const char* e = std::getenv("SPARKINFER_K3_MOE_2D")) {
            if (e[0]) {
                moe_2d_explicit = true;
                moe_2d_groups   = std::atoi(e);
                if (moe_2d_groups < 1) moe_2d_groups = 1;
            }
        }
        if (tp_size > 1 && moe_2d_groups > 1) {
            // 256 is the K-quant block every K3 expert tensor uses; a coarser block
            // than the tensor's own only makes the check stricter, never wrong.
            const tp::ShardError se =
                tp::moe_2d_dims(cfg.n_experts, cfg.moe_ffn, tp_size, r,
                                moe_2d_groups, /*block_elems=*/256, &R.weights.shard);
            if (!se.ok()) {
                // ASKED FOR: hard error. CHOSEN FOR YOU: fall back.
                //
                // Now that this is a default, a shape whose experts or ffn width do
                // not divide must still LOAD — refusing would turn a tuning default
                // into a portability regression for a config that works today.
                // moe_2d_dims validates before it writes, so the ShardDims still
                // holds the whole-expert band and the fallback is simply to use it.
                if (moe_2d_explicit) {
                    rank_err[(size_t)r] = "2-D MoE: " + se.message;
                    return false;
                }
                moe_2d_note[(size_t)r] = se.message;
            }
        }

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
        // Remembered so every token can START from the same assignment. The 93 per-layer
        // swaps are odd, so without this the pair alternates and a captured graph's baked
        // addresses would be right on one token and inverted on the next.
        R.x_canon      = R.x;
        R.x_next_canon = R.x_next;
        // Rank 0 keeps the FULL vocab buffer whatever the band decides, so
        // SPARKINFER_K3_HEAD_BAND=0 restores the single-rank head on the same binary
        // without a second allocation path. Ranks 1-7 get exactly their band — 82 KB
        // against rank 0's 655 KB — and nothing at all when the band declines.
        K3HeadBand hb;
        const size_t n_logits =
            (r == 0) ? (size_t)cfg.vocab
                     : (k3_head_band(cfg.vocab, (size_t)R.weights.output.n_bytes,
                                     tp_size, r, &hb) ? (size_t)hb.rows : 0);
        if (n_logits &&
            cudaMalloc(&R.logits, n_logits * sizeof(float)) != cudaSuccess) {
            rank_err[(size_t)r] = "cudaMalloc logits failed";
            return false;
        }

        // Buffered, not printed: eight ranks writing stderr concurrently interleaves
        // mid-line. Emitted in rank order after the join so the log reads as before.
        char line[220];
        if (R.weights.shard.moe_2d) {
            std::snprintf(line, sizeof(line),
                          "[k3-tp] rank %d: device %d, experts [%d,%d), ffn [%d,%d) "
                          "(2-D MoE %dx%d)\n",
                          r, R.device, R.weights.shard.expert_band.offset,
                          R.weights.shard.expert_band.end(),
                          R.weights.shard.moe_ffn_band.offset,
                          R.weights.shard.moe_ffn_band.end(),
                          R.weights.shard.moe_expert_groups,
                          R.weights.shard.moe_ffn_shards);
        } else {
            std::snprintf(line, sizeof(line),
                          "[k3-tp] rank %d: device %d, experts [%d,%d)\n",
                          r, R.device, R.weights.shard.expert_band.offset,
                          R.weights.shard.expert_band.end());
        }
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
    // Once per model, not once per rank: every rank derives the same answer from the
    // same shape, so eight identical lines would only bury it.
    if (!moe_2d_note[0].empty())
        std::fprintf(stderr, "[k3-tp] 2-D MoE off (whole-expert sharding): %s\n",
                     moe_2d_note[0].c_str());
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
    // The widest payload any reduce in the token will carry, and the two candidates
    // are not the ones they used to be:
    //   attention (sharded bands) reduces at hidden                       = 7168
    //   the MoE layer reduces the expert accumulator AND the shared-expert
    //   partial as ONE fused payload (kimi_k3_partial_buffer)             = 10752
    // Sizing to the attention reduce alone would fail every MoE layer at run time,
    // after a two-minute weight load.
    const size_t attn_count = (size_t)cfg.hidden;
    const size_t moe_count  = (size_t)cfg.expert_latent + (size_t)cfg.hidden;
    // ...times the chunk size, because kimi_k3_tp_forward_prompt reduces a whole
    // chunk's partials in ONE call. They are contiguous by construction (the batch
    // arena in kimi_k3.cpp lays slots out end to end), so the payload is exactly
    // B * the per-token width. At B=1 this is the value it has always been.
    // Scaled by the prefill chunk size, because kimi_k3_tp_forward_prompt reduces
    // a chunk's partials in one call and they are contiguous by construction.
    //
    // AN EARLIER ATTEMPT AT THIS WAS BACKED OUT ON A BAD DIAGNOSIS. Scaling it
    // coincided with an 8-token chunk failing the WEIGHT load at layer 68, and that
    // was recorded here as "the owned buffers are peer-mapped per rank per slot, so
    // the multiplier lands on the allocation the model has no headroom for". The
    // arithmetic does not support it: PeerOneShotAllreduce allocates
    // count*4*(kSlotCount+2) bytes PER DEVICE, which at B=8 is ~1.7 MB against ~70
    // GiB of headroom. The real cause was the driver still reclaiming the previous
    // run's 1.1 TiB — nvidia-smi reads 0 MiB before reclamation completes, which is
    // exactly the race kimi_k3_eval.sh's settle_gpus() exists for and which my
    // launcher did not wait out.
    //
    // Capped so a mistyped SPARKINFER_K3_PREFILL_BATCH cannot ask for gigabytes.
    // The larger of the two prefill drivers' widths: the chunked walk reduces B tokens
    // in one call and the tile driver reduces T, and one binary serves both.
    const size_t per_tok    = attn_count > moe_count ? attn_count : moe_count;
    const int    pb_want    = prefill_batch_env() > prefill_tile_env()
                                  ? prefill_batch_env() : prefill_tile_env();
    const int    pb_cap     = pb_want > 512 ? 512 : pb_want;
    const size_t max_count  = per_tok * (size_t)pb_cap;
    out.coll = tp::make_collective(devices, requested, &err, max_count,
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
    // Owned-buffer backends run Mode B proper: the MoE partial is produced into
    // reduce_in() and consumed from reduce_out() (see the swaps in the forward),
    // so no staging copies exist on the collective path. supports_f32 above
    // remains the gate that matters; a backend that passes it can be driven.
    const bool host_reduce_dbg = [] {
        const char* e = std::getenv("SPARKINFER_K3_TP_HOST_REDUCE");
        return e && e[0] == '1';
    }();
    if (out.coll->owns_buffers() && !host_reduce_dbg &&
        out.coll->max_count() >= max_count) {
        const int n = (int)out.ranks.size();
        out.zc_in.resize((size_t)n);
        out.zc_out.resize((size_t)n);
        out.orig_moe.resize((size_t)n);
        out.orig_attn.resize((size_t)n);
        for (int r = 0; r < n; ++r) {
            out.zc_in[(size_t)r]  = (float*)out.coll->reduce_in(r);
            out.zc_out[(size_t)r] = (float*)out.coll->reduce_out(r);
            out.orig_moe[(size_t)r] = kimi_k3_partial_buffer(
                out.ranks[(size_t)r].fwd, cfg.leading_dense,
                K3LayerPhase::FfnPartial, nullptr);
            out.orig_attn[(size_t)r] = kimi_k3_partial_buffer(
                out.ranks[(size_t)r].fwd, cfg.leading_dense,
                K3LayerPhase::Attn, nullptr);
            if (!out.zc_in[(size_t)r] || !out.zc_out[(size_t)r] ||
                !out.orig_moe[(size_t)r] || !out.orig_attn[(size_t)r]) {
                out.zc_in.clear(); out.zc_out.clear();
                out.orig_moe.clear(); out.orig_attn.clear();
                break;
            }
        }
        out.zero_copy = !out.zc_in.empty();
        if (out.zero_copy)
            std::fprintf(stderr, "[k3-tp] f32 zero-copy: the expert partial writes the "
                                 "collective's peer buffers directly (no staging)\n");
    }

    // ---- rotating input slots, so the reduce needs only its ENTRY barrier -----
    //
    // The count of collectives per token is what decides whether the rotation is
    // sound, because a captured graph restarts it at slot 0 every token and the
    // wrap is therefore the binding case (k3_coll_1bar.h). Derive that count from
    // the SAME predicates the forward uses — a second copy of the arithmetic is
    // how a formula like this goes stale — and decline the whole factor if it
    // does not clear the check, rather than rotating anyway on a geometry the
    // proof does not cover.
    if (out.zero_copy && tp_size > 1) {
        int per_token = 0;
        for (int L = 0; L < cfg.n_layers; ++L) {
            const bool kda = cfg.is_kda_layer(L);
            const bool ar =
                (kda && KimiK3Weights::shards_kda(out.ranks[0].weights.policy)) ||
                (!kda && KimiK3Weights::shards_mla(out.ranks[0].weights.policy));
            if (ar) ++per_token;
            if (L >= cfg.leading_dense) ++per_token;
        }
        const int slots = out.coll->reduce_slots();
        if (slots > 1 && tp::k3_coll_1bar_ok(per_token, slots)) {
            const int n = (int)out.ranks.size();
            out.zc_in_slot.assign((size_t)slots, std::vector<float*>((size_t)n, nullptr));
            bool ok = true;
            for (int s = 0; s < slots && ok; ++s)
                for (int r = 0; r < n; ++r) {
                    out.zc_in_slot[(size_t)s][(size_t)r] =
                        (float*)out.coll->reduce_in_slot(r, s);
                    if (!out.zc_in_slot[(size_t)s][(size_t)r]) { ok = false; break; }
                }
            // Slot 0 must BE zc_in, or the two ways of naming the same buffer have
            // diverged and the swap sites would aim at one while the kernel reads
            // the other.
            if (ok)
                for (int r = 0; r < n; ++r)
                    if (out.zc_in_slot[0][(size_t)r] != out.zc_in[(size_t)r]) ok = false;
            if (ok) {
                out.n_coll_slots = slots;
                std::fprintf(stderr, "[k3-tp] collective: 1 barrier/reduce over %d "
                                     "rotating input slots (%d collectives/token)\n",
                             slots, per_token);
            } else {
                out.zc_in_slot.clear();
            }
        }
        if (out.n_coll_slots <= 1) out.zc_in_slot.clear();
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

    // Start every token from the SAME x/x_next assignment. See KimiK3TPRank::x_canon:
    // the 93 per-layer swaps are odd, so the pair alternates between tokens, and a
    // captured graph bakes the addresses it saw. Free eagerly (two pointer writes),
    // and it is what lets one recorded graph serve every token.
    for (auto& R : p.ranks) {
        if (R.x_canon) { R.x = R.x_canon; R.x_next = R.x_next_canon; }
    }

    // Token-local index of the next collective, and therefore which input slot it
    // reduces out of. Reset here, beside the x_canon reset it mirrors and for the
    // same reason: a captured graph bakes the pointers it recorded, so every token
    // has to present the identical sequence. That reset is also exactly what makes
    // the WRAP the case the rotation has to survive — see k3_coll_1bar.h, and note
    // the check in kimi_k3_tp_init that refuses the rotation when it would not.
    int coll_k = 0;
    const int n_slots = p.n_coll_slots;

    // GRAPH CAPTURE. Off with SPARKINFER_K3_GRAPH=0, which is the A/B control: same
    // binary, same kernels, only the submission mechanism differs.
    static const bool want_graph = [] {
        const char* e = std::getenv("SPARKINFER_K3_GRAPH");
        return !(e && e[0] == '0');
    }();
    // `splits` sizes the grid and picks which MLA kernel runs; a graph can change
    // neither, so the plan is part of the graph's identity. Derived from the SAME
    // function the launcher uses so the two cannot disagree about the live plan.
    const int live_plan = k3k::k3_mla_decode_plan(cfg.n_q_heads, cfg.kv_lora_rank,
                                                  p.ranks[0].state.position + 1);
    const bool graph_on = want_graph && !p.graph_disabled && tp_size > 1 && parallel_issue;
    if (graph_on && p.graph_ready && p.captured_plan != live_plan) {
        // The plan moved (a kMlaSplitMinCtx boundary). Throw the graph away rather than
        // replay a grid that no longer fits the context.
        for (auto& R : p.ranks) {
            if (R.exec)  { cudaSetDevice(R.device); cudaGraphExecDestroy(R.exec);  R.exec  = nullptr; }
            if (R.graph) { cudaGraphDestroy(R.graph); R.graph = nullptr; }
        }
        p.graph_ready = false;
    }
    const bool replaying = graph_on && p.graph_ready;
    const bool capturing = graph_on && !p.graph_ready;
    bool launch_fused = false;   // set when the embed job also launched the graph

    {
        const IClock::time_point t0 = ip.on ? IClock::now() : IClock::time_point{};
        // ON REPLAY, fuse the embed and the graph launch into ONE pool job per
        // rank: they are back-to-back on the same stream from the same worker,
        // so one rendezvous (wake + join) is pure overhead. Stream order is
        // identical, so this is bit-identical trivially. Capture tokens keep
        // the two-step form — cudaStreamBeginCapture must sit between them.
        // SPARKINFER_K3_FUSE_ISSUE=0 opts out.
        static const bool fuse_issue = [] {
            const char* e = std::getenv("SPARKINFER_K3_FUSE_ISSUE");
            return !(e && e[0] == '0');
        }();
        if (replaying && fuse_issue) {
            if (!issue_all([&](int r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    if (!embed_token(R.weights, cfg, token_id, R.x, R.stream))
                        return false;
                    return cudaGraphLaunch(R.exec, R.stream) == cudaSuccess;
                })) return false;
            if (ip.on) {
                ip.t_issue += secs_since(t0);
                if (!parallel_issue) ip.n_setdev += tp_size;
            }
            launch_fused = true;
        }
        if (!launch_fused) {
            if (!issue_all([&](int r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    return embed_token(R.weights, cfg, token_id, R.x, R.stream);
                })) return false;
            if (ip.on) {
                ip.t_issue += secs_since(t0);
                if (!parallel_issue) ip.n_setdev += tp_size;
            }
        }
    }

    // Capture starts AFTER the embed. The embed is the only per-token input and it is one
    // kernel out of ~4,376 — leaving it eager keeps the token id out of the graph entirely
    // instead of having to make it device-resident, and costs nothing measurable.
    if (capturing) {
        for (auto& R : p.ranks) {
            if (cudaSetDevice(R.device) != cudaSuccess) return false;
            // RELAXED, not ThreadLocal. This driver enqueues each rank's compute from a
            // PINNED POOL WORKER while the collective is enqueued from the main thread,
            // so the work for one capturing stream legitimately arrives from two threads.
            // ThreadLocal ties the capture to the thread that began it and invalidates it
            // when the other one enqueues — which is the "previous error during capture"
            // this hit at the first all-reduce. Relaxed is the mode that matches a
            // multi-threaded submitter; the pre-warmed split scratch (k3_mla_prewarm_...)
            // is what removes the allocation Relaxed would otherwise stop policing.
            if (cudaStreamBeginCapture(R.stream, cudaStreamCaptureModeRelaxed)
                != cudaSuccess) {
                std::fprintf(stderr, "[k3-graph] BeginCapture failed on rank %d — "
                                     "eager for the rest of the run\n", R.rank);
                for (auto& Q : p.ranks) {
                    cudaGraph_t g = nullptr;
                    cudaSetDevice(Q.device);
                    cudaStreamEndCapture(Q.stream, &g);   // unwind any that did begin
                    if (g) cudaGraphDestroy(g);
                }
                cudaGetLastError();
                p.graph_disabled = true;
                return kimi_k3_tp_forward_token(p, token_id, out_logits);
            }
        }
    }

    // WHERE DID THE CAPTURE DIE? cudaGetLastError only ever reports the CASCADE
    // ("previous error during capture") from wherever the next launch happens to be, which
    // pointed at layer 1's FfnPartial and sent me hunting through the MoE dispatch for a
    // fault that was somewhere else entirely. This asks the stream directly, so the first
    // report is the actual site.
    auto cap_ok = [&](const char* where, int layer) -> bool {
        if (!capturing) return true;
        for (auto& R : p.ranks) {
            cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
            if (cudaSetDevice(R.device) != cudaSuccess) return false;
            const cudaError_t e = cudaStreamGetCaptureInfo(R.stream, &cs, nullptr);
            if (e != cudaSuccess || cs != cudaStreamCaptureStatusActive) {
                std::fprintf(stderr,
                    "[k3-graph] CAPTURE LOST at %s (layer %d, rank %d): status=%d err=%s\n",
                    where, layer, R.rank, (int)cs, cudaGetErrorString(e));
                return false;
            }
        }
        return true;
    };
    if (!cap_ok("after-embed", -1)) return false;

    // When replaying, none of this is issued: the recorded graph already contains every
    // launch. The loop is skipped wholesale rather than guarded per-call, because a
    // partially-skipped token would leave the host-side pointer swaps out of step with
    // what the graph does.
    if (!replaying)
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
            KimiK3Weights::shards_kda(p.ranks[0].weights.policy);
        const bool mla_reduce = tp_size > 1 && !cfg.is_kda_layer(layer) &&
            KimiK3Weights::shards_mla(p.ranks[0].weights.policy);
        const bool attn_reduce = kda_reduce || mla_reduce;

        // Zero-copy: aim the expert accumulator at the collective's peer-visible
        // input before the dispatch writes it, so the reduce needs no staging.
        // Reusing one in/out pair across the token's collectives is race-free: the
        // one-shot kernel's exit barrier proves every peer finished reading this
        // rank's input, and the next layer's dispatch is stream-ordered behind it.
        //
        // Done HERE, by the submitting thread, with the issue pool parked. These
        // calls retarget scratch pointers the workers dereference, so performing
        // them from inside a worker — or while one is running — would be a race.
        const bool zc_attn = p.zero_copy && attn_reduce;

        // WHICH INPUT SLOT EACH OF THIS LAYER'S TWO COLLECTIVES REDUCES OUT OF.
        // Both partials are aimed at their buffer HERE, at the top of the layer,
        // before either phase runs — so the MoE slot has to be known before the
        // attention reduce has happened. It is: the indices are a running count,
        // and the attention reduce, when it exists, is the one immediately before.
        // n_slots == 1 gives slot -1 everywhere, which is main's kernel over
        // main's single buffer.
        const int k_attn = coll_k;
        const int k_moe  = coll_k + (attn_reduce ? 1 : 0);
        const int slot_attn = n_slots > 1 ? (k_attn % n_slots) : -1;
        const int slot_moe  = n_slots > 1 ? (k_moe  % n_slots) : -1;
        float* const* in_attn = slot_attn >= 0
            ? p.zc_in_slot[(size_t)slot_attn].data() : p.zc_in.data();
        float* const* in_moe = slot_moe >= 0
            ? p.zc_in_slot[(size_t)slot_moe].data() : p.zc_in.data();

        if (p.zero_copy && tp_size > 1 && (is_moe || zc_attn)) {
            for (std::size_t r = 0; r < p.ranks.size(); ++r) {
                if (is_moe)
                    kimi_k3_swap_partial_buffer(p.ranks[r].fwd, K3LayerPhase::FfnPartial,
                                                in_moe[r]);
                // The attention partial rides the SAME owned pair. The reuse is
                // safe for the same reason 185 sequential collectives already
                // share it: the attn value is written to zc_in, reduced into
                // zc_out, and fully consumed by the FfnPartial kernels before
                // the expert dispatch (stream-ordered behind them) overwrites
                // zc_in — and the one-shot's exit barrier proves every peer
                // finished reading before that.
                if (zc_attn)
                    kimi_k3_swap_partial_buffer(p.ranks[r].fwd, K3LayerPhase::Attn,
                                                in_attn[r]);
            }
        }

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
            // Owned (no staging copies) when the partial was aimed at zc_in
            // above; the staged group call otherwise. Same reduce kernel, same
            // bits — only the two D2D copies around it disappear. All 93
            // attention collectives paid those copies while the MoE reduce
            // beside them ran with none, because every swap site passed
            // FfnPartial even though kimi_k3_partial_buffer has always handled
            // Attn.
            const bool okk = zc_attn
                ? p.coll->allreduce_f32_owned_slot((size_t)count, p.streams, slot_attn)
                : p.coll->allreduce_f32_group(p.reduce_bufs, (size_t)count,
                                              p.streams);
            if (ip.on) ip.t_coll += secs_since(tk);
            if (!okk) {
                std::fprintf(stderr, "[k3-tp] attention all-reduce failed at layer %d\n", layer);
                return false;
            }
            ++p.n_collectives;
            ++coll_k;
            // FfnPartial reads the reduced attention output where the
            // collective wrote it.
            if (zc_attn) {
                for (std::size_t r = 0; r < p.ranks.size(); ++r)
                    kimi_k3_swap_partial_buffer(p.ranks[r].fwd, K3LayerPhase::Attn,
                                                p.zc_out[r]);
            }

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
                // Zero-copy: the partials are ALREADY in reduce_in() — the swap
                // above aimed them there — so this launches the reduce with no
                // staging copies. The gather above still ran: it is what validates
                // the width every rank agrees on.
                const bool okc = p.zero_copy
                    ? p.coll->allreduce_f32_owned_slot((size_t)count, p.streams, slot_moe)
                    : p.coll->allreduce_f32_group(p.reduce_bufs, (size_t)count,
                                                  p.streams);
                if (ip.on) ip.t_coll += secs_since(tc);
                if (!okc) {
                    std::fprintf(stderr, "[k3-tp] all-reduce failed at layer %d\n", layer);
                    return false;
                }
            }
            ++p.n_collectives;
            ++coll_k;
            // Phase 3's routed_norm reads (and normalises in place) the reduced
            // sum where the collective wrote it. In-place writes to reduce_out are
            // rank-private: peers only ever read inputs.
            if (p.zero_copy) {
                for (std::size_t r = 0; r < p.ranks.size(); ++r)
                    kimi_k3_swap_partial_buffer(p.ranks[r].fwd, K3LayerPhase::FfnPartial,
                                                p.zc_out[r]);
            }
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

    // --- head: every rank holds the identical hidden state ---------------------
    //
    // "so rank 0 suffices" is what this used to say, and it is true — but suffices
    // is not the same as costs nothing. The head is the largest single projection in
    // the model by a wide margin: at vocab 163840 x hidden 7168 the Q8_0 weight is
    // (7168/32) * 34 * 163840 = 1.25 GB, 45x the biggest per-layer projection, and
    // it ran on ONE device while the other seven sat idle holding the same hidden
    // state. It is also completely exposed — the driver blocks on this stream a few
    // lines below with nothing else in flight — so it is the one place in the token
    // where a saving transfers 1:1 instead of being hidden behind other work.
    //
    // Every rank already HOLDS output.weight. weight_plan.cpp declares it RowShard,
    // but upload_sliced only consults the rule table for the expert/KDA/MLA stacks,
    // so the head short-circuits to a full replica on all eight devices — VRAM they
    // pay for and never read. Banding it is therefore a pointer offset and a smaller
    // N: no loader change, no new collective, no change to which numbers exist.
    //
    // The mix and the norm are recomputed on every rank rather than broadcast. They
    // are two small kernels over one 7168-float vector, against the 28 KB that a
    // broadcast plus its rendezvous would cost, and the inputs are provably identical
    // across ranks: R.x at the head is the last layer's output, which every rank
    // reads back from the same all-reduce, and res_bank holds snapshots of that same
    // post-reduce hidden state (kimi_k3.cpp: the bank push copies `hidden_in`, which
    // enters phase 1 straight out of the previous layer's reduced result). n_ckpt is
    // host bookkeeping advanced identically on every rank by the same code path.
    KimiK3TPRank& R0 = p.ranks[0];
    if (cudaSetDevice(R0.device) != cudaSuccess) return false;
    const KimiK3Weights& w = R0.weights;

    // Decided once per token from rank 0, and every rank is then required to agree:
    // a band that resolved on some ranks and not others would leave a hole in the
    // logits rather than a slow token.
    K3HeadBand hb0;
    bool band_head = w.output.ok() &&
                     k3_head_band(cfg.vocab, (size_t)w.output.n_bytes, tp_size, 0, &hb0);
    if (band_head) {
        for (int r = 1; r < tp_size; ++r) {
            K3HeadBand hbr;
            const KimiK3Weights& wr = p.ranks[(size_t)r].weights;
            if (p.ranks[(size_t)r].logits && wr.output.ok() &&
                wr.output.n_bytes == w.output.n_bytes &&
                k3_head_band(cfg.vocab, (size_t)wr.output.n_bytes, tp_size, r, &hbr))
                continue;
            band_head = false;
            break;
        }
    }

    if (!replaying) {
    if (band_head) {
        for (int r = 0; r < tp_size; ++r) {
            KimiK3TPRank& R = p.ranks[(size_t)r];
            const KimiK3Weights& wr = R.weights;
            K3HeadBand hb;
            if (!k3_head_band(cfg.vocab, (size_t)wr.output.n_bytes, tp_size, r, &hb))
                return false;
            if (cudaSetDevice(R.device) != cudaSuccess) return false;
            if (cfg.attn_res_block_size > 0) {
                if (!wr.has_output_res_score || !wr.output_res_score.ok()) return false;
                k3k::attn_res_mix_f32(R.x_next, R.state.res_bank, R.x,
                                      (const float*)wr.output_res_score.data, H,
                                      R.state.n_ckpt, cfg.rms_eps, R.stream);
                // Every rank now takes the extra swap rank 0 alone used to take. The
                // per-token reset to x_canon is what keeps a captured graph's baked
                // addresses valid, and it already runs for every rank.
                std::swap(R.x, R.x_next);
            }
            if (!wr.output_norm.ok()) return false;
            k3k::rms_norm_f32(R.x_next, R.x, (const float*)wr.output_norm.data, H,
                              cfg.rms_eps, R.stream);
            // BIT-IDENTICAL. Logit n is one dot product over the whole of K, and the
            // kernel accumulates it the same way whatever N is: rows [0,N) map to
            // blocks, a row's accumulator sums quant blocks b = threadIdx.x, +BLOCK,
            // ... in that order, and N only decides how many rows a grid carries.
            // N = 163840 and N = 20480 both land in the same ROWS=16 tier, so the
            // same kernel with the same block width computes each row identically —
            // only on a different device.
            if (!k3k::k3_proj_f32(R.logits, R.x_next,
                                  (const void*)((const char*)wr.output.data + hb.byte_off),
                                  wr.output.type, hb.rows, H, R.stream))
                return false;
        }
        if (cudaSetDevice(R0.device) != cudaSuccess) return false;
    } else {
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
    }

    // THE DEVICE POSITION ADVANCES INSIDE THE GRAPH. This is the single line that makes a
    // replay a different token rather than the same one again: the KV store indexes with
    // *d_pos and the attention lengths its loop with *d_pos + 1, so without a bump in the
    // recorded region every replayed token would overwrite one KV row and attend over a
    // frozen context. The host mirror is advanced separately, at the end of the call.
    //
    // Each rank's device must be CURRENT for its own launch: this runs inside the head,
    // where rank 0's device was made current, and enqueueing onto another rank's stream
    // from the wrong device is an "invalid resource handle" — which is what it was.
    for (auto& R : p.ranks) {
        if (!R.state.d_pos) continue;
        if (cudaSetDevice(R.device) != cudaSuccess) return false;
        k3k::k3_bump_pos(R.state.d_pos, R.stream);
    }
    if (cudaSetDevice(R0.device) != cudaSuccess) return false;   // restore for the head's sync
    }  // if (!replaying)

    // Close the capture and instantiate. Nothing has EXECUTED yet on a capture token —
    // capture records — so the launch below is what runs this token's work, on the
    // capture pass and on every replay alike.
    if (capturing) {
        bool ok_cap = true;
        for (auto& R : p.ranks) {
            if (cudaSetDevice(R.device) != cudaSuccess) { ok_cap = false; break; }
            if (cudaStreamEndCapture(R.stream, &R.graph) != cudaSuccess ||
                R.graph == nullptr) { ok_cap = false; break; }
            if (cudaGraphInstantiate(&R.exec, R.graph, nullptr, nullptr, 0)
                != cudaSuccess) { ok_cap = false; break; }
        }
        if (!ok_cap) {
            // A FAILED CAPTURE MUST COST SPEED, NEVER CORRECTNESS.
            //
            // This used to `return false`, which failed the token and took the whole run
            // down — so a binary built with the default (graph on) would not decode at all
            // on a box where capture is unavailable. Backwards: capture is an optimisation,
            // and the model runs perfectly well without it. Disable it for the rest of the
            // process and let this token be re-issued eagerly by the caller.
            std::fprintf(stderr, "[k3-graph] capture failed — eager for the rest of the run\n");
            for (auto& R : p.ranks) {
                cudaSetDevice(R.device);
                if (R.exec)  { cudaGraphExecDestroy(R.exec);  R.exec  = nullptr; }
                if (R.graph) { cudaGraphDestroy(R.graph);     R.graph = nullptr; }
            }
            cudaGetLastError();          // clear the sticky capture error
            p.graph_disabled = true;     // never attempt capture again
            p.graph_ready    = false;
            return kimi_k3_tp_forward_token(p, token_id, out_logits);   // eager retry
        }
        p.graph_ready   = true;
        p.captured_plan = live_plan;
        ++p.n_captures;
        size_t nnodes = 0;
        cudaGraphGetNodes(p.ranks[0].graph, nullptr, &nnodes);
        std::fprintf(stderr, "[k3-graph] captured decode step: %zu nodes/rank, "
                             "%ld collectives/token, mla splits=%d, from position %d\n",
                     nnodes, p.n_collectives, live_plan, p.ranks[0].state.position);
    }

    if (graph_on) {
        // LAUNCH THE EIGHT GRAPHS CONCURRENTLY, NOT IN RANK ORDER.
        //
        // This is the point of the whole change and it is easy to get wrong by writing the
        // obvious loop. Measured on #86's capture-ON build, per-rank mean lateness at the
        // collective rises MONOTONICALLY with rank — 19.5 us at rank 0 to 34.4 us at rank 7,
        // which was last to arrive at 40% of the 185 barriers. Routing skew is
        // data-dependent and would scatter; a clean 0..7 gradient is submission order.
        //
        // Capture was supposed to kill that and does not, because a host loop still issues
        // cudaGraphLaunch eight times in rank order: rank 7's token starts after seven
        // launches have gone ahead of it, every token, by construction. The barrier then
        // charges everyone for it — 81.5% of collective time is ranks waiting, 4.80 ms of
        // 5.89 ms per rank per token.
        //
        // The issue pool already owns one thread per rank with its device pinned at thread
        // start, so dispatching the launch through it costs nothing and removes the ordering.
        // Skipped when the embed job already launched the graph (FUSE_ISSUE).
        if (!launch_fused && !issue_all([&](int r) {
                return cudaGraphLaunch(p.ranks[(size_t)r].exec,
                                       p.ranks[(size_t)r].stream) == cudaSuccess;
            })) {
            std::fprintf(stderr, "[k3-graph] graph launch failed\n");
            return false;
        }
        ++p.n_replays;
    }

    const IClock::time_point t_s0 = ip.on ? IClock::now() : IClock::time_point{};
    if (band_head) {
        // Eight disjoint 82 KB copies instead of one 655 KB copy, dispatched through
        // the pool so they run on eight threads with eight devices current rather
        // than serialising behind rank 0. The sync moves inside for the same reason:
        // the ranks finish within a few microseconds of each other (the last layer's
        // all-reduce is a rendezvous), so waiting on them in parallel costs about
        // what waiting on rank 0 alone used to.
        // PIN THE LANDING BUFFER, once, lazily. The caller's out_logits is
        // pageable (the bench hands us a std::vector), so 8 workers each pay a
        // driver bounce-buffer staging on every token. cudaHostRegister makes
        // the same bytes DMA-able in place; cached by pointer so the cost is
        // one-time. Same bytes, same destination — bit-identical trivially.
        // SPARKINFER_K3_PIN_LOGITS=0 opts out.
        static const bool pin_logits = [] {
            const char* e = std::getenv("SPARKINFER_K3_PIN_LOGITS");
            return !(e && e[0] == '0');
        }();
        static float* pinned_ptr = nullptr;
        if (pin_logits && out_logits != pinned_ptr) {
            if (pinned_ptr) cudaHostUnregister(pinned_ptr);
            pinned_ptr = (cudaHostRegister(out_logits,
                              (size_t)cfg.vocab * sizeof(float),
                              cudaHostRegisterDefault) == cudaSuccess)
                             ? out_logits : nullptr;
            if (!pinned_ptr) cudaGetLastError();   // clear; pageable path still works
        }
        if (!issue_all([&](int r) {
                KimiK3TPRank& R = p.ranks[(size_t)r];
                K3HeadBand hb;
                if (!k3_head_band(cfg.vocab, (size_t)R.weights.output.n_bytes,
                                  tp_size, r, &hb)) return false;
                if (cudaStreamSynchronize(R.stream) != cudaSuccess) return false;
                return cudaMemcpy(out_logits + hb.offset, R.logits,
                                  (size_t)hb.rows * sizeof(float),
                                  cudaMemcpyDeviceToHost) == cudaSuccess;
            })) return false;
        if (ip.on) ip.t_sync += secs_since(t_s0);
    } else {
    if (cudaStreamSynchronize(R0.stream) != cudaSuccess) return false;
    if (ip.on) ip.t_sync += secs_since(t_s0);
    if (cudaMemcpy(out_logits, R0.logits, (size_t)cfg.vocab * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    }

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
    //
    // BOTH copies advance, and the device one advances ON THE STREAM. The host mirror
    // picks the launch plan (a graph cannot resize its own grid); the device value is what
    // the KV store indexes with and the attention lengths its loop with. Bumping the
    // device side from inside the stream is what makes a REPLAY advance — a host-only
    // increment would leave every replayed token writing the capture-time row.
    // Host mirror only. The DEVICE side was bumped inside the captured region above —
    // doing it again here would double-advance and skip every other KV row.
    for (auto& R : p.ranks) ++R.state.position;
    return true;
}

// ---------------------------------------------------------------------------
// Prompt prefill: the layer loop OUTSIDE the token loop
// ---------------------------------------------------------------------------
//
// PER-TOKEN STATE THE REORDER HAS TO REPLICATE, in full. The list is short, and its
// shortness is the only reason this reorder is tractable:
//
//   position / d_pos    the KV row this token writes, and the length it attends over
//   res_bank / n_ckpt   the cross-layer residual checkpoints, reset every token
//
// Everything else in KimiK3RuntimeState is per SEQUENCE, not per token: the KDA
// conv/delta recurrence and the MLA KV cache both advance one token at a time, which is
// exactly the order the inner loop walks them in, so they need nothing at all. Scratch
// (s.qkv_q, s.normed, ...) is reused per (layer, token) and is safe for the same reason
// it is safe to reuse across layers today — STREAM ORDER. Token t's kernels are enqueued
// ahead of token t+1's on the same stream, so t is complete before t+1 reads the buffer.
//
// WHAT WOULD GO WRONG WITHOUT EACH. A shared res_bank would let token t+1's checkpoint
// push overwrite a row token t has not mixed yet; a shared position would write every
// token of the tile to one KV row. Neither crashes. Both produce fluent, wrong output,
// which is the failure mode this file refuses to ship anywhere else.

// Ceiling on the tile width, so the per-token bank pointers can be gathered into a
// stack array inside a pool worker rather than a heap allocation on the hot path.
// k3_prefill_pick_tile never returns more than this.
enum { K3_PREFILL_MAX_TILE = 16 };

struct K3PrefillPool {
    int tile = 0;                       // T
    int qkv = 0;                        // this rank's KDA q/k/v/g width
    std::vector<K3PrefillTile> tiles;   // per rank
    std::vector<float*> banks;          // per rank, [tile][max_ckpt][hidden]
    std::vector<float*> orig_bank;      // per rank, the state's own bank, restored after
    // Canonical tile residual pair, for the same reason KimiK3TPRank::x_canon exists:
    // 93 per-layer swaps is ODD, so the pair comes out of a tile exchanged relative to
    // how it went in. Eagerly harmless, fatal to a captured tile graph.
    std::vector<float*> x_canon, x_next_canon;

    // ONE RECORDED TILE, REPLAYED FOR EVERY SUBSEQUENT TILE — and it is not an
    // optimisation on top of this driver, it is what makes the driver worth having.
    // Eager submission costs 12.59 ms/token measured (SPARKINFER_K3_GRAPH=1 vs =0, same
    // binary: 15.62 vs 28.21 ms/token), against the ~2.6 ms the batched projections save.
    // A tile driver without capture therefore measures a large REGRESSION while doing
    // strictly less arithmetic, which is a fact about the submission mechanism and not
    // about the optimisation.
    //
    // Every requirement decode's capture has is met here:
    //   - positions advance by k3_add_pos, RELATIVE, so a replay runs its own tile;
    //   - x/x_next are reset to canonical per tile, so baked addresses stay valid;
    //   - res_bank slices are fixed for the pool's life and n_ckpt is a function of the
    //     layer index alone, so both bake correctly;
    //   - the collective rotation restarts per tile, which is why the wrap check is
    //     applied to per_token * T rather than per_token;
    //   - the MLA launch plan is a HOST decision from st.position, and a tile spans T
    //     positions, so all T must agree AND match the recorded one (see plan_for_tile).
    std::vector<cudaGraph_t> graph;
    std::vector<cudaGraphExec_t> exec;
    bool graph_ready = false;
    bool graph_disabled = false;
    int  captured_plan = 0;
    long n_captures = 0, n_replays = 0;
};

namespace {

void k3_prefill_pool_free(K3PrefillPool*& pool, std::vector<KimiK3TPRank>& ranks) {
    if (!pool) return;
    for (std::size_t r = 0; r < pool->tiles.size() && r < ranks.size(); ++r) {
        cudaSetDevice(ranks[r].device);
        if (r < pool->exec.size() && pool->exec[r]) cudaGraphExecDestroy(pool->exec[r]);
        if (r < pool->graph.size() && pool->graph[r]) cudaGraphDestroy(pool->graph[r]);
        // Restore the canonical pair before freeing: the tile swaps leave x/x_next
        // exchanged, and free must release what alloc handed out.
        if (r < pool->x_canon.size() && pool->x_canon[r]) {
            pool->tiles[r].x = pool->x_canon[r];
            pool->tiles[r].x_next = pool->x_next_canon[r];
        }
        k3_prefill_tile_free(pool->tiles[r]);
        if (r < pool->banks.size() && pool->banks[r]) cudaFree(pool->banks[r]);
        // Point the state's bank back at its own allocation. Not load-bearing for
        // teardown (the state frees via its owned list) but a decode step after a
        // prefill would otherwise be reading tile memory that no longer exists.
        if (r < pool->orig_bank.size() && pool->orig_bank[r])
            ranks[r].state.res_bank = pool->orig_bank[r];
    }
    delete pool;
    pool = nullptr;
}

// The tile width this geometry admits, or 0 to decline the tile path entirely.
//
// The binding constraint is NOT memory, it is the collective slot rotation. A tile is
// the unit the rotation restarts at, so the wrap check applies to per_token * T rather
// than to per_token — and it does not hold for every T. At K3's 185 collectives/token
// with 3 slots, T=4 passes ((740-1)%3 = 1) and T=2 FAILS ((370-1)%3 = 0). A T=2 tile
// would put two writes to one input slot a distance of 1 apart, which is a cross-GPU
// data race: a fast rank overwriting an input a slow peer is still reading.
int k3_prefill_pick_tile(const KimiK3TP& p, int want) {
    int per_token = 0;
    for (int L = 0; L < p.cfg.n_layers; ++L) {
        const bool kda = p.cfg.is_kda_layer(L);
        const bool ar = (kda && KimiK3Weights::shards_kda(p.ranks[0].weights.policy)) ||
                        (!kda && KimiK3Weights::shards_mla(p.ranks[0].weights.policy));
        if (ar) ++per_token;
        if (L >= p.cfg.leading_dense) ++per_token;
    }
    if (want > K3_PREFILL_MAX_TILE) want = K3_PREFILL_MAX_TILE;
    for (int T = want; T >= 1; --T) {
        // Rotation off (one slot) means every collective already uses one buffer with
        // its exit barrier intact, so there is no reuse distance to protect and any T
        // is sound.
        if (p.n_coll_slots <= 1) return T;
        // NOT per_token * T ANY MORE. The tile reduces a whole tile's partials in ONE
        // call per phase, so a tile issues the same number of collectives as a single
        // token does — the count no longer scales with T, and neither does the rotation
        // constraint. That is what frees T from the {2,5,8,11,14} exclusion it used to
        // have at 185 collectives with 3 slots.
        if (tp::k3_coll_1bar_ok(per_token, p.n_coll_slots)) return T;
    }
    return 0;
}

bool k3_prefill_pool_init(KimiK3TP& p, int want_tile) {
    if (p.prefill) return true;
    const KimiK3Config& cfg = p.cfg;
    const int T = k3_prefill_pick_tile(p, want_tile);
    if (T <= 1) return false;   // a tile of 1 is the per-token path with extra bookkeeping

    // Every rank's KDA shard width must agree, because one tile width is allocated for
    // all of them and the fill's launch geometry is shared.
    //
    // Read from weights.kda.qkv and NOTHING ELSE. That is the field the consumer in
    // kimi_k3.cpp derives its own `qkv` from, and under every unsharded policy it holds
    // the tp=1 identity — so this cannot drift from it. Recomputing the width here as
    // n_q_heads * kda_head_dim would agree today and silently disagree the moment a
    // policy shards KDA, and the tile guard would then reject every layer.
    int qkv = 0;
    for (std::size_t r = 0; r < p.ranks.size(); ++r) {
        const int q = p.ranks[r].weights.kda.qkv;
        if (r == 0) qkv = q;
        else if (q != qkv) return false;
    }
    if (qkv <= 0) return false;

    K3PrefillPool* pool = new K3PrefillPool();
    pool->tile = T;
    pool->qkv = qkv;
    pool->tiles.resize(p.ranks.size());
    pool->banks.assign(p.ranks.size(), nullptr);
    pool->orig_bank.assign(p.ranks.size(), nullptr);
    pool->x_canon.assign(p.ranks.size(), nullptr);
    pool->x_next_canon.assign(p.ranks.size(), nullptr);
    pool->graph.assign(p.ranks.size(), nullptr);
    pool->exec.assign(p.ranks.size(), nullptr);

    bool ok = true;
    for (std::size_t r = 0; r < p.ranks.size() && ok; ++r) {
        KimiK3TPRank& R = p.ranks[r];
        if (cudaSetDevice(R.device) != cudaSuccess) { ok = false; break; }
        if (!k3_prefill_tile_alloc(cfg, T, qkv, R.state.max_ckpt, pool->tiles[r])) {
            ok = false; break;
        }
        pool->x_canon[r] = pool->tiles[r].x;
        pool->x_next_canon[r] = pool->tiles[r].x_next;
        pool->orig_bank[r] = R.state.res_bank;
        if (R.state.max_ckpt > 0) {
            const size_t n = (size_t)T * (size_t)R.state.max_ckpt * (size_t)cfg.hidden;
            void* q = nullptr;
            if (cudaMalloc(&q, n * sizeof(float)) != cudaSuccess) { ok = false; break; }
            pool->banks[r] = (float*)q;
        }
        // THE BATCH ARENA, and it is what makes phase-major order possible at all.
        //
        // A phase-major tile runs every token's attention, THEN reduces, THEN every
        // token's FFN. That requires each token's attention partial to still exist when
        // its FFN reads it — and s.attn_out is ONE TOKEN WIDE, so in token-major order
        // token t+1 overwrites token t's before the reduce ever sees it. The arena gives
        // each token its own attn_out/ffn_out/moe_fused slot, laid out end to end, which
        // is also exactly the contiguity the batched collective needs.
        //
        // Allocated here rather than in the chunked walk's prompt_alloc because the two
        // drivers size it differently (T vs B) and whichever runs first wins; alloc_batch
        // returns true when the existing arena is already big enough.
        if (!R.fwd.state) { ok = false; break; }
        if (!kimi_k3_forward_alloc_batch(cfg, R.fwd, T)) { ok = false; break; }
    }
    if (!ok) {
        k3_prefill_pool_free(pool, p.ranks);
        std::fprintf(stderr, "[k3-prefill] tile pool alloc failed; per-token path\n");
        return false;
    }
    p.prefill = pool;
    std::fprintf(stderr, "[k3-prefill] tile driver: T=%d tokens resident per layer, "
                         "qkv=%d/rank\n", T, qkv);
    return true;
}

}  // namespace

// Ingest exactly `T` tokens with the layer loop outside the token loop. No logits: the
// caller runs the final token through the per-token path, which owns the head.
static bool k3_tp_prefill_tile(KimiK3TP& p, const int* ids, int T) {
    const KimiK3Config& cfg = p.cfg;
    const int H = cfg.hidden;
    const int tp_size = (int)p.ranks.size();
    K3PrefillPool& pool = *p.prefill;
    if (T <= 0 || T > pool.tile) return false;

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
    auto issue_all = [&](const std::function<bool(int)>& job) -> bool {
        if (parallel_issue) return p.issue.run(job);
        for (int r = 0; r < tp_size; ++r) {
            if (cudaSetDevice(p.ranks[(size_t)r].device) != cudaSuccess) return false;
            if (!job(r)) return false;
        }
        return true;
    };

    // Use the batched attention-side fill? On by default; SPARKINFER_K3_PREFILL_BATCH=0
    // runs the identical loop with the per-token projections, which is the A/B control
    // for the whole optimisation on ONE binary.
    static const bool want_batch = [] {
        const char* e = std::getenv("SPARKINFER_K3_PREFILL_BATCH");
        return !(e && e[0] == '0');
    }();

    // Start every tile from the SAME pair assignment, for the reason KimiK3TPRank's
    // x_canon exists one level down: 93 per-layer swaps is odd.
    for (std::size_t r = 0; r < p.ranks.size(); ++r) {
        pool.tiles[r].x = pool.x_canon[r];
        pool.tiles[r].x_next = pool.x_next_canon[r];
    }

    // The position every rank enters this tile at. Ranks run the same attention, so a
    // rank whose position drifted would index a different KV row for the same token.
    const int base = p.ranks[0].state.position;
    for (auto& R : p.ranks)
        if (R.state.position != base) return false;

    // Per-token checkpoint counts. They evolve identically for every token (n_ckpt is a
    // function of the layer index alone), but carrying them per token is what makes that
    // an observation rather than an assumption the tile depends on.
    std::vector<std::vector<int>> ckpt((size_t)tp_size, std::vector<int>((size_t)T, 0));

    // ---- graph capture over the whole tile -----------------------------------------
    // SPARKINFER_K3_GRAPH=0 is the same A/B control the decode step uses, so the two
    // paths can be compared with capture on or off together.
    static const bool want_graph = [] {
        const char* e = std::getenv("SPARKINFER_K3_GRAPH");
        return !(e && e[0] == '0');
    }();
    // THE PLAN MUST BE UNIFORM ACROSS THE TILE, not merely equal to the recorded one.
    // `splits` sizes the grid and picks which MLA kernel runs, and a tile spans T
    // positions — so a tile straddling a kMlaSplitMinCtx boundary would need two
    // different grids inside ONE recorded graph, which a graph cannot express. Such a
    // tile runs eagerly; there are a handful of them in a 32k prompt.
    const int plan_lo = k3k::k3_mla_decode_plan(cfg.n_q_heads, cfg.kv_lora_rank, base + 1);
    const int plan_hi = k3k::k3_mla_decode_plan(cfg.n_q_heads, cfg.kv_lora_rank, base + T);
    const bool plan_uniform = (plan_lo == plan_hi);
    // THE TILE-WIDE REDUCE MUST GO THROUGH THE COLLECTIVE'S OWNED BUFFERS, and capture
    // is what makes that a hard requirement rather than a preference.
    //
    // The first phase-major version reduced the batch arena with allreduce_f32_group and
    // DEADLOCKED under capture — GPUs 0 and 1 pinned at 100% with the rest idle, a peer
    // barrier waiting on ranks that never arrived. The group entry point takes arbitrary
    // pointers and was only ever exercised by the chunked walk, which never captured;
    // allreduce_f32_owned_slot is the one the tile used under capture, and the rotating
    // slots exist precisely to make it capture-safe. Swapping them was the bug.
    //
    // Eagerly the reorder is fine either way and bit-exact (0.0 KLD, argmax and logit
    // identical to the per-token reference), so this gates CAPTURE, not the reorder: a
    // backend without owned buffers still runs the tile, just without a graph, rather
    // than hanging.
    const bool coll_owned = p.zero_copy && p.coll->owns_buffers() &&
                            p.coll->max_count() > 0;
    const bool graph_on = want_graph && !pool.graph_disabled && plan_uniform &&
                          parallel_issue && tp_size > 1 && coll_owned;
    if (want_graph && !coll_owned && !pool.graph_disabled) {
        static bool said = false;
        if (!said) {
            said = true;
            std::fprintf(stderr, "[k3-prefill] collective has no owned buffers; the "
                                 "tile-wide reduce cannot be captured — running eager\n");
        }
    }
    if (graph_on && pool.graph_ready && pool.captured_plan != plan_lo) {
        for (std::size_t r = 0; r < p.ranks.size(); ++r) {
            cudaSetDevice(p.ranks[r].device);
            if (pool.exec[r])  { cudaGraphExecDestroy(pool.exec[r]);  pool.exec[r]  = nullptr; }
            if (pool.graph[r]) { cudaGraphDestroy(pool.graph[r]);     pool.graph[r] = nullptr; }
        }
        pool.graph_ready = false;
    }
    const bool replaying = graph_on && pool.graph_ready;
    const bool capturing = graph_on && !pool.graph_ready;

    // Embed all T rows up front: one row per token, into the tile's own residual buffer.
    // ALWAYS EAGER, and outside the capture for the same reason the decode step leaves
    // its embed outside: the token ids are this tile's only per-call input, so recording
    // them would freeze the prompt into the graph.
    if (!issue_all([&](int r) {
            KimiK3TPRank& R = p.ranks[(size_t)r];
            for (int t = 0; t < T; ++t)
                if (!embed_token(R.weights, cfg, ids[t],
                                 pool.tiles[(size_t)r].x + (size_t)t * H, R.stream))
                    return false;
            return true;
        })) return false;

    // Capture starts AFTER the embeds, exactly as the decode step's does.
    if (capturing) {
        for (auto& R : p.ranks) {
            if (cudaSetDevice(R.device) != cudaSuccess) return false;
            // RELAXED, not ThreadLocal: this driver enqueues compute from pool workers
            // while the collectives are enqueued from this thread, so one capturing
            // stream legitimately receives work from two threads. See the decode step.
            if (cudaStreamBeginCapture(R.stream, cudaStreamCaptureModeRelaxed)
                != cudaSuccess) {
                std::fprintf(stderr, "[k3-prefill] BeginCapture failed on rank %d — "
                                     "eager for the rest of the run\n", R.rank);
                for (auto& Q : p.ranks) {
                    cudaGraph_t g = nullptr;
                    cudaSetDevice(Q.device);
                    cudaStreamEndCapture(Q.stream, &g);   // unwind any that did begin
                    if (g) cudaGraphDestroy(g);
                }
                cudaGetLastError();
                pool.graph_disabled = true;
                return k3_tp_prefill_tile(p, ids, T);   // eager retry, same tile
            }
        }
    }

    // Counts the collectives this tile issues, and indexes the rotation. A tile now
    // issues TWO per layer instead of two per token, so the wrap the 1bar rotation has to
    // survive is per TILE — which is why k3_prefill_pick_tile stopped multiplying its
    // check by T, and why T is no longer restricted to the widths that check admitted.
    int coll_k = 0;
    const int n_slots = p.n_coll_slots > 0 ? p.n_coll_slots : 1;

    // Skipped wholesale on a replay rather than guarded per call: the recorded graph
    // already contains every launch, and a partially-skipped tile would leave the
    // host-side pointer swaps out of step with what the graph does.
    if (!replaying)
    for (int layer = 0; layer < cfg.n_layers; ++layer) {
        const bool is_moe = layer >= cfg.leading_dense;
        const bool kda_reduce = tp_size > 1 && cfg.is_kda_layer(layer) &&
            KimiK3Weights::shards_kda(p.ranks[0].weights.policy);
        const bool mla_reduce = tp_size > 1 && !cfg.is_kda_layer(layer) &&
            KimiK3Weights::shards_mla(p.ranks[0].weights.policy);
        const bool attn_reduce = kda_reduce || mla_reduce;

        // THE POINT OF THE WHOLE EXERCISE. One pass over this layer's q/k/v/g weights,
        // multiplied into all T tokens, instead of T passes over the same ~11.7 MB.
        // Declining (a non-KDA layer, a weight that is not Q8_0) leaves layer == -1 on
        // the tile, and the consumer's guard then falls through to the per-token
        // projection — the fallback is the tile simply not matching, not a branch here.
        // The fill reads each token's checkpoint bank AS IT STANDS ON ENTRY to this
        // layer, which is what the consumer will see too: a token's push happens inside
        // its own attention phase, after its own mix, so nothing in this layer has
        // touched the bank yet. `ckpt` holds the matching counts for the same reason.
        bool tile_filled = false;
        if (want_batch && cfg.is_kda_layer(layer)) {
            if (!issue_all([&](int r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    float* bk[K3_PREFILL_MAX_TILE];
                    for (int t = 0; t < T; ++t)
                        bk[t] = pool.banks[(size_t)r]
                              ? pool.banks[(size_t)r] +
                                (size_t)t * (size_t)R.state.max_ckpt * (size_t)H
                              : nullptr;
                    if (!k3_prefill_fill_qkvg(pool.tiles[(size_t)r],
                                              R.weights.layers[(size_t)layer],
                                              cfg, layer, T, bk, ckpt[(size_t)r].data(),
                                              R.stream))
                        pool.tiles[(size_t)r].layer = -1;   // decline; consumer falls back
                    return true;
                })) return false;
            tile_filled = true;
        }
        if (!tile_filled)
            for (auto& t : pool.tiles) t.layer = -1;

        // ---- PHASE-MAJOR: three passes over the tile, one collective per phase -----
        //
        // WHY THE ORDER CHANGED. Token-major — run token t's whole layer, then t+1's —
        // issues a collective PER TOKEN. That is 185 per token and ~5.5% of the recorded
        // graph's nodes. Phase-major runs every token's attention, reduces ONCE over the
        // whole tile, then every token's FFN, so a tile issues what one token used to.
        //
        // THIS IS A NODE CHANGE, NOT A LATENCY ONE, and the distinction is the entire
        // reason it is worth doing. Collectives cost ~0.01 ms of an 18.8 ms token, so the
        // chunked walk's 1.77x for the same idea DOES NOT TRANSFER — that came off an
        // uncaptured path where it was buying back submission overhead capture removes
        // for free. What it buys here is graph SIZE: capture is worth 1.93x on the
        // per-token path's 3308-node graph and only 1.56x on a T=16 tile's 61258-node
        // one. Reading the earlier result as "batching collectives is worthless" would
        // have skipped this — it was worthless for the reason it was first proposed.
        //
        // WHAT MAKES IT LEGAL. The only cross-token dependencies inside a layer run along
        // the RECURRENT axis — the KDA conv/delta state and the MLA KV rows — and both are
        // consumed by ATTENTION, which still runs strictly in token order in pass 1.
        // Nothing in token t+1's attention reads token t's FFN output; that flows to the
        // next LAYER. So deferring every FFN until after every attention reorders only
        // work that was already independent. Measured bit-exact against the per-token
        // reference (0.0 KLD, top-1 100%, argmax 10677 and logit 19.774452 on both).
        //
        // WHERE THE PARTIALS LIVE. Token t's partial goes straight into the COLLECTIVE'S
        // OWNED input buffer at offset t*width, and the reduce is one owned-slot call
        // over T*width. Not the batch arena: reducing the arena means allreduce_f32_group,
        // which takes arbitrary pointers, was only ever exercised by the uncaptured
        // chunked walk, and DEADLOCKED here under capture. The owned buffers are sized
        // per_tok * T at init for exactly this.
        //
        // The arena still backs res_bank and the leading dense layer's ffn_out via
        // select_slot, which is why it is still allocated.
        const int w_attn = H;
        const int w_moe  = is_moe ? (cfg.expert_latent + H) : H;
        const int slot_attn = attn_reduce ? (coll_k % n_slots) : -1;
        const int slot_moe  = (coll_k + (attn_reduce ? 1 : 0)) % n_slots;
        auto set_tok = [&](int t) {
            // Host-side fields the workers dereference, written by the submitting thread
            // with the pool parked — the rule the zero-copy pointer swaps followed, and
            // for the same reason.
            for (int r = 0; r < tp_size; ++r) {
                KimiK3TPRank& R = p.ranks[(size_t)r];
                R.state.position = base + t;      // host mirror: picks the MLA plan
                R.fwd.prefill_tile = &pool.tiles[(size_t)r];
                R.fwd.prefill_tok = t;
            }
        };
        // Address token t's slot on rank r. TWO steps, and the second is not optional:
        // select_slot aims state->res_bank at the ARENA's bank, but this tile owns its
        // own bank and the fill reads THAT one. Leaving select_slot's choice in place
        // would have the fill and the consumer mixing over different memory — same
        // shapes, different values, fluent wrong output.
        auto pick_slot = [&](int r, int t) {
            KimiK3TPRank& R = p.ranks[(size_t)r];
            kimi_k3_forward_select_slot(R.fwd, t);
            if (pool.banks[(size_t)r]) {
                R.state.res_bank = pool.banks[(size_t)r] +
                    (size_t)t * (size_t)R.state.max_ckpt * (size_t)H;
                R.state.n_ckpt = ckpt[(size_t)r][(size_t)t];
            }
        };
        auto save_ckpt = [&](int t) {
            for (int r = 0; r < tp_size; ++r)
                ckpt[(size_t)r][(size_t)t] = p.ranks[(size_t)r].state.n_ckpt;
        };
        // Aim `phase`'s partial at token t's row of a peer-visible region. `in` picks the
        // rotation slot's input (where the phase WRITES) or the shared output (where the
        // next phase READS the sum).
        auto aim = [&](int r, int t, K3LayerPhase phase, int width, int slot, bool in) {
            if (!coll_owned) return;
            float* base_p = in ? (float*)p.coll->reduce_in_slot(r, slot)
                               : (float*)p.coll->reduce_out(r);
            if (!base_p) return;
            kimi_k3_swap_partial_buffer(p.ranks[(size_t)r].fwd, phase,
                                        base_p + (size_t)t * (size_t)width);
        };
        // One call for the whole tile, or as few as the owned buffers' capacity admits.
        auto reduce_tile = [&](int width, int slot) -> bool {
            const size_t cap = p.coll->max_count();
            int per_call = T;
            if (cap > 0 && (size_t)T * (size_t)width > cap) {
                per_call = (int)(cap / (size_t)width);
                if (per_call < 1) return false;      // one token does not even fit
            }
            for (int off = 0; off < T; off += per_call) {
                const int m = std::min(per_call, T - off);
                // The payload starts at the slot's base, so a sliced reduce has to begin
                // at token `off` — which the owned entry point cannot express. Slicing is
                // therefore only correct when it does not slice: capacity is sized
                // per_tok * T at init precisely so this is one call.
                if (off != 0 || m != T) return false;
                if (!p.coll->allreduce_f32_owned_slot((size_t)m * (size_t)width,
                                                      p.streams, slot))
                    return false;
                ++p.n_collectives;
                ++coll_k;
            }
            return true;
        };

        // ---- pass 1: every token's attention, still in token order -----------------
        //
        // The position advance rides INSIDE the job rather than following it on the host:
        // the worker already has this rank's device current, and enqueueing onto another
        // rank's stream from the wrong device is an "invalid resource handle".
        //
        // RELATIVE, never a set. +1 between tokens and -(T-1) after the last, which
        // returns the position to `base` so the next layer starts where this one did. An
        // absolute set would bake this tile's base into a captured graph and rewind every
        // replayed tile to the one it was recorded at — fluent output over wrong KV rows.
        for (int t = 0; t < T; ++t) {
            set_tok(t);
            if (!issue_all([&](int r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    K3PrefillTile& tl = pool.tiles[(size_t)r];
                    pick_slot(r, t);
                    if (attn_reduce) aim(r, t, K3LayerPhase::Attn, w_attn, slot_attn, true);
                    if (!kimi_k3_forward_layer_phase(R.fwd, layer, K3LayerPhase::Attn,
                                                     tl.x + (size_t)t * H,
                                                     tl.x_next + (size_t)t * H))
                        return false;
                    if (R.state.d_pos)
                        k3k::k3_add_pos(R.state.d_pos, (t + 1 < T) ? 1 : -(T - 1),
                                        R.stream);
                    return true;
                })) return false;
            save_ckpt(t);
        }
        if (attn_reduce && !reduce_tile(w_attn, slot_attn)) {
            std::fprintf(stderr, "[k3-prefill] attention all-reduce failed at layer %d\n",
                         layer);
            return false;
        }

        // ---- pass 2: every token's FFN partial --------------------------------------
        // Reads its attention sum out of reduce_out and writes its MoE partial into the
        // next rotation slot's input. Different regions, so both aims coexist.
        for (int t = 0; t < T; ++t) {
            set_tok(t);
            if (!issue_all([&](int r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    K3PrefillTile& tl = pool.tiles[(size_t)r];
                    pick_slot(r, t);
                    if (attn_reduce) aim(r, t, K3LayerPhase::Attn, w_attn, slot_attn, false);
                    if (is_moe && tp_size > 1)
                        aim(r, t, K3LayerPhase::FfnPartial, w_moe, slot_moe, true);
                    return kimi_k3_forward_layer_phase(R.fwd, layer,
                                                       K3LayerPhase::FfnPartial,
                                                       tl.x + (size_t)t * H,
                                                       tl.x_next + (size_t)t * H);
                })) return false;
            save_ckpt(t);
        }
        // The leading dense layer's FFN partial is replicated, never reduced, which is
        // why this is gated on is_moe exactly as the token-major loop was.
        if (is_moe && tp_size > 1 && !reduce_tile(w_moe, slot_moe)) {
            std::fprintf(stderr, "[k3-prefill] FFN all-reduce failed at layer %d\n",
                         layer);
            return false;
        }

        // ---- pass 3: every token's FFN finish ---------------------------------------
        //
        // NOTE the absence of the token driver's std::swap(R.x, R.x_next). The tile's
        // pair is swapped ONCE PER LAYER, below, after all T tokens have used it —
        // swapping here would hand token t+1 the buffer token t is still reading.
        for (int t = 0; t < T; ++t) {
            set_tok(t);
            if (!issue_all([&](int r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    K3PrefillTile& tl = pool.tiles[(size_t)r];
                    pick_slot(r, t);
                    if (is_moe && tp_size > 1)
                        aim(r, t, K3LayerPhase::FfnPartial, w_moe, slot_moe, false);
                    return kimi_k3_forward_layer_phase(R.fwd, layer,
                                                       K3LayerPhase::FfnFinish,
                                                       tl.x + (size_t)t * H,
                                                       tl.x_next + (size_t)t * H);
                })) return false;
            save_ckpt(t);
        }
        // One swap for the whole tile, after every token has finished the layer.
        for (auto& tl : pool.tiles) k3_prefill_tile_swap(tl);
    }

    // The tile is ingested: every rank's device position is back at `base` (the layer
    // loop restored it T-1 times), so advance it once, by T, to the first position the
    // NEXT call will write.
    //
    // INSIDE the captured region. This is the single step that makes a replay a
    // DIFFERENT tile rather than the same one again — without it every replayed tile
    // would rewrite the same T KV rows and attend over a frozen context.
    if (!replaying && !issue_all([&](int r) {
            KimiK3TPRank& R = p.ranks[(size_t)r];
            if (R.state.d_pos) k3k::k3_add_pos(R.state.d_pos, T, R.stream);
            return true;
        })) return false;

    // Close the capture and instantiate. Nothing has EXECUTED yet on a capture tile —
    // capture records — so the launch below is what runs this tile's work, on the
    // capture pass and on every replay alike.
    if (capturing) {
        bool ok_cap = true;
        for (std::size_t r = 0; r < p.ranks.size(); ++r) {
            if (cudaSetDevice(p.ranks[r].device) != cudaSuccess) { ok_cap = false; break; }
            if (cudaStreamEndCapture(p.ranks[r].stream, &pool.graph[r]) != cudaSuccess ||
                pool.graph[r] == nullptr) { ok_cap = false; break; }
            if (cudaGraphInstantiate(&pool.exec[r], pool.graph[r], nullptr, nullptr, 0)
                != cudaSuccess) { ok_cap = false; break; }
        }
        if (!ok_cap) {
            // A FAILED CAPTURE MUST COST SPEED, NEVER CORRECTNESS — the same rule the
            // decode step follows. Give up on capture for the rest of the process and
            // re-issue this tile eagerly; the KV cache has not been written, because
            // capture RECORDS rather than executes.
            std::fprintf(stderr, "[k3-prefill] capture failed — eager for the rest of "
                                 "the run\n");
            for (std::size_t r = 0; r < p.ranks.size(); ++r) {
                cudaSetDevice(p.ranks[r].device);
                if (pool.exec[r])  { cudaGraphExecDestroy(pool.exec[r]);  pool.exec[r]  = nullptr; }
                if (pool.graph[r]) { cudaGraphDestroy(pool.graph[r]);     pool.graph[r] = nullptr; }
            }
            cudaGetLastError();          // clear the sticky capture error
            pool.graph_disabled = true;
            pool.graph_ready    = false;
            return k3_tp_prefill_tile(p, ids, T);
        }
        pool.graph_ready   = true;
        pool.captured_plan = plan_lo;
        ++pool.n_captures;
        size_t nnodes = 0;
        cudaGraphGetNodes(pool.graph[0], nullptr, &nnodes);
        std::fprintf(stderr, "[k3-prefill] captured tile: %zu nodes/rank, T=%d, "
                             "mla splits=%d, from position %d\n",
                     nnodes, T, plan_lo, base);
        // SPARKINFER_K3_PREFILL_NODES=1 prints WHICH kernels those nodes are.
        //
        // The tile only beats the per-token path if a T-token tile records about what one
        // token records today — capture is worth 1.93x on a 3308-node graph and 1.56x on a
        // 61258-node one — so the batching order is decided by the node histogram and
        // nothing else. Reading it off the layer body would be guessing, and nsys cannot
        // run in this container (it dies enumerating /sys/devices/virtual/nvidia-pci-gpu),
        // so the count is taken from the captured graph itself, which is the authority
        // anyway: these ARE the nodes that get replayed.
        //
        // Diagnostic only, off by default, and it runs ONCE per capture rather than per
        // replay, so it cannot perturb the measurement it exists to direct.
        if (const char* e = std::getenv("SPARKINFER_K3_PREFILL_NODES")) if (e[0] == '1') {
            std::vector<cudaGraphNode_t> nodes(nnodes);
            if (cudaGraphGetNodes(pool.graph[0], nodes.data(), &nnodes) == cudaSuccess) {
                std::map<std::string, long> hist;
                long other = 0;
                for (size_t i = 0; i < nnodes; ++i) {
                    cudaGraphNodeType ty{};
                    if (cudaGraphNodeGetType(nodes[i], &ty) != cudaSuccess) continue;
                    if (ty == cudaGraphNodeTypeKernel) {
                        cudaKernelNodeParams kp{};
                        const char* nm = nullptr;
                        if (cudaGraphKernelNodeGetParams(nodes[i], &kp) == cudaSuccess &&
                            cudaFuncGetName(&nm, kp.func) == cudaSuccess && nm)
                            ++hist[nm];
                        else ++hist["<kernel:unnamed>"];
                    } else if (ty == cudaGraphNodeTypeMemcpy) {
                        ++hist["<memcpy>"];
                    } else if (ty == cudaGraphNodeTypeMemset) {
                        ++hist["<memset>"];
                    } else {
                        ++other;
                    }
                }
                std::vector<std::pair<long, std::string>> rows;
                rows.reserve(hist.size());
                for (auto& kv : hist) rows.push_back({kv.second, kv.first});
                std::sort(rows.begin(), rows.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
                std::fprintf(stderr, "[k3-nodes] T=%d, %zu nodes, %zu distinct, "
                                     "%ld non-kernel-non-copy\n",
                             T, nnodes, rows.size(), other);
                for (auto& r : rows)
                    std::fprintf(stderr, "[k3-nodes] %8ld  %7.2f/token  %s\n",
                                 r.first, (double)r.first / (double)T, r.second.c_str());
            }
        }
    }

    if (graph_on) {
        // Launched through the pool so all eight go concurrently rather than in rank
        // order: rank 7 starting after seven launches have gone ahead of it is a clean
        // submission-order gradient the collective then charges everyone for.
        if (!issue_all([&](int r) {
                return cudaGraphLaunch(pool.exec[(size_t)r],
                                       p.ranks[(size_t)r].stream) == cudaSuccess;
            })) {
            std::fprintf(stderr, "[k3-prefill] graph launch failed\n");
            return false;
        }
        ++pool.n_replays;
    }

    // Hand every rank back its own state before returning. A decode step that inherited
    // the tile's residual bank would read memory this driver owns, and one that inherited
    // a live prefill_tile pointer would take q/k/v/g rows belonging to a prompt token.
    for (int r = 0; r < tp_size; ++r) {
        KimiK3TPRank& R = p.ranks[(size_t)r];
        R.state.position = base + T;
        if (pool.orig_bank[(size_t)r]) R.state.res_bank = pool.orig_bank[(size_t)r];
        R.state.n_ckpt = 0;
        R.fwd.prefill_tile = nullptr;
        R.fwd.prefill_tok = 0;
        pool.tiles[(size_t)r].layer = -1;
    }
    return true;
}

bool kimi_k3_tp_prefill(KimiK3TP& p, const int* ids, int n_ids, float* out_logits) {
    if (p.ranks.empty() || !p.coll || !ids || n_ids <= 0) return false;

    // SPARKINFER_K3_PREFILL=0 declines the tile path on the same binary, so the whole
    // driver can be A/B'd without a rebuild — every reliable measurement on this branch
    // has come from a same-binary control.
    static const bool want_tile = [] {
        const char* e = std::getenv("SPARKINFER_K3_PREFILL");
        return !(e && e[0] == '0');
    }();
    static const int want_T = [] {
        const char* e = std::getenv("SPARKINFER_K3_PREFILL_TILE");
        const int v = e ? std::atoi(e) : 4;
        return v > 0 ? v : 4;
    }();

    // The LAST token goes through the per-token path, which owns the head — so this
    // driver never duplicates the banded output projection or the collective around it.
    // Tail tokens do too: a tile of 2 is barred by the rotation check
    // (k3_prefill_pick_tile), and ≤3 tokens out of a 32k prompt is not worth a second
    // ragged code path that would need its own proof.
    if (want_tile && n_ids > 1 && p.ranks.size() > 1 && p.prefill == nullptr)
        k3_prefill_pool_init(p, want_T);

    const int T = (want_tile && p.prefill) ? p.prefill->tile : 0;
    const int n_tiled = T > 0 ? ((n_ids - 1) / T) * T : 0;

    for (int i = 0; i < n_tiled; i += T) {
        // No fallback once a tile has started: it has already written KV rows and
        // advanced the KDA recurrence, so re-running those tokens per-token would
        // double-ingest them. Failing loudly is the only honest option left here.
        if (!k3_tp_prefill_tile(p, ids + i, T)) {
            std::fprintf(stderr, "[k3-prefill] tile at token %d failed; the KV cache is "
                                 "already part-written, so there is no fallback\n", i);
            return false;
        }
    }
    for (int i = n_tiled; i < n_ids; ++i)
        if (!kimi_k3_tp_forward_token(p, ids[i], out_logits)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Chunked prompt ingestion
// ---------------------------------------------------------------------------
namespace {

// Grow (once) the per-rank chunk buffers and the per-rank batch arena.
bool prompt_alloc(KimiK3TP& p, int B) {
    const KimiK3Config& cfg = p.cfg;
    const int H = cfg.hidden;
    for (auto& R : p.ranks) {
        if (R.b_cap >= B) continue;
        if (R.b_cap > 0) return false;   // regrow would strand the old allocations
        if (cudaSetDevice(R.device) != cudaSuccess) return false;
        void* a = nullptr; void* b = nullptr; void* c = nullptr;
        if (cudaMalloc(&a, (size_t)B * H * sizeof(float)) != cudaSuccess) return false;
        if (cudaMalloc(&b, (size_t)B * H * sizeof(float)) != cudaSuccess) return false;
        if (cudaMalloc(&c, sizeof(int)) != cudaSuccess) return false;
        R.xb = (float*)a; R.xnb = (float*)b; R.d_pos_base = (int*)c;
        R.b_cap = B;
        // The forward's own arena: attn_out / moe_fused / ffn_out / res_bank, xB.
        // fwd.state must already be wired, because the residual bank is sized from
        // state->max_ckpt — allocating before that silently gives a zero-row bank
        // and every checkpoint push lands out of bounds.
        if (!R.fwd.state) return false;
        if (!kimi_k3_forward_alloc_batch(cfg, R.fwd, B)) return false;
    }
    return true;
}

}  // namespace

bool kimi_k3_tp_forward_prompt(KimiK3TP& p, const int* ids, int n,
                               float* out_logits) {
    if (p.ranks.empty() || !p.coll || !ids || n <= 0) return false;
    const KimiK3Config& cfg = p.cfg;
    const int H = cfg.hidden;
    const int tp_size = (int)p.ranks.size();

    static const int want_B = [] {
        const char* e = std::getenv("SPARKINFER_K3_PREFILL_BATCH");
        const int v = e ? std::atoi(e) : 1;
        return v > 0 ? v : 1;
    }();
    // B == 1 IS NOT SHORT-CIRCUITED TO forward_token. It runs the chunked walk
    // through a one-slot arena so that "B=1 matches main" is evidence about THIS
    // code, not about code it bypassed.
    const int B = want_B;

    if (!prompt_alloc(p, B)) {
        std::fprintf(stderr, "[k3-tp] prefill: chunk arena alloc failed (B=%d)\n", B);
        return false;
    }

    // GRAPH CAPTURE IS OFF ON THIS PATH, AND IT IS NOT FREE. Measured 53.10 tok/s
    // captured against 29.30 uncaptured at 1k context: the graph is worth ~1.8x,
    // so a chunk has to earn that back before it earns anything. The walk below is
    // deliberately a CONSTANT shape for a full chunk (same layers, same phase
    // order, same B) precisely so it can be captured later; the ragged tail is the
    // only variable part and it is split off into its own pass.
    const bool serial_issue = std::getenv("SPARKINFER_K3_SERIAL_ISSUE") != nullptr;
    const bool parallel_issue = (tp_size > 1) && !serial_issue;
    if (parallel_issue && !p.issue.started()) {
        std::vector<int> devs;
        devs.reserve(p.ranks.size());
        for (auto& R : p.ranks) devs.push_back(R.device);
        p.issue.start(devs);
    }
    auto issue_all = [&](const std::function<bool(int)>& job) -> bool {
        if (parallel_issue) return p.issue.run(job);
        for (int r = 0; r < tp_size; ++r) {
            if (cudaSetDevice(p.ranks[(size_t)r].device) != cudaSuccess) return false;
            if (!job(r)) return false;
        }
        return true;
    };

    // Reduce a chunk's partials IN SLICES THAT FIT THE COLLECTIVE'S CAPACITY.
    //
    // The obvious move — size the collective B-fold at init — does not survive
    // contact with this model: its owned buffers are peer-mapped and replicated per
    // rank per slot, and multiplying them by 8 made the WEIGHT load fail at layer
    // 68 on an idle box. There is no headroom to buy.
    //
    // So the payload is sliced instead. Slots are contiguous, so slice s covers
    // tokens [s*k, (s+1)*k) and one call still carries k tokens rather than one —
    // the collective count drops from B per layer to ceil(B/k), which is the whole
    // point, just bounded by what fits. cap == 0 means the backend states no limit
    // (NCCL reduces in place) and the chunk goes in one call.
    auto reduce_chunk = [&](int layer, K3LayerPhase phase, int Bc) -> bool {
        int count = 0;
        std::vector<float*> bases((size_t)tp_size, nullptr);
        for (int r = 0; r < tp_size; ++r) {
            int nn = 0;
            float* buf = kimi_k3_batch_partial_buffer(p.ranks[(size_t)r].fwd, layer,
                                                      phase, Bc, &nn);
            if (!buf || nn <= 0) return false;
            if (r == 0) count = nn; else if (nn != count) return false;
            bases[(size_t)r] = buf;
        }
        const int per_tok = count / Bc;              // exact: the arena is [B][width]
        const size_t cap = p.coll->max_count();
        int per_call = Bc;
        if (cap > 0 && (size_t)count > cap) {
            per_call = (int)(cap / (size_t)per_tok);
            if (per_call < 1) return false;          // one token does not even fit
        }
        for (int off = 0; off < Bc; off += per_call) {
            const int m = std::min(per_call, Bc - off);
            for (int r = 0; r < tp_size; ++r)
                p.reduce_bufs[(size_t)r] = bases[(size_t)r] + (size_t)off * per_tok;
            if (!p.coll->allreduce_f32_group(p.reduce_bufs,
                                             (size_t)m * per_tok, p.streams))
                return false;
            ++p.n_collectives;
        }
        return true;
    };

    for (int base = 0; base < n; base += B) {
        const int Bc = std::min(B, n - base);
        const int pos0 = p.ranks[0].state.position;

        // ---- embed the chunk, and plant the chunk's base position -------------
        if (!issue_all([&](int r) {
                KimiK3TPRank& R = p.ranks[(size_t)r];
                for (int t = 0; t < Bc; ++t)
                    if (!embed_token(R.weights, cfg, ids[base + t],
                                     R.xb + (size_t)t * H, R.stream)) return false;
                if (R.d_pos_base &&
                    cudaMemcpyAsync(R.d_pos_base, &pos0, sizeof(int),
                                    cudaMemcpyHostToDevice, R.stream) != cudaSuccess)
                    return false;
                return true;
            })) return false;

        for (auto& R : p.ranks) kimi_k3_forward_batch_begin(R.fwd);

        for (int layer = 0; layer < cfg.n_layers; ++layer) {
            const bool kda_reduce = tp_size > 1 && cfg.is_kda_layer(layer) &&
                KimiK3Weights::shards_kda(p.ranks[0].weights.policy);
            const bool mla_reduce = tp_size > 1 && !cfg.is_kda_layer(layer) &&
                KimiK3Weights::shards_mla(p.ranks[0].weights.policy);
            const bool attn_reduce = kda_reduce || mla_reduce;

            // RESET THE DEVICE POSITION TO THE CHUNK BASE, ONCE PER LAYER.
            //
            // The token loop only ever moves position forward, so it bumps. A
            // layer-major walk revisits the same B positions on EVERY layer, so it
            // has to rewind — and rewinding from the host would be a synchronising
            // 4-byte copy per layer per rank on the critical path. d_pos_base holds
            // the chunk's first position in device memory and this is a D2D copy,
            // stream-ordered and free.
            if (!issue_all([&](int r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    if (!R.state.d_pos || !R.d_pos_base) return true;
                    return cudaMemcpyAsync(R.state.d_pos, R.d_pos_base, sizeof(int),
                                           cudaMemcpyDeviceToDevice,
                                           R.stream) == cudaSuccess;
                })) return false;

            // ---- phase 1, token by token ------------------------------------
            if (!issue_all([&](int r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    for (int t = 0; t < Bc; ++t) {
                        kimi_k3_forward_select_slot(R.fwd, t);
                        // Host mirror only; the device side was planted above and is
                        // advanced by the bump below. Assigning the field directly
                        // rather than via kimi_k3_set_position is what keeps this off
                        // the synchronising path.
                        R.state.position = pos0 + t;
                        if (!kimi_k3_forward_layer_phase(R.fwd, layer,
                                                         K3LayerPhase::Attn,
                                                         R.xb + (size_t)t * H,
                                                         R.xnb + (size_t)t * H))
                            return false;
                        // Only attention indexes the KV store, so the position moves
                        // here and not around the FFN phases.
                        if (R.state.d_pos) k3k::k3_bump_pos(R.state.d_pos, R.stream);
                    }
                    return true;
                })) return false;

            // ---- the chunk's attention partials, in capacity-sized slices -----
            if (attn_reduce && !reduce_chunk(layer, K3LayerPhase::Attn, Bc)) {
                std::fprintf(stderr,
                             "[k3-tp] prefill: attention all-reduce failed at "
                             "layer %d\n", layer);
                return false;
            }

            // ---- phase 2, token by token ------------------------------------
            if (!issue_all([&](int r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    for (int t = 0; t < Bc; ++t) {
                        kimi_k3_forward_select_slot(R.fwd, t);
                        R.state.position = pos0 + t;
                        if (!kimi_k3_forward_layer_phase(R.fwd, layer,
                                                         K3LayerPhase::FfnPartial,
                                                         R.xb + (size_t)t * H,
                                                         R.xnb + (size_t)t * H))
                            return false;
                    }
                    return true;
                })) return false;

            // ---- the chunk's FFN partials, in capacity-sized slices -----------
            if (tp_size > 1 &&
                !reduce_chunk(layer, K3LayerPhase::FfnPartial, Bc)) {
                std::fprintf(stderr,
                             "[k3-tp] prefill: FFN all-reduce failed at layer %d\n",
                             layer);
                return false;
            }

            // ---- phase 3, token by token ------------------------------------
            if (!issue_all([&](int r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    for (int t = 0; t < Bc; ++t) {
                        kimi_k3_forward_select_slot(R.fwd, t);
                        R.state.position = pos0 + t;
                        if (!kimi_k3_forward_layer_phase(R.fwd, layer,
                                                         K3LayerPhase::FfnFinish,
                                                         R.xb + (size_t)t * H,
                                                         R.xnb + (size_t)t * H))
                            return false;
                    }
                    return true;
                })) return false;
            // ONE swap for the whole chunk. Per-token buffers are slices of the same
            // pair, so swapping the bases moves all Bc of them at once — and unlike
            // the token loop this walk is not inside a captured graph, so the
            // alternating parity that x_canon exists to defend against cannot bite.
            for (auto& R : p.ranks) std::swap(R.xb, R.xnb);
        }

        // ---- advance the host position past the chunk ------------------------
        for (auto& R : p.ranks) {
            kimi_k3_forward_batch_end(R.fwd);
            R.state.position = pos0 + Bc;
        }
    }

    // ---- the head, on the LAST token of the prompt only ---------------------
    //
    // Prefill needs one distribution: the one after the final prompt token. The
    // token loop pays a 1.25 GB output projection 32,768 times to throw 32,767 of
    // them away. Rank 0 alone is enough here — every rank holds the identical
    // hidden state at this point, and this runs once per prompt rather than once
    // per token, so banding it would save nothing worth the code.
    KimiK3TPRank& R0 = p.ranks[0];
    const KimiK3Weights& w = R0.weights;
    const int last = (n - 1) % B;
    if (cudaSetDevice(R0.device) != cudaSuccess) return false;
    {
        float* xl = R0.xb + (size_t)last * H;
        float* xo = R0.xnb + (size_t)last * H;
        if (cfg.attn_res_block_size > 0) {
            if (!w.has_output_res_score || !w.output_res_score.ok()) return false;
            // The bank for the LAST token, which is the slot the head reads. Selecting
            // it also restores that slot's n_ckpt, which attn_res_mix lengths its
            // softmax with — reading another token's count here mixes the wrong prefix.
            kimi_k3_forward_select_slot(R0.fwd, last);
            k3k::attn_res_mix_f32(xo, R0.state.res_bank, xl,
                                  (const float*)w.output_res_score.data, H,
                                  R0.state.n_ckpt, cfg.rms_eps, R0.stream);
            std::swap(xl, xo);
        }
        if (!w.output_norm.ok() || !w.output.ok()) return false;
        k3k::rms_norm_f32(xo, xl, (const float*)w.output_norm.data, H,
                          cfg.rms_eps, R0.stream);
        if (!k3k::k3_proj_f32(R0.logits, xo, w.output.data, w.output.type,
                              cfg.vocab, H, R0.stream))
            return false;
    }
    if (cudaStreamSynchronize(R0.stream) != cudaSuccess) return false;
    for (auto& R : p.ranks) kimi_k3_forward_batch_end(R.fwd);
    if (out_logits &&
        cudaMemcpy(out_logits, R0.logits, (size_t)cfg.vocab * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    return true;
}

void kimi_k3_tp_free(KimiK3TP& p) {
    // Join the submission threads BEFORE any rank buffer is freed. They only
    // spin while parked, but a worker that is still inside a phase call would
    // be launching against memory this loop is about to release. This also has to
    // precede the pointer restore below, for the same reason the swaps are done
    // by the submitting thread: no worker may be reading fwd.s while it changes.
    p.issue.shutdown();

    // Point the scratch field back at scratch before teardown. Not load-bearing
    // today (scratch frees via its owned list, not this field) but keeps the
    // struct truthful for anything that walks it during shutdown.
    if (p.zero_copy) {
        for (std::size_t r = 0; r < p.ranks.size(); ++r) {
            kimi_k3_swap_partial_buffer(p.ranks[r].fwd, K3LayerPhase::FfnPartial,
                                        p.orig_moe[r]);
            kimi_k3_swap_partial_buffer(p.ranks[r].fwd, K3LayerPhase::Attn,
                                        p.orig_attn[r]);
        }
        p.zero_copy = false;
    }
    // Before any rank buffer goes: the pool holds tile allocations on those devices and
    // a saved res_bank pointer it has to hand back to the state.
    k3_prefill_pool_free(p.prefill, p.ranks);

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
    p.zc_in.clear();
    p.zc_in_slot.clear();
    p.n_coll_slots = 1;
    p.zc_out.clear();
    p.orig_moe.clear();
    p.orig_attn.clear();
    p.coll.reset();
}

}  // namespace sparkinfer
