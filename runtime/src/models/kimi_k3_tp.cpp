// Tensor-parallel Kimi K3 decode. See kimi_k3_tp.h for the scope and the reasoning.

#include "sparkinfer/models/kimi_k3_tp.h"

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/models/k3_head_band.h"
#include "sparkinfer/tp/k3_coll_1bar.h"
#include "sparkinfer/tp/shard.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
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
                    const std::vector<int>& devices, int max_ctx, KimiK3TP& out,
                    int prefill_chunk) {
    const int tp_size = (int)devices.size();
    if (tp_size < 1) return false;
    // Clamped, not trusted. The chunk multiplies the collective's peer allocation and
    // the per-rank hidden/bank buffers, and a caller passing a large number would turn
    // a typo into an out-of-memory at rank 6 of 8 after a 150 s load.
    if (prefill_chunk < 0) prefill_chunk = 0;
    if (prefill_chunk > 64) prefill_chunk = 64;
    out.prefill_chunk = prefill_chunk > 1 ? prefill_chunk : 0;
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
    const size_t max_count  = attn_count > moe_count ? attn_count : moe_count;
    // CHUNKED PREFILL REDUCES B TOKENS' PARTIALS IN ONE CALL, so the peer buffers have
    // to hold B of them side by side — that adjacency IS the optimisation, and without
    // it the chunk path would be B separate reduces and worth nothing. At K3's
    // max_count of 10752 floats a chunk of 16 costs 688 KB of output per rank (and the
    // same again per rotating input slot), which is noise against 124 GiB of weights.
    //
    // Multiplied only when a chunk was asked for: a decode-only run allocates exactly
    // what main allocates, so the scored path cannot move because this exists.
    const size_t coll_count = max_count * (size_t)(out.prefill_chunk > 1
                                                   ? out.prefill_chunk : 1);
    out.coll = tp::make_collective(devices, requested, &err, coll_count,
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
    // Gated on ONE token's payload, as it always was. A backend that could not give the
    // chunk its bigger allocation must still run zero-copy decode — the chunk path is
    // what stands down in that case (checked separately, below), never the scored path.
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

    // ---- chunked prefill buffers ------------------------------------------
    //
    // Stood down rather than half-built. Every reason below leaves prefill_chunk at 0,
    // which makes kimi_k3_tp_forward_prompt a forward_token loop — main's behaviour,
    // the same numbers, just not faster. A chunk path that ran on a collective whose
    // buffers hold one token would silently reduce garbage past the first.
    if (out.prefill_chunk > 1) {
        const int B = out.prefill_chunk;
        const char* why = nullptr;
        if (tp_size <= 1)                             why = "tp_size 1";
        else if (!out.zero_copy)                      why = "collective does not own its buffers";
        else if (out.coll->max_count() < coll_count)  why = "collective could not size for the chunk";
        if (why) {
            std::fprintf(stderr, "[k3-tp] chunked prefill OFF (%s) — prompt ingestion "
                                 "stays one forward_token per token\n", why);
            out.prefill_chunk = 0;
        } else {
            bool ok = true;
            for (int r = 0; r < tp_size && ok; ++r) {
                KimiK3TPRank& R = out.ranks[(size_t)r];
                if (cudaSetDevice(R.device) != cudaSuccess) { ok = false; break; }
                R.xb.assign((size_t)B, nullptr);
                R.xb_next.assign((size_t)B, nullptr);
                R.res_bank_b.assign((size_t)B, nullptr);
                const size_t bank_elems = (size_t)cfg.hidden * (size_t)R.state.max_ckpt;
                for (int b = 0; b < B && ok; ++b) {
                    if (cudaMalloc(&R.xb[(size_t)b], (size_t)cfg.hidden * sizeof(float))
                            != cudaSuccess ||
                        cudaMalloc(&R.xb_next[(size_t)b], (size_t)cfg.hidden * sizeof(float))
                            != cudaSuccess) { ok = false; break; }
                    // max_ckpt is 0 when the cross-layer residual is disabled; the mix
                    // never runs then and a null bank is what the single-token path
                    // already carries.
                    if (bank_elems &&
                        cudaMalloc(&R.res_bank_b[(size_t)b], bank_elems * sizeof(float))
                            != cudaSuccess) { ok = false; break; }
                }
                if (ok && cudaMalloc(&R.d_pos_chunk, (size_t)B * sizeof(int)) != cudaSuccess)
                    ok = false;
                // The allocation order IS the canonical order; every chunk is reset to it.
                R.xb_canon      = R.xb;
                R.xb_next_canon = R.xb_next;
                R.d_pos_single    = R.state.d_pos;
                R.res_bank_single = R.state.res_bank;
            }
            if (!ok) {
                std::fprintf(stderr, "[k3-tp] chunked prefill OFF (per-rank buffers "
                                     "failed to allocate)\n");
                for (auto& R : out.ranks) {
                    cudaSetDevice(R.device);
                    if (R.chunk_exec)  { cudaGraphExecDestroy(R.chunk_exec); R.chunk_exec  = nullptr; }
                    if (R.chunk_graph) { cudaGraphDestroy(R.chunk_graph);    R.chunk_graph = nullptr; }
                    for (auto* q : R.xb)         if (q) cudaFree(q);
                    for (auto* q : R.xb_next)    if (q) cudaFree(q);
                    for (auto* q : R.res_bank_b) if (q) cudaFree(q);
                    if (R.d_pos_chunk) cudaFree(R.d_pos_chunk);
                    R.xb.clear(); R.xb_next.clear(); R.res_bank_b.clear();
                    R.d_pos_chunk = nullptr;
                }
                out.prefill_chunk = 0;
            } else {
                std::fprintf(stderr, "[k3-tp] chunked prefill: %d tokens/pass, one "
                                     "collective per layer-phase instead of %d\n", B, B);
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// The output head, factored so the decode step and chunked prefill share ONE copy.
//
// Extracted rather than duplicated on purpose. This is the only place the residual
// mix, the output norm and the 1.25 GB vocab projection are written down, and a second
// copy for the prefill path would be free to drift — silently, because ingestion's
// logits are compared against a reference at only a handful of depths.
// ---------------------------------------------------------------------------
using K3IssueAll = std::function<bool(const std::function<bool(int)>&)>;

// Decided once from rank 0, and every rank is then required to agree: a band that
// resolved on some ranks and not others would leave a hole in the logits rather than a
// slow token.
static bool k3_tp_head_banded(const KimiK3TP& p) {
    const int tp_size = (int)p.ranks.size();
    const KimiK3Weights& w = p.ranks[0].weights;
    K3HeadBand hb0;
    if (!w.output.ok() ||
        !k3_head_band(p.cfg.vocab, (size_t)w.output.n_bytes, tp_size, 0, &hb0))
        return false;
    for (int r = 1; r < tp_size; ++r) {
        K3HeadBand hbr;
        const KimiK3Weights& wr = p.ranks[(size_t)r].weights;
        if (p.ranks[(size_t)r].logits && wr.output.ok() &&
            wr.output.n_bytes == w.output.n_bytes &&
            k3_head_band(p.cfg.vocab, (size_t)wr.output.n_bytes, tp_size, r, &hbr))
            continue;
        return false;
    }
    return true;
}

// Issue the head on every rank. Reads R.x, leaves the band's logits in R.logits.
// Enqueue only — the caller decides when to wait.
static bool k3_tp_issue_head(KimiK3TP& p, bool band_head) {
    const KimiK3Config& cfg = p.cfg;
    const int H = cfg.hidden;
    const int tp_size = (int)p.ranks.size();
    KimiK3TPRank& R0 = p.ranks[0];
    const KimiK3Weights& w = R0.weights;

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
        return cudaSetDevice(R0.device) == cudaSuccess;
    }

    if (cudaSetDevice(R0.device) != cudaSuccess) return false;
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
    return k3k::k3_proj_f32(R0.logits, R0.x_next, w.output.data, w.output.type,
                            cfg.vocab, H, R0.stream);
}

// Wait for the head and bring the logits back to the host.
static bool k3_tp_copy_logits(KimiK3TP& p, bool band_head, float* out_logits,
                       const K3IssueAll& issue_all) {
    const KimiK3Config& cfg = p.cfg;
    const int tp_size = (int)p.ranks.size();
    if (band_head) {
        // Eight disjoint 82 KB copies instead of one 655 KB copy, dispatched through
        // the pool so they run on eight threads with eight devices current rather
        // than serialising behind rank 0. The sync moves inside for the same reason:
        // the ranks finish within a few microseconds of each other (the last layer's
        // all-reduce is a rendezvous), so waiting on them in parallel costs about
        // what waiting on rank 0 alone used to.
        //
        // PIN THE LANDING BUFFER, once, lazily. The caller's out_logits is pageable
        // (the bench hands us a std::vector), so 8 workers each pay a driver
        // bounce-buffer staging on every token. cudaHostRegister makes the same bytes
        // DMA-able in place; cached by pointer so the cost is one-time. Same bytes,
        // same destination — bit-identical trivially.
        // SPARKINFER_K3_PIN_LOGITS=0 opts out.
        //
        // This lives HERE, in the shared copy, rather than in forward_token: chunked
        // prefill lands logits through the same path at every dump depth, and a copy
        // of this that only the decode driver had would leave prefill on the pageable
        // bounce for exactly the transfers the chunk path exists to make rarer.
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
        return issue_all([&](int r) {
            KimiK3TPRank& R = p.ranks[(size_t)r];
            K3HeadBand hb;
            if (!k3_head_band(cfg.vocab, (size_t)R.weights.output.n_bytes,
                              tp_size, r, &hb)) return false;
            if (cudaStreamSynchronize(R.stream) != cudaSuccess) return false;
            return cudaMemcpy(out_logits + hb.offset, R.logits,
                              (size_t)hb.rows * sizeof(float),
                              cudaMemcpyDeviceToHost) == cudaSuccess;
        });
    }
    KimiK3TPRank& R0 = p.ranks[0];
    if (cudaSetDevice(R0.device) != cudaSuccess) return false;
    if (cudaStreamSynchronize(R0.stream) != cudaSuccess) return false;
    return cudaMemcpy(out_logits, R0.logits, (size_t)cfg.vocab * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool kimi_k3_tp_forward_token(KimiK3TP& p, int token_id, float* out_logits) {
    if (p.ranks.empty() || !p.coll) return false;
    const KimiK3Config& cfg = p.cfg;
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
        //
        // SHARD THE ROUTED UP-PROJECTION, when there is a collective to stitch it with.
        // ffn_routed_up is Replicate, so the plain FfnFinish below has all eight ranks
        // computing the identical 7168x3584 result — 2.51 GB/token/rank over 92 layers,
        // seven eighths of it redundant. FfnUp computes only this rank's band into a
        // zeroed hidden-wide buffer; one all-reduce sums the bands; FfnTail then runs
        // unchanged on the complete value.
        //
        // Costs one extra collective per MoE layer against ~7/8 of the projection. Falls
        // back to FfnFinish for the leading dense layer (no routed_up to shard, so the
        // partial buffer is null) and whenever the band declines.
        const IClock::time_point t_p3 = ip.on ? IClock::now() : IClock::time_point{};
        // The reduce must run over a buffer the COLLECTIVE owns. Aiming s.ffn_out at
        // zc_in, reducing, then aiming it at zc_out is the same dance FfnPartial already
        // does — and skipping it is why the first version of this reduced nothing and left
        // every rank holding only its own band: a wrong model that still returned success.
        int up_count = 0;
        const bool shard_up = k3_routed_up_band_enabled() && tp_size > 1 && p.zero_copy &&
                              layer >= cfg.leading_dense;
        if (shard_up) {
            kimi_k3_partial_buffer(p.ranks[0].fwd, layer, K3LayerPhase::FfnUp, &up_count);
            if (up_count > 0 && (size_t)up_count > p.coll->max_count()) up_count = 0;
        }
        if (shard_up && up_count > 0) {
            // GET THE MoE PAYLOAD OUT OF zc_out FIRST. Slots vary only the reduce INPUT;
            // every reduce writes the same zc_out, and swap_partial_buffer(FfnPartial)
            // left s.shexp_out pointing at zc_out + expert_latent. So the routed_up reduce
            // would land on top of the shared expert that FfnTail is about to fold in —
            // which is exactly what the first two attempts at this did, silently.
            const int moe_w = cfg.expert_latent + cfg.hidden;
            for (int r = 0; r < tp_size; ++r) {
                KimiK3TPRank& R = p.ranks[(size_t)r];
                if (cudaSetDevice(R.device) != cudaSuccess) return false;
                if (cudaMemcpyAsync(p.orig_moe[(size_t)r], p.zc_out[(size_t)r],
                                    (size_t)moe_w * sizeof(float),
                                    cudaMemcpyDeviceToDevice, R.stream) != cudaSuccess)
                    return false;
                kimi_k3_swap_partial_buffer(R.fwd, K3LayerPhase::FfnPartial,
                                            p.orig_moe[(size_t)r]);
            }
            for (int r = 0; r < tp_size; ++r)
                kimi_k3_swap_partial_buffer(p.ranks[(size_t)r].fwd, K3LayerPhase::FfnUp,
                                            p.zc_in[(size_t)r]);
            if (!issue_all([&](int r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    return kimi_k3_forward_layer_phase(R.fwd, layer, K3LayerPhase::FfnUp,
                                                       R.x, R.x_next);
                })) return false;
            if (!p.coll->allreduce_f32_owned_slot((size_t)up_count, p.streams, slot_moe)) {
                std::fprintf(stderr, "[k3-tp] routed_up all-reduce failed at layer %d\n",
                             layer);
                return false;
            }
            ++p.n_collectives;
            ++coll_k;
            for (int r = 0; r < tp_size; ++r)
                kimi_k3_swap_partial_buffer(p.ranks[(size_t)r].fwd, K3LayerPhase::FfnUp,
                                            p.zc_out[(size_t)r]);
        }
        const K3LayerPhase p3 = (shard_up && up_count > 0) ? K3LayerPhase::FfnTail
                                                           : K3LayerPhase::FfnFinish;
        if (!issue_all([&](int r) {
                KimiK3TPRank& R = p.ranks[(size_t)r];
                if (!kimi_k3_forward_layer_phase(R.fwd, layer, p3,
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

    const bool band_head = k3_tp_head_banded(p);

    if (!replaying) {
    if (!k3_tp_issue_head(p, band_head)) return false;

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
    if (!k3_tp_copy_logits(p, band_head, out_logits, issue_all)) return false;
    if (ip.on) ip.t_sync += secs_since(t_s0);

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
// Chunked prefill
// ---------------------------------------------------------------------------
namespace {

// One chunk: B prompt tokens carried through all 93 layers together.
//
// The shape of this function is forward_token's layer loop with a token loop nested
// INSIDE each phase, and the collective lifted OUT of it. That inversion is the whole
// optimisation: 185 rendezvous per chunk rather than per token.
//
// `base` is the position of ids[0]. Every rank's state is re-aimed per token before
// each phase call — position, d_pos slot, residual bank, checkpoint count, hidden pair
// — so kimi_k3_forward_layer_phase runs completely unmodified and cannot tell it is in
// a chunk. That is deliberate: the numerics of a prompt token must not depend on
// whether it was ingested alone or with fifteen neighbours.
bool k3_prefill_chunk(KimiK3TP& p, const int* ids, int base, int B,
                      bool want_logits, float* out_logits,
                      const std::function<bool(const std::function<bool(int)>&)>& issue_all) {
    const KimiK3Config& cfg = p.cfg;
    const int tp_size = (int)p.ranks.size();
    const int n_slots = p.n_coll_slots;

    // Per-token checkpoint counts. n_ckpt is a scalar in KimiK3RuntimeState because a
    // token owns it for its whole descent; here B descents are interleaved by layer, so
    // the live value has to be parked between phases. Every rank advances it
    // identically (the bank push is a host-side decision on `layer % res_bs`), so one
    // array covers all ranks.
    std::vector<int> nck((size_t)B, 0);

    // Position slots: base+0 .. base+B-1, written once. The kernels read *d_pos, so
    // handing token b a pointer to slot b is what makes B positions coexist inside one
    // layer. Nothing bumps them — the chunk owns its range and the next chunk rewrites.
    //
    // Blocking cudaMemcpy, not the async form: the source is a stack vector that dies at
    // the end of this block, and one synchronous 64-byte copy per rank per CHUNK is
    // nothing against the B end-of-token drains this path exists to remove.
    {
        std::vector<int> pos((size_t)B);
        for (int b = 0; b < B; ++b) pos[(size_t)b] = base + b;
        for (int r = 0; r < tp_size; ++r) {
            KimiK3TPRank& R = p.ranks[(size_t)r];
            if (cudaSetDevice(R.device) != cudaSuccess) return false;
            if (cudaMemcpy(R.d_pos_chunk, pos.data(), (size_t)B * sizeof(int),
                           cudaMemcpyHostToDevice) != cudaSuccess)
                return false;
        }
    }

    // Aim rank r's state at token b. Called by the SUBMITTING thread with the issue
    // pool parked, never from a worker: these are the pointers the workers dereference.
    auto aim = [&](KimiK3TPRank& R, int b) {
        R.state.position = base + b;
        R.state.d_pos    = R.d_pos_chunk + b;
        if (R.state.max_ckpt > 0) R.state.res_bank = R.res_bank_b[(size_t)b];
        R.state.n_ckpt   = nck[(size_t)b];
        R.x      = R.xb[(size_t)b];
        R.x_next = R.xb_next[(size_t)b];
    };

    // EVERY CHUNK STARTS FROM THE SAME POINTER ASSIGNMENT. Phase 3 flips a token's pair
    // once per layer and 93 is odd, so a chunk ends with xb/xb_next swapped. The embeds
    // below write to xb[b], and a captured graph reads whatever xb[b] was AT CAPTURE, so
    // without this reset the second chunk embeds into the buffer the graph does not read.
    // Cheap, unconditional, and it makes the eager and captured paths start identically.
    for (int r = 0; r < tp_size; ++r) {
        KimiK3TPRank& R = p.ranks[(size_t)r];
        if (!R.xb_canon.empty()) { R.xb = R.xb_canon; R.xb_next = R.xb_next_canon; }
    }

    // Embed all B tokens up front — they are the only per-token input and they depend
    // on nothing but the id.
    for (int b = 0; b < B; ++b) {
        const int id = ids[b];
        if (!issue_all([&](int r) {
                KimiK3TPRank& R = p.ranks[(size_t)r];
                return embed_token(R.weights, cfg, id, R.xb[(size_t)b], R.stream);
            })) return false;
    }

    // ---- CAPTURE THE CHUNK ------------------------------------------------
    //
    // WHY THIS IS NOT OPTIONAL. Uncaptured, the chunk lost to the very loop it replaces:
    // at ctx 1024 the forward_token loop ran 18.81 ms/token and the chunk 26.53. The
    // amortisation is real (185 rendezvous per chunk, not per token) and was simply
    // smaller than what leaving capture off costs — ~3,449 eager launches per token per
    // rank, ~6.9 ms predicted against 7.72 ms measured. Capture is what makes the
    // amortisation collectable.
    //
    // WHAT MAKES A CHUNK CAPTURABLE. Exactly what makes a token capturable: everything
    // that differs between chunks is either in DEVICE memory or issued outside the
    // recorded region. The positions live in d_pos_chunk (written above, before capture
    // begins); the B embeds are issued above; the head is issued below. What remains is
    // 93 layers of phases whose pointers the host aims identically every chunk — the same
    // xb/xb_next pair, the same res_bank_b, the same peer slices at the same offsets — so
    // the addresses the graph bakes stay valid for every later chunk.
    //
    // The graph's identity is (B, mla_plan). The tail chunk of a prompt has a smaller B
    // and runs eager; a context that crosses a kMlaSplitMinCtx boundary invalidates, the
    // same rule the decode step applies to itself.
    // Pin to the whole prompt's final context when the caller set it, so the graph is
    // captured once instead of once per 4096-token step.
    const int plan_ctx = p.prefill_plan_ctx > 0 ? p.prefill_plan_ctx : (base + B);
    const int live_plan = k3k::k3_mla_decode_plan(cfg.n_q_heads, cfg.kv_lora_rank,
                                                  plan_ctx);
    static const bool want_graph = [] {
        const char* e = std::getenv("SPARKINFER_K3_PREFILL_GRAPH");
        return !(e && e[0] == '0');
    }();
    const bool graph_on = want_graph && !p.chunk_graph_off && B == p.prefill_chunk;
    if (graph_on && p.chunk_graph_ready &&
        (p.chunk_graph_plan != live_plan || p.chunk_graph_b != B)) {
        for (auto& R : p.ranks) {
            cudaSetDevice(R.device);
            if (R.chunk_exec)  { cudaGraphExecDestroy(R.chunk_exec); R.chunk_exec  = nullptr; }
            if (R.chunk_graph) { cudaGraphDestroy(R.chunk_graph);    R.chunk_graph = nullptr; }
        }
        p.chunk_graph_ready = false;
    }
    const bool replaying = graph_on && p.chunk_graph_ready;
    const bool capturing = graph_on && !replaying;

    if (capturing) {
        for (auto& R : p.ranks) {
            if (cudaSetDevice(R.device) != cudaSuccess) return false;
            // Relaxed for the reason the decode capture gives: the compute is enqueued
            // from pool workers while the collective is enqueued from this thread, so one
            // capturing stream legitimately receives work from two threads.
            if (cudaStreamBeginCapture(R.stream, cudaStreamCaptureModeRelaxed)
                != cudaSuccess) {
                for (auto& Q : p.ranks) {
                    cudaGraph_t g = nullptr;
                    cudaSetDevice(Q.device);
                    cudaStreamEndCapture(Q.stream, &g);
                    if (g) cudaGraphDestroy(g);
                }
                cudaGetLastError();
                p.chunk_graph_off = true;
                std::fprintf(stderr, "[k3-prefill] BeginCapture failed — chunks stay eager\n");
                return false;   // caller retries; chunk_graph_off makes the retry eager
            }
        }
    }

    int coll_k = 0;
    const long coll0 = p.n_collectives;
    if (!replaying)
    for (int layer = 0; layer < cfg.n_layers; ++layer) {
        const bool is_moe = layer >= cfg.leading_dense;
        // The same three predicates forward_token uses, tp_size guard included. The
        // chunk path only runs at tp_size > 1, but a copy of this arithmetic that
        // dropped a term is exactly how the two drivers would come to disagree about
        // which layers reduce — and that disagreement is silent.
        const bool kda_reduce = tp_size > 1 && cfg.is_kda_layer(layer) &&
            KimiK3Weights::shards_kda(p.ranks[0].weights.policy);
        const bool mla_reduce = tp_size > 1 && !cfg.is_kda_layer(layer) &&
            KimiK3Weights::shards_mla(p.ranks[0].weights.policy);
        const bool attn_reduce = kda_reduce || mla_reduce;

        const int k_attn = coll_k;
        const int k_moe  = coll_k + (attn_reduce ? 1 : 0);
        const int slot_attn = n_slots > 1 ? (k_attn % n_slots) : -1;
        const int slot_moe  = n_slots > 1 ? (k_moe  % n_slots) : -1;
        float* const* in_attn = slot_attn >= 0
            ? p.zc_in_slot[(size_t)slot_attn].data() : p.zc_in.data();
        float* const* in_moe = slot_moe >= 0
            ? p.zc_in_slot[(size_t)slot_moe].data() : p.zc_in.data();

        // Widths this layer reduces. Taken from the forward's own accessor so the
        // stride between tokens in the peer buffer cannot drift from what the kernels
        // write — a stride that disagreed would reduce token b's partial into token
        // b+1's slot and stay fluent.
        int attn_count = 0, moe_count = 0;
        if (attn_reduce) {
            kimi_k3_partial_buffer(p.ranks[0].fwd, layer, K3LayerPhase::Attn, &attn_count);
            if (attn_count <= 0) return false;
        }
        if (is_moe) {
            kimi_k3_partial_buffer(p.ranks[0].fwd, layer, K3LayerPhase::FfnPartial,
                                   &moe_count);
            if (moe_count <= 0) return false;
        }

        // PHASE 3 CANNOT ALWAYS BE DEFERRED, AND THE LEADING DENSE LAYER IS WHY.
        //
        // Interleaving B tokens is only sound where the value crossing a phase boundary
        // is per-token. Across the MoE reduce it is: FfnFinish reads s.moe_fused (aimed
        // at this token's slice of the peer buffer) and hidden_out (this token's x_next),
        // and it REWRITES s.ffn_out before reading it. On the leading dense layer there
        // is no reduce and FfnFinish reads the s.ffn_out that FfnPartial wrote — shared
        // scratch, so deferring it would hand every token layer 0's LAST token's FFN.
        // Fluent, wrong, and invisible to any timing run.
        //
        // So phase 3 is deferred only when a collective actually separates it. Layer 0 is
        // one layer of 93 and reduces nothing, so folding it back in costs no rendezvous.
        const bool defer_finish = is_moe;

        // --- phase 1 (+2, +3) token by token ---------------------------------
        for (int b = 0; b < B; ++b) {
            for (int r = 0; r < tp_size; ++r) {
                KimiK3TPRank& R = p.ranks[(size_t)r];
                aim(R, b);
                if (attn_reduce)
                    kimi_k3_swap_partial_buffer(R.fwd, K3LayerPhase::Attn,
                                                in_attn[r] + (size_t)b * attn_count);
                if (is_moe && !attn_reduce)
                    kimi_k3_swap_partial_buffer(R.fwd, K3LayerPhase::FfnPartial,
                                                in_moe[r] + (size_t)b * moe_count);
            }
            if (!issue_all([&](int r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    if (!kimi_k3_forward_layer_phase(R.fwd, layer, K3LayerPhase::Attn,
                                                     R.x, R.x_next)) return false;
                    if (attn_reduce) return true;
                    if (!kimi_k3_forward_layer_phase(R.fwd, layer,
                                                     K3LayerPhase::FfnPartial,
                                                     R.x, R.x_next)) return false;
                    if (defer_finish) return true;
                    return kimi_k3_forward_layer_phase(R.fwd, layer,
                                                       K3LayerPhase::FfnFinish,
                                                       R.x, R.x_next);
                })) return false;
            // The phase may have pushed a checkpoint; park the advanced count. Rank 0
            // is representative because the push is a host-side `layer % res_bs`.
            nck[(size_t)b] = p.ranks[0].state.n_ckpt;
            if (!attn_reduce && !defer_finish)
                for (int r = 0; r < tp_size; ++r)
                    std::swap(p.ranks[(size_t)r].xb[(size_t)b],
                              p.ranks[(size_t)r].xb_next[(size_t)b]);
        }

        // --- the attention collective: ONE call over all B partials ----------
        if (attn_reduce) {
            const size_t count = (size_t)attn_count * (size_t)B;
            if (!p.coll->allreduce_f32_owned_slot(count, p.streams, slot_attn)) {
                std::fprintf(stderr, "[k3-tp] chunk attention all-reduce failed at "
                                     "layer %d\n", layer);
                return false;
            }
            ++p.n_collectives;
            ++coll_k;

            for (int b = 0; b < B; ++b) {
                for (int r = 0; r < tp_size; ++r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    aim(R, b);
                    // FfnPartial reads the reduced attention where the collective put
                    // it, and writes its own partial into this token's MoE slot.
                    kimi_k3_swap_partial_buffer(R.fwd, K3LayerPhase::Attn,
                                                p.zc_out[r] + (size_t)b * attn_count);
                    if (is_moe)
                        kimi_k3_swap_partial_buffer(R.fwd, K3LayerPhase::FfnPartial,
                                                    in_moe[r] + (size_t)b * moe_count);
                }
                if (!issue_all([&](int r) {
                        KimiK3TPRank& R = p.ranks[(size_t)r];
                        if (!kimi_k3_forward_layer_phase(R.fwd, layer,
                                                         K3LayerPhase::FfnPartial,
                                                         R.x, R.x_next)) return false;
                        if (defer_finish) return true;
                        return kimi_k3_forward_layer_phase(R.fwd, layer,
                                                           K3LayerPhase::FfnFinish,
                                                           R.x, R.x_next);
                    })) return false;
                if (!defer_finish)
                    for (int r = 0; r < tp_size; ++r)
                        std::swap(p.ranks[(size_t)r].xb[(size_t)b],
                                  p.ranks[(size_t)r].xb_next[(size_t)b]);
            }
        }

        // --- the FFN collective: ONE call over all B partials ----------------
        if (is_moe) {
            const size_t count = (size_t)moe_count * (size_t)B;
            if (!p.coll->allreduce_f32_owned_slot(count, p.streams, slot_moe)) {
                std::fprintf(stderr, "[k3-tp] chunk all-reduce failed at layer %d\n",
                             layer);
                return false;
            }
            ++p.n_collectives;
            ++coll_k;
        }

        // --- phase 3, only where the MoE reduce deferred it ---------------------
        if (!defer_finish) continue;
        for (int b = 0; b < B; ++b) {
            for (int r = 0; r < tp_size; ++r) {
                KimiK3TPRank& R = p.ranks[(size_t)r];
                aim(R, b);
                kimi_k3_swap_partial_buffer(R.fwd, K3LayerPhase::FfnPartial,
                                            p.zc_out[r] + (size_t)b * moe_count);
            }
            if (!issue_all([&](int r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    return kimi_k3_forward_layer_phase(R.fwd, layer,
                                                       K3LayerPhase::FfnFinish,
                                                       R.x, R.x_next);
                })) return false;
            // The swap that phase 3 owns, applied to THIS token's pair. Storing it back
            // is what carries the layer's output into the next layer's input.
            for (int r = 0; r < tp_size; ++r) {
                KimiK3TPRank& R = p.ranks[(size_t)r];
                std::swap(R.xb[(size_t)b], R.xb_next[(size_t)b]);
            }
        }
    }

    // ---- close the capture, then run it ------------------------------------
    if (capturing) {
        // PRICE EVERY CAPTURE, because a per-band measurement cannot tell a slow band from
        // a re-captured one. A 32k ingestion showed 33.4 ms/token across 2048->4096 against
        // 18.9 either side — ~29 s of excess, seven times what one capture costs — and
        // there is no way to attribute that from the outside. This says how many captures
        // happened, when, and how the cost splits between recording the graph and
        // instantiating it: if instantiate dominates, cudaGraphExecUpdate is the fix; if
        // recording does, the graph has to get smaller.
        const auto t_cap0 = std::chrono::steady_clock::now();
        double ms_record = 0.0;
        bool ok_cap = true;
        for (auto& R : p.ranks) {
            if (cudaSetDevice(R.device) != cudaSuccess) { ok_cap = false; break; }
            if (cudaStreamEndCapture(R.stream, &R.chunk_graph) != cudaSuccess ||
                R.chunk_graph == nullptr) { ok_cap = false; break; }
            ms_record = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_cap0).count();
            if (cudaGraphInstantiate(&R.chunk_exec, R.chunk_graph, nullptr, nullptr, 0)
                != cudaSuccess) { ok_cap = false; break; }
        }
        if (ok_cap) {
            size_t n_nodes = 0;
            cudaGraphGetNodes(p.ranks[0].chunk_graph, nullptr, &n_nodes);
            const double ms_all = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t_cap0).count();
            ++p.n_chunk_captures;
            p.ms_chunk_capture += ms_all;
            std::fprintf(stderr,
                         "[k3-prefill] chunk graph capture #%ld at base=%d plan=%d B=%d: "
                         "%.0f ms total (record %.0f, instantiate %.0f), %zu nodes/rank\n",
                         p.n_chunk_captures, base, live_plan, B,
                         ms_all, ms_record, ms_all - ms_record, n_nodes);
        }
        if (!ok_cap) {
            // A failed capture costs speed, never correctness — the same rule the decode
            // step applies. Stand the chunk graph down for the rest of the process and
            // let the caller re-issue this chunk eagerly.
            std::fprintf(stderr, "[k3-prefill] chunk capture failed — chunks stay eager\n");
            for (auto& R : p.ranks) {
                cudaSetDevice(R.device);
                if (R.chunk_exec)  { cudaGraphExecDestroy(R.chunk_exec); R.chunk_exec  = nullptr; }
                if (R.chunk_graph) { cudaGraphDestroy(R.chunk_graph);    R.chunk_graph = nullptr; }
            }
            cudaGetLastError();
            p.chunk_graph_off   = true;
            p.chunk_graph_ready = false;
            return false;
        }
        p.chunk_graph_ready = true;
        p.chunk_graph_plan  = live_plan;
        p.chunk_graph_b     = B;
        // The checkpoint count a chunk ENDS on. The graph carries kernels, not host
        // bookkeeping, so a replay has to be told where the capture left this.
        p.chunk_ckpt_end    = nck[0];
        p.n_coll_per_chunk  = p.n_collectives - coll0;
        size_t nnodes = 0;
        cudaGraphGetNodes(p.ranks[0].chunk_graph, nullptr, &nnodes);
        std::fprintf(stderr, "[k3-prefill] captured chunk: %zu nodes/rank, %d tokens/pass, "
                             "mla splits=%d, from position %d\n",
                     nnodes, B, live_plan, base);
    }

    if (replaying) {
        // HOST BOOKKEEPING THE GRAPH DOES NOT CARRY. The recorded kernels already hold the
        // baked pointers, but the host's own view has to end where the capture's did or
        // the head below reads the wrong buffer of the pair.
        //
        // Each layer's phase 3 swaps a token's pair exactly once, so after n_layers the
        // parity is all that survives; and every chunk starts from nck = 0 and advances by
        // the same host-side `layer % res_bs`, so the end count is a constant.
        if (cfg.n_layers & 1)
            for (int b = 0; b < B; ++b)
                for (int r = 0; r < tp_size; ++r) {
                    KimiK3TPRank& R = p.ranks[(size_t)r];
                    std::swap(R.xb[(size_t)b], R.xb_next[(size_t)b]);
                }
        for (int b = 0; b < B; ++b) nck[(size_t)b] = p.chunk_ckpt_end;
        p.n_collectives += (long)p.n_coll_per_chunk;
    }

    if (graph_on) {
        // Dispatch through the pool so the eight launches are concurrent rather than in
        // rank order: a host loop makes rank 7 start after seven launches have gone ahead
        // of it, and the collective then charges every rank for that skew.
        if (!issue_all([&](int r) {
                return cudaGraphLaunch(p.ranks[(size_t)r].chunk_exec,
                                       p.ranks[(size_t)r].stream) == cudaSuccess;
            })) {
            std::fprintf(stderr, "[k3-prefill] chunk graph launch failed\n");
            return false;
        }
    }

    // --- head: the LAST token of the chunk only -----------------------------
    //
    // Ingestion discards the other B-1 distributions, and the head is the largest
    // projection in the model (1.25 GB of Q8_0) followed by a full 8-rank drain. Paying
    // it per prompt token was the single most wasteful thing about ingestion; chunks
    // are cut so that every depth anyone asked for lands on a chunk boundary.
    if (want_logits) {
        for (int r = 0; r < tp_size; ++r) aim(p.ranks[(size_t)r], B - 1);
        const bool band_head = k3_tp_head_banded(p);
        if (!k3_tp_issue_head(p, band_head)) return false;
        if (!k3_tp_copy_logits(p, band_head, out_logits, issue_all)) return false;
    }

    // Restore the partial buffers to the scratch the forward owns, so a later
    // forward_token (or the teardown in kimi_k3_tp_free) sees the pointers it expects
    // rather than peer memory this chunk aimed them at.
    for (int r = 0; r < tp_size; ++r) {
        KimiK3TPRank& R = p.ranks[(size_t)r];
        if (!p.orig_attn.empty())
            kimi_k3_swap_partial_buffer(R.fwd, K3LayerPhase::Attn, p.orig_attn[(size_t)r]);
        if (!p.orig_moe.empty())
            kimi_k3_swap_partial_buffer(R.fwd, K3LayerPhase::FfnPartial,
                                        p.orig_moe[(size_t)r]);
    }
    return true;
}

}  // namespace

bool kimi_k3_tp_forward_prompt(KimiK3TP& p, const int* ids, int n_ids,
                               const int* dump_at, int n_dump,
                               float* out_logits,
                               const std::function<bool(int)>& on_depth) {
    if (p.ranks.empty() || !p.coll || n_ids <= 0 || !ids || !out_logits) return false;

    // The depths at which a distribution is wanted, ascending and unique, always
    // including the end of the prompt. These are also the CHUNK BOUNDARIES: a chunk
    // computes logits for its last token, so a requested depth that fell mid-chunk
    // would have to be recomputed.
    std::vector<int> stops;
    stops.reserve((size_t)n_dump + 1);
    for (int i = 0; i < n_dump; ++i)
        if (dump_at[i] > 0 && dump_at[i] <= n_ids) stops.push_back(dump_at[i]);
    stops.push_back(n_ids);
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());

    // FALLBACK: main's ingestion, one forward_token per token. Reached when the chunk
    // machinery stood down at init, and it is not a degraded mode — it is exactly what
    // this repo does today, so a box where the chunk path cannot run still ingests
    // correctly at the speed it always had.
    if (p.prefill_chunk <= 1) {
        size_t next = 0;
        for (int i = 0; i < n_ids; ++i) {
            if (!kimi_k3_tp_forward_token(p, ids[i], out_logits)) return false;
            if (next < stops.size() && stops[next] == i + 1) {
                if (on_depth && !on_depth(stops[next])) return false;
                ++next;
            }
        }
        return true;
    }

    const int tp_size = (int)p.ranks.size();
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
    std::function<bool(const std::function<bool(int)>&)> issue_all =
        [&](const std::function<bool(int)>& job) -> bool {
            if (parallel_issue) return p.issue.run(job);
            for (int r = 0; r < tp_size; ++r) {
                if (cudaSetDevice(p.ranks[(size_t)r].device) != cudaSuccess) return false;
                if (!job(r)) return false;
            }
            return true;
        };

    // GRAPH CAPTURE IS OFF FOR THE CHUNK PATH, and that is a decision, not an omission.
    // The captured decode graph is one token's 3,449 nodes; a chunk's is B times that,
    // recorded and instantiated for a shape that is used once per prompt. The chunk's
    // own saving (185 rendezvous instead of 185B) does not need it, and the host issue
    // it would hide is ~2 us x 3,449 per rank per token against a token that still
    // costs milliseconds. If a measurement later says otherwise, the structure here is
    // already capture-shaped: the per-token pointer aiming repeats identically every
    // chunk, and the positions live in device memory.
    const int base0 = p.ranks[0].state.position;

    // Put every rank back on its single-token buffers. RUNS ON EVERY EXIT, including the
    // failure ones: a chunk that returned early would otherwise leave state.res_bank and
    // R.x aimed at chunk allocations, and kimi_k3_tp_free would then cudaFree the same
    // pointer twice. Restoring only on success is the version of this that works in
    // testing and double-frees in the field.
    auto restore = [&](int pos) -> bool {
        bool ok = true;
        for (auto& R : p.ranks) {
            if (cudaSetDevice(R.device) != cudaSuccess) { ok = false; continue; }
            R.state.d_pos    = R.d_pos_single;
            R.state.res_bank = R.res_bank_single;
            R.x      = R.x_canon;
            R.x_next = R.x_next_canon;
            R.state.n_ckpt = 0;
            // Writes the host mirror AND the device copy, which is what the next
            // forward_token's KV store indexes with. cudaSetDevice above is not
            // optional: kimi_k3_set_position memcpys to this rank's d_pos.
            if (!kimi_k3_set_position(R.state, pos)) ok = false;
        }
        return ok;
    };

    // Pin the chunk graph's MLA split plan to where this prompt ENDS, so the graph is
    // captured once for the whole ingestion instead of being thrown away at every 4096
    // boundary. Restored below so a later decode is unaffected.
    const int plan_ctx_saved = p.prefill_plan_ctx;
    p.prefill_plan_ctx = base0 + n_ids;

    int done = 0;
    for (size_t si = 0; si < stops.size(); ++si) {
        const int target = stops[si];
        while (done < target) {
            int B = p.prefill_chunk;
            if (B > target - done) B = target - done;
            const bool last = (done + B == target);
            if (!k3_prefill_chunk(p, ids + done, base0 + done, B, last, out_logits,
                                  issue_all)) {
                // Land the position on what was actually ingested, not on the prompt
                // length: a caller that reports the failure and stops still has to be
                // able to free, and one that retries must not attend over a gap.
                restore(base0 + done);
                return false;
            }
            done += B;
        }
        if (on_depth && !on_depth(target)) { restore(base0 + done); return false; }
    }

    // What the ingestion spent BUILDING graphs, as a share of the tokens it ingested.
    // Printed next to PREFILL_TOTAL so the two are read together: a tok/s number averages
    // capture cost over the whole prompt and hides it, and this is the term that decides
    // whether the chunk's steady-state parity is worth anything end to end.
    p.prefill_plan_ctx = plan_ctx_saved;

    if (p.n_chunk_captures > 0)
        std::fprintf(stderr,
                     "[k3-prefill] %ld chunk graph captures over %d tokens: %.0f ms total, "
                     "%.3f ms/token of the ingestion\n",
                     p.n_chunk_captures, n_ids, p.ms_chunk_capture,
                     n_ids > 0 ? p.ms_chunk_capture / (double)n_ids : 0.0);

    // Leave every rank where a forward_token loop would have left it: position past the
    // prompt, host and device mirrors agreeing. A caller that ingests a prompt and then
    // generates must see no seam.
    return restore(base0 + n_ids);
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
    for (auto& R : p.ranks) {
        cudaSetDevice(R.device);
        // The chunk path re-aims these at its own buffers; put them back before the
        // state frees what it owns, or kimi_k3_free_state releases a chunk allocation
        // and leaks the one it allocated.
        if (R.d_pos_single)    R.state.d_pos    = R.d_pos_single;
        if (R.res_bank_single) R.state.res_bank = R.res_bank_single;
        for (auto* q : R.xb)         if (q) cudaFree(q);
        for (auto* q : R.xb_next)    if (q) cudaFree(q);
        for (auto* q : R.res_bank_b) if (q) cudaFree(q);
        if (R.d_pos_chunk) cudaFree(R.d_pos_chunk);
        R.xb.clear(); R.xb_next.clear(); R.res_bank_b.clear();
        R.d_pos_chunk = nullptr;
        // x/x_next may be swapped relative to how they were allocated; free the
        // canonical pair, which is the pair cudaMalloc returned.
        if (R.x_canon) { R.x = R.x_canon; R.x_next = R.x_next_canon; }
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
