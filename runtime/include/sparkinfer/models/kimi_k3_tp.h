#pragma once
// Kimi K3 decode across N GPUs with TENSOR parallelism.
//
// Distinct from the layer-split pipeline in kimi_k3.h, and not a replacement for it
// in every case — they trade different things:
//
//   pipeline   Each GPU owns a contiguous LAYER RANGE. No collectives, one small
//              handoff per stage boundary. But only one GPU is busy at a time, so
//              token latency is the SUM of the stages: it buys capacity, not speed.
//
//   this       Every GPU participates in every layer. The routed experts are banded
//              across ranks, so the 70%-of-runtime MoE dispatch is divided by
//              tp_size, at the cost of one all-reduce per MoE layer.
//
// SCOPE: ShardPolicy::ExpertsOnly. The 896 routed experts (531 of UD-IQ1_S's 553 GiB)
// are banded; everything else is replicated. Consequences, all deliberate:
//
//   - The MoE dispatch is the ONLY sharded op, so there is exactly ONE collective per
//     MoE layer — 92 per token, not the 186 a fully-sharded K3 would need. It lands
//     at expert_latent (3584), before ffn_routed_norm, because rms_norm is not linear
//     and cannot be applied to a partial sum. See kimi_k3_decode_plan.cpp.
//   - Attention runs REDUNDANTLY on every rank. That is 8x wasted attention FLOPs and
//     it is the honest cost of not yet threading per-rank head counts through the KDA
//     and MLA kernels. Given the measured split (ffn_moe 69.7%, attention 26.8%), the
//     ceiling here is ~2.6x rather than ~8x — real, but not the end state.
//   - Every rank therefore holds an IDENTICAL hidden state at every layer boundary,
//     which is what lets rank 0's logits be the answer with no gather.
//
// The collective is f32, not bf16: K3's residual stream is f32 by design and reducing
// it in bf16 would truncate to ~8 mantissa bits 92 times per token. make_collective is
// called with need_f32=true; peer-oneshot and multimem both implement the f32 path, and
// kimi_k3_tp_load refuses any backend that reports !supports_f32() rather than failing
// after a twenty-minute weight load.

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/tp/collective.h"

#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

namespace sparkinfer {

// ---------------------------------------------------------------------------
// K3IssuePool — one host thread per rank, so submission is not serial
// ---------------------------------------------------------------------------
// MEASURED, at the scored 128k context, before this existed:
//
//     total 109.7 ms/token | host issue 88.4 (80.6%) | collective 5.9 | sync 14.9
//     2232 phase calls/token, 1496 cudaSetDevice/token
//
// 80% of a decode token was ONE host thread asking eight GPUs to do things. The
// GPU is starved, not saturated: when submission finally stops there are only
// ~15 ms of work left to drain. Two consequences follow, and the second is the
// one that matters.
//
//   1. Kernel-level optimisation is nearly invisible here. Making a kernel
//      faster shortens something that was already hidden behind submission.
//
//   2. The rank skew at the collective is MANUFACTURED BY ISSUE ORDER. Rank 0's
//      layer is submitted ~200 launches before rank 7's, so rank 7 arrives late
//      at all 92 barriers, every token, by construction. That is not expert
//      imbalance and no collective backend can fix it — the ranks genuinely
//      were not asked at the same time.
//
// The eight ranks are independent between collectives, so they can be submitted
// concurrently. Each worker owns exactly one device and calls cudaSetDevice ONCE
// at thread start rather than 1496 times per token: the current device is
// thread-local state, which is precisely what makes this both cheap and safe.
//
// THREAD SAFETY, and why this is sound rather than lucky. Every mutable static
// on the forward path (g_mla_part_*, the IQ1_S/IQ2_XS lattice `ready` flags, the
// MLA split cache) is indexed by cudaGetDevice(). Distinct workers own distinct
// devices, so no two threads ever address the same slot. The per-device state
// convention the kernels already follow is what makes per-rank threading safe;
// a shared cache would have to be locked instead.
//
// Spin, not condvar: 186 releases per token against ~240 idle cores. A condvar
// round trip is 10-20 us and would cost ~3 ms/token, eating a third of what
// parallel submission buys. The spin yields after a bounded number of pauses so
// an oversubscribed box degrades instead of livelocking.
class K3IssuePool {
public:
    K3IssuePool() = default;
    ~K3IssuePool() { shutdown(); }

    K3IssuePool(const K3IssuePool&) = delete;
    K3IssuePool& operator=(const K3IssuePool&) = delete;

    bool started() const { return !workers_.empty(); }
    int size() const { return (int)workers_.size(); }

    // Spawn one thread per device. Each pins itself to devices[r] and parks.
    void start(const std::vector<int>& devices);

    // Run job(rank) on every rank concurrently; block until all finish.
    // Returns false if ANY rank returned false. Never runs jobs partially: a
    // failing rank still completes the barrier, so the pool cannot deadlock on
    // an error path.
    bool run(const std::function<bool(int)>& job);

    void shutdown();

private:
    std::vector<std::thread> workers_;
    std::vector<int> devices_;
    const std::function<bool(int)>* job_ = nullptr;
    std::atomic<unsigned> epoch_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> failed_{false};
    std::atomic<bool> stop_{false};
};

struct KimiK3TPRank {
    int device = 0;
    int rank = 0;
    KimiK3Weights weights;
    KimiK3RuntimeState state;
    KimiK3Forward fwd;
    cudaStream_t stream = nullptr;   // this rank's stream; the collective is enqueued
                                     // on it, so compute/collective ordering is
                                     // stream-ordered and needs no host barrier
    float* x = nullptr;              // [hidden]
    float* x_next = nullptr;         // [hidden]
    float* logits = nullptr;         // [vocab], rank 0 only

    // ---- chunked prompt ingestion (kimi_k3_tp_forward_prompt) --------------
    // The hidden pair, B tokens wide. Separate from x/x_next rather than a
    // resize of them: the decode path's captured graph bakes x/x_next, so
    // growing those would invalidate every recorded graph on the box.
    float* xb = nullptr;             // [b_cap][hidden]
    float* xnb = nullptr;            // [b_cap][hidden]
    // The chunk's base position, ON THE DEVICE. The per-layer reset of d_pos has
    // to be stream-ordered — a host memcpy per layer per rank would be 93 * 8
    // synchronising copies per chunk on the critical path — so the base lives in
    // device memory and the reset is a 4-byte D2D copy.
    int*   d_pos_base = nullptr;
    int    b_cap = 0;

    // THE CANONICAL ASSIGNMENT OF THE x/x_next PAIR, AND WHY IT HAS TO EXIST.
    //
    // Phase 3 swaps x and x_next once per layer. 93 layers is an ODD number, and rank 0
    // swaps once more in the attention-residual head, so the pair comes out of a token
    // exchanged relative to how it went in — the assignment ALTERNATES between tokens.
    //
    // Eagerly that is harmless: every launch reads the live pointer. A captured graph
    // bakes addresses, so a graph recorded on one parity would, on the next token, read
    // the buffer it should be writing and write the one it should be reading. Fluent,
    // wrong, and invisible to any timing benchmark.
    //
    // Resetting to these at the start of every token makes each token begin identically.
    // The contents need no preservation: the hidden state does not carry across tokens
    // (the embed overwrites x; only the KV cache and the KDA recurrent state persist).
    float* x_canon = nullptr;
    float* x_next_canon = nullptr;

    // Per-rank capture. One graph per rank, because each rank owns its own stream and
    // its own device; they rendezvous inside the collective at replay time exactly as
    // they do eagerly.
    cudaGraph_t     graph = nullptr;
    cudaGraphExec_t exec  = nullptr;
};

struct KimiK3TP {
    KimiK3Config cfg;
    K3PlanOptions opt;
    std::vector<KimiK3TPRank> ranks;
    std::unique_ptr<tp::Collective> coll;
    std::vector<cudaStream_t> streams;   // cached in rank order for the group call
    std::vector<void*> reduce_bufs;      // scratch, refilled per collective
    long n_collectives = 0;              // counted, so a run can assert it saw 92/token

    // Graph state. `captured_plan` is the MLA `splits` the graph was recorded against;
    // when the live plan differs the graph is thrown away and re-recorded, because
    // `splits` sizes the grid and picks the kernel and a graph can change neither.
    // Both come from k3_mla_decode_plan() so the check cannot drift from the launcher.
    bool graph_disabled = false;  // set once capture has failed; never retried
    bool graph_ready   = false;
    int  captured_plan = 0;
    long n_replays     = 0;
    long n_captures    = 0;

    // One submission thread per rank. Empty until the first decode step, and
    // bypassed entirely at tp_size 1 — a single rank has nothing to parallelise
    // and the serial path stays byte-identical.
    K3IssuePool issue;

    // Mode-B zero-copy: with an owned-buffer f32 backend, the expert partial is
    // written into the collective's reduce_in() and consumed from reduce_out()
    // directly (kimi_k3_swap_partial_buffer), skipping both staging copies of
    // every collective. Off for NCCL (Mode A reduces in place already) and under
    // SPARKINFER_K3_TP_HOST_REDUCE=1 (the debug path sums the swapped-in
    // pointers and would leave reduce_out() unwritten).
    //
    // The swaps are done by the SUBMITTING thread while the issue pool is parked
    // between phases, never by a worker mid-phase: they retarget scratch pointers
    // the workers read, so doing them anywhere else would be a data race.
    bool zero_copy = false;
    std::vector<float*> zc_in, zc_out;   // reduce_in/out, rank order
    // zc_in one row per rotating input slot (sparkinfer/tp/k3_coll_1bar.h). Row 0
    // IS zc_in, so a backend offering one slot leaves every pointer where main
    // put it. n_coll_slots is 1 whenever the rotation is off or unproven.
    std::vector<std::vector<float*>> zc_in_slot;
    int n_coll_slots = 1;
    std::vector<float*> orig_moe;        // scratch pointer, restored at free
    std::vector<float*> orig_attn;       // same, for the attention partial

    // Prefill tile scratch: the per-rank tile buffers and the per-token residual
    // banks kimi_k3_tp_prefill needs. Allocated on the first prefill call and
    // released by kimi_k3_tp_free, so a decode-only run never pays for it.
    //
    // Opaque here on purpose. Naming the type would pull the prefill kernels'
    // header into every translation unit that drives a decode step, and nothing
    // outside the prefill driver has any business reaching into it.
    struct K3PrefillPool* prefill = nullptr;
};

// Load the model once per rank, banding the routed experts. `devices` gives tp_size.
// Returns false rather than falling back to one GPU: a silent downgrade would make a
// TP benchmark measure a single card.
bool kimi_k3_tp_init(const GGUF& g, const KimiK3Config& cfg, const K3PlanOptions& opt,
                    const std::vector<int>& devices, int max_ctx, KimiK3TP& out);

// One decode step across every rank. `out_logits` must hold cfg.vocab floats.
//
// Equivalent to kimi_k3_forward_token on a single device up to float32 summation
// order — the expert partials are reassociated by the reduce, so agreement is to
// ~1 ulp, not bitwise. kimi_k3_tp_moe_check pins that bound.
bool kimi_k3_tp_forward_token(KimiK3TP& p, int token_id, float* out_logits);

// TWO PROMPT-INGESTION DRIVERS LIVE HERE, AND THE SPLIT IS THE MEASUREMENT, NOT TASTE.
//
// kimi_k3_tp_prefill is the tile driver and the one that ships: it keeps CUDA graph
// capture. kimi_k3_tp_forward_prompt is the chunked walk, which does not capture and
// batches the COLLECTIVES instead. That second axis turned out not to be worth having:
// under capture the host issues 0.08 ms of a token's 18.83 ms and the collectives cost
// 0.01 ms, so the 1.77x the chunked walk measured for batched collectives was buying
// back overhead that capture removes for free. It is kept as a control — it is the only
// path that can isolate a collective effect — and not as a candidate.
//
// Both leave the same state behind and produce the same final logits.

// Ingest a prompt, running the LAYER loop outside the TOKEN loop.
//
// WHAT THIS CHANGES, AND WHAT IT DOES NOT. Every token still computes exactly what
// kimi_k3_tp_forward_token would compute for it, in the same order, against the same
// KV cache and the same KDA recurrence — the inner loop walks tokens in sequence, which
// is the order those recurrences require. What changes is that T tokens are resident at
// one layer at a time, so a weight block can be read once and multiplied into T
// activations instead of being re-streamed from HBM for every token. At ~22 GB of active
// weights per token per rank that stream IS prefill's cost, which is why prefill tok/s
// and decode tok/s are the same number today.
//
// `out_logits` (cfg.vocab floats) receives the LAST token's logits, so the caller can
// sample from it exactly as it would after the final forward_token of a per-token loop.
//
// Falls back to the per-token path — never fails — for any prompt or geometry outside
// the tile path's contract, so a caller gets a correct result either way and the only
// observable difference is how long it took.
bool kimi_k3_tp_prefill(KimiK3TP& p, const int* ids, int n_ids, float* out_logits);

// ---------------------------------------------------------------------------
// Chunked prompt ingestion.
//
// Ingests `n` prompt tokens and leaves `out_logits` holding the distribution
// after the LAST of them — the same contract as calling kimi_k3_tp_forward_token
// n times and keeping the final result, and the same state afterwards.
//
// SPARKINFER_K3_PREFILL_BATCH picks the chunk size B (default 1). At B == 1 this
// runs the identical per-token phase sequence through a one-slot arena and is
// bit-identical to the token loop, which is what makes it the control: any
// divergence at B > 1 is the batching, not the batched code.
//
// WHAT BATCHING BUYS, AND WHAT IT DOES NOT. Measured on 8x H200 at 1k context,
// same binary and session: with CUDA graph capture ON the host issues 0.08 ms of
// a token's 18.83 ms (0.4%) and the collectives cost 0.01 ms, because the graph
// replays 2232 phase calls as 2. Prefill under capture is GPU-BOUND. So the
// motivation is NOT fewer launches or fewer rendezvous — those are already free —
// it is that every projection runs as an M=1 GEMV at ~13% of the bandwidth
// roofline, and a chunk turns them into M=B GEMMs.
//
// That also sets the floor on B. Under 8-way expert sharding a rank holds 112 of
// the 896 experts and sees ~2 of a token's 16 selections, so the rows per expert
// reaching its MoE GEMM are B/56: a chunk of 8 leaves the tensor cores as idle as
// a chunk of 1, and one full mma.m16n8k32 tile needs B ~ 896. Chunks here are
// meant to be hundreds of tokens, not a handful.
bool kimi_k3_tp_forward_prompt(KimiK3TP& p, const int* ids, int n,
                               float* out_logits);

void kimi_k3_tp_free(KimiK3TP& p);

}  // namespace sparkinfer
