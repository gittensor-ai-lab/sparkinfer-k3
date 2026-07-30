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
// called with need_f32=true, which downgrades a bf16-only fast backend to NCCL up
// front rather than failing after a twenty-minute weight load.

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/tp/collective.h"

#include <memory>
#include <vector>

#include <cuda_runtime.h>

namespace sparkinfer {

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
};

struct KimiK3TP {
    KimiK3Config cfg;
    K3PlanOptions opt;
    std::vector<KimiK3TPRank> ranks;
    std::unique_ptr<tp::Collective> coll;
    std::vector<cudaStream_t> streams;   // cached in rank order for the group call
    std::vector<void*> reduce_bufs;      // scratch, refilled per collective
    long n_collectives = 0;              // counted, so a run can assert it saw 92/token
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

void kimi_k3_tp_free(KimiK3TP& p);

}  // namespace sparkinfer
