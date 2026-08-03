#pragma once
// How many all-reduces a TP decode issues, derived from the shard policy.
//
// This exists because the number is an INVARIANT that two places have to agree on, and
// they drifted. kimi_k3_tp_forward_token issues the reduces; kimi_k3_tp_check asserts how
// many it saw. The check carried its own hand-written formula — "one per MoE layer" — which
// was right under ShardPolicy::ExpertsOnly and silently wrong the moment attention sharding
// became the default: every layer then reduces its attention band as well, so an 8-layer
// tp=2 run issues 45 collectives where the formula still said 21, and the check failed on
// unmodified main with logits that matched BITWISE. See issue #75.
//
// Kept CUDA-free (KimiK3Config only, no cuda_runtime.h) so the arithmetic can be unit-tested
// on an ordinary CI runner — the same reasoning as tp_shard_cpu_test. A wrong expectation
// here does not crash: it turns a correctness gate into either a permanent false alarm, which
// is what #75 is, or a blind spot that lets a genuinely missing reduce through.
//
// THE PREDICATES ARE COPIED FROM THE FORWARD ON PURPOSE, not re-derived. In
// kimi_k3_tp.cpp's layer loop:
//
//     kda_reduce  = tp_size > 1 && cfg.is_kda_layer(layer) && shards_kda(policy)
//     mla_reduce  = tp_size > 1 && !cfg.is_kda_layer(layer) && shards_mla(policy)
//     attn_reduce = kda_reduce || mla_reduce                      -> ++n_collectives
//     is_moe && tp_size > 1  (is_moe = layer >= cfg.leading_dense) -> ++n_collectives
//
// Anything that changes those two sites has to change this file, and the CPU test is what
// makes that a failing build rather than a mystery on the node.

#include "sparkinfer/models/kimi_k3_config.h"

namespace sparkinfer {

// Split out rather than summed, because the two answer different questions when the count
// is wrong: a bad `attn` means the shard policy is not what the caller thought it was, a
// bad `moe` means a dispatch reduce went missing.
struct KimiK3CollectiveCount {
    long attn = 0;   // one per layer whose attention band is column-sharded
    long moe  = 0;   // one per MoE layer (the leading dense layer's FFN is replicated)

    long per_token() const { return attn + moe; }
};

// Collectives per token for `cfg` under a policy described by its two band flags.
//
// Takes bools rather than KimiK3Weights::ShardPolicy so this header does not have to pull
// in kimi_k3.h and, with it, cuda_runtime.h. Callers holding a policy pass
// KimiK3Weights::shards_kda(p) / shards_mla(p), which is the enum's own accessor pair.
inline KimiK3CollectiveCount kimi_k3_expected_collectives(const KimiK3Config& cfg,
                                                          int tp_size,
                                                          bool shards_kda,
                                                          bool shards_mla) {
    KimiK3CollectiveCount c;
    // Every reduce site in the forward is guarded by tp_size > 1: a single rank owns the
    // whole tensor, so there is nothing to sum. This is also what the tp=1 reference run
    // asserts, and it is not vacuous — a reduce that fires at tp=1 would double the value.
    if (tp_size <= 1) return c;

    for (int layer = 0; layer < cfg.n_layers; ++layer) {
        const bool kda = cfg.is_kda_layer(layer);
        if ((kda && shards_kda) || (!kda && shards_mla)) ++c.attn;
        if (layer >= cfg.leading_dense) ++c.moe;
    }
    return c;
}

}  // namespace sparkinfer
