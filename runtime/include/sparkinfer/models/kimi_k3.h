#pragma once
// Kimi K3 forward pass: weight loading + per-token decode, single GPU (tp_size=1).
//
// Everything upstream of this file — the kernels (kernels/include/sparkinfer/kernels/
// kimi_k3.h), the GGUF reader with partial-shard support (sparkinfer/gguf.h), the
// decode SCHEDULE (kimi_k3_decode_plan.h), and the tensor manifest — was built and
// validated independently, several of them against real weights or real ggml. This
// file is the executor: it walks the schedule that plan describes and issues the
// kernel calls, on real device buffers, against a real (possibly partial) GGUF.
//
// SCOPE, DELIBERATELY. This is the correctness-first pass, not the throughput pass:
//
//   - fp32 hidden state end to end, matching the already-validated K3 kernels
//     (which are all f32 in/out — see kimi_k3.h's own header comment on why this is
//     a real decision, not a shortcut: bf16 would reintroduce the precision loss
//     those kernels' float64 validation exists to eliminate).
//   - One CUDA stream, sequential kernel launches, no CUDA graph capture. The
//     existing Qwen3.5 path (runtime/src/models/qwen35.cpp) graph-captures and
//     multi-streams for throughput; K3 does not need that to be CORRECT, and
//     correctness against llama.cpp is the open question right now, not tok/s.
//   - tp_size=1 only. The decode plan already carries shard rules and reduce
//     points (kimi_k3_decode_plan.h) for when this executor grows a sharded path;
//     at tp_size=1 every rule degenerates to Replicate and there is nothing to
//     shard or reduce, so this is not a stub — it is the same code TP will run,
//     minus the collective calls this file does not yet issue.
//
// WEIGHT LOADING IS SIMPLER THAN QWEN3.5'S. Every K3 weight is read NATIVE at
// forward time — k3_proj_f32 reads Q8_0/F32 bytes directly, the MoE dispatch
// kernels read IQ1_S/IQ2_XS bytes directly — so there is no equivalent of
// Qwen3.5's load-time dequantize-to-bf16 pass. Loading is: for every tensor the
// manifest requires, cudaMalloc + cudaMemcpy the raw GGUF bytes, record the
// pointer and ggml type. That's the whole loader.

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/models/kimi_k3_decode_plan.h"
#include "sparkinfer/tp/shard.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace sparkinfer {

// One native GGUF tensor, uploaded as-is. `type` is the ggml type id; callers that
// need to read it (k3_proj_f32, the MoE dispatch kernels, or a plain memcpy for a
// pre-dequantized F32 vector) dispatch on `type` themselves — this struct does not
// interpret the bytes.
struct KimiK3Tensor {
    const void* data = nullptr;   // device pointer, owned by KimiK3Weights
    int         type = -1;       // ggml type id; -1 = tensor absent
    long        n_bytes = 0;

    // THIS RANK'S logical shape, ggml ne order. Identical to the file's shape at
    // tp_size 1, and a band of it otherwise. Carried on the tensor rather than
    // recomputed by the forward from cfg + rule, because the two would be free to
    // disagree: block_aligned_band() hands a remainder to the low ranks, so per-rank
    // extents legitimately differ by one block and a forward that derives
    // "cfg.hidden / tp_size" would be wrong on exactly the ranks that differ.
    long rank_ne[4] = {1, 1, 1, 1};

    bool ok() const { return data != nullptr; }
};

struct KimiK3LayerWeights {
    bool is_kda = false;

    KimiK3Tensor attn_norm, ffn_norm;
    KimiK3Tensor attn_res_score, ffn_res_score;   // 1-D scorers, see kimi_k3_decode_plan.h

    // KDA
    KimiK3Tensor attn_q, attn_k, attn_v;
    KimiK3Tensor ssm_conv1d_q, ssm_conv1d_k, ssm_conv1d_v;   // f32, [d_conv,1,qkv]
    KimiK3Tensor ssm_f_a, ssm_f_b;
    KimiK3Tensor ssm_beta;
    KimiK3Tensor ssm_dt_bias;    // f32, 1-D [qkv]
    KimiK3Tensor ssm_a;          // f32, 1-D [n_head]
    KimiK3Tensor ssm_g;
    KimiK3Tensor ssm_norm;       // f32, 1-D [head_dim]
    KimiK3Tensor attn_output;    // KDA's wo; MLA's wo is the same field, see below

    // MLA
    bool has_q_lora = false;
    KimiK3Tensor attn_q_a, attn_q_a_norm, attn_q_b;   // q_lora path
    KimiK3Tensor attn_q_dense;    // dense-q fallback when !has_q_lora (MLA's attn_q.weight;
                                 // a distinct field from KDA's attn_q above, since a KDA
                                 // layer and an MLA layer never coexist in the same
                                 // KimiK3LayerWeights instance but the struct is shared)
    KimiK3Tensor attn_kv_a_mqa, attn_kv_a_norm;
    bool has_fused_kv_b = false;
    KimiK3Tensor attn_kv_b;                 // fused fallback (NOT numerically wired yet —
                                            // see kimi_k3_decode_plan.h's note on why the
                                            // split it needs is not derived)
    KimiK3Tensor attn_k_b, attn_v_b;         // split path (expected for the real file)
    bool has_attn_gate = false;
    KimiK3Tensor attn_gate;
    // attn_output (wo) reuses the KDA field above — both branches write the same
    // GGUF tensor name and the same [*, hidden] shape contract.

    // FFN — leading dense
    KimiK3Tensor ffn_gate, ffn_up, ffn_down;

    // FFN — routed MoE
    KimiK3Tensor ffn_gate_inp;
    KimiK3Tensor exp_probs_b;                // 1-D [n_expert] bias
    KimiK3Tensor ffn_gate_exps, ffn_up_exps, ffn_down_exps;   // native quant, per-expert
    KimiK3Tensor ffn_routed_down, ffn_routed_up;
    bool has_routed_norm = false;
    KimiK3Tensor ffn_routed_norm;
    bool has_shared_experts = false;
    KimiK3Tensor ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
};

struct KimiK3Weights {
    KimiK3Tensor token_embd;
    KimiK3Tensor output_norm;
    KimiK3Tensor output;             // lm_head
    bool has_output_res_score = false;
    KimiK3Tensor output_res_score;
    std::vector<KimiK3LayerWeights> layers;

    // HOW MUCH OF THE MODEL TO SHARD.
    //
    //   ExpertsOnly  Band the 896 routed experts across ranks; replicate everything
    //                else. The routed experts are ~531 of UD-IQ1_S's 553 GiB, so this
    //                captures essentially the whole memory win, and it needs exactly
    //                ONE collective per MoE layer (the expert partial sum at
    //                expert_latent) instead of two. Critically, no activation changes
    //                width, so the forward runs at full cfg dims on every rank and
    //                needs no per-rank shape threading.
    //
    //   ExpertsAndKda  ExpertsOnly, plus a head-parallel shard of the 69 KDA
    //                attention layers: attn_q/k/v/ssm_g/ssm_f_b row-shard by qkv,
    //                attn_output col-shards, and the recurrent state divides with
    //                them. Costs ONE extra all-reduce per KDA layer, at hidden,
    //                after attn_output.
    //
    //                MEASURED, nsys at ctx 131072 on main + #57: the two Q8_0
    //                projection kernels are 35.7% of GPU kernel time and the KDA
    //                share of them is head-parallel, so ExpertsOnly has all eight
    //                ranks streaming the same ~468 MB of KDA weights per layer
    //                every token — 32.3 GB per rank per token, 8x redundant.
    //
    //                MLA IS DELIBERATELY LEFT REPLICATED. It is MQA: all 96 query
    //                heads attend over ONE latent cache, so sharding query heads
    //                divides the FLOPs but not the cache read. It buys a fraction
    //                and pays a full collective.
    //
    //   Full         Also row/col-shard the MLA and dense-FFN projections, per
    //                weight_plan. NOT YET SUPPORTED by the forward — the loader
    //                would shard weights the executor still indexes at full width,
    //                which reads past the end of the slice.
    //
    // Kept explicit rather than inferred from tp_size so that turning one on is a
    // deliberate act with a matching forward, not a silent consequence of tp>1.
    //   ExpertsAndMla  Also head-band the 24 MLA layers' per-head projections
    //                (q_b, k_b, v_b, gate) and col-shard their output. The latent
    //                KV projection and the KV cache stay REPLICATED — that is what
    //                MQA means. Costs one all-reduce per MLA layer.
    //
    // MLA IS SHARDED AND KDA IS NOT, WHICH LOOKS BACKWARDS UNTIL IT IS PRICED.
    // KDA is 69 of the 93 layers and MLA only 24, but measured on the shipped
    // build MLA attention is 36.9% of GPU kernel time against KDA's 6.6% — 16x
    // more per layer. Both cost one collective per layer, and collectives are
    // ~0.088% of GPU time each, so:
    //
    //     shard KDA:  +5.8% compute, -6.1% collectives  =  -0.3%, a REGRESSION
    //     shard MLA: +32.3% compute, -2.1% collectives  = +30.2%
    //
    // ExpertsAndKda therefore stays available and stays OFF; it is kept because
    // the arithmetic above is only obvious once both branches exist to compare.
    //   ExpertsAndAttn Both attention bands. One all-reduce per layer, 93 of them
    //                on top of the 92 MoE ones.
    //
    // WHY BOTH, AFTER THE ARITHMETIC SAID KDA ALONE LOSES MONEY. That estimate
    // priced a collective at ~0.088% of GPU time, taken from a profile whose
    // NCCL time was inflated by rank-arrival skew (35 ms max against a 71 us
    // mean). With submission parallelised that skew is gone and a reduce is far
    // cheaper than the estimate, so the 5.8% of KDA compute is worth its 69
    // extra reduces after all. Measured head-to-head on this box: an all-93
    // shard (PR #64) runs 63.6 ms against this tree's MLA-only 70.3.
    //
    // The narrower policies stay because they are the evidence: ExpertsAndMla is
    // what isolates the MLA band's contribution, and ExpertsAndKda the KDA one.
    enum class ShardPolicy { ExpertsOnly, ExpertsAndKda, ExpertsAndMla, ExpertsAndAttn, Full };
    ShardPolicy policy = ShardPolicy::ExpertsOnly;

    // Which bands a policy shards. Every site that used to compare the enum
    // directly asks through these instead — adding ExpertsAndAttn by hand at
    // each `== ExpertsAndKda` would have been six chances to miss one, and a
    // missed site shards weights the forward still indexes at full width.
    static bool shards_kda(ShardPolicy p) {
        return p == ShardPolicy::ExpertsAndKda || p == ShardPolicy::ExpertsAndAttn;
    }
    static bool shards_mla(ShardPolicy p) {
        return p == ShardPolicy::ExpertsAndMla || p == ShardPolicy::ExpertsAndAttn;
    }

    // WHICH SLICE OF EACH TENSOR THIS RANK HOLDS. Set BEFORE calling the loader;
    // the default (tp_size 1) makes every rule degenerate to Replicate, so the
    // loader takes byte-for-byte the same path it always did and the single-GPU
    // build is unchanged.
    //
    // It lives on the weights rather than being threaded through the loader's ~49
    // upload call sites because it is a property of the loaded model, and the
    // forward needs it too: an all-reduce whose tp_size disagrees with the tp_size
    // the weights were sharded under is a partial sum presented as an answer.
    tp::ShardDims shard;

    // This rank's KDA slice, under ShardPolicy::ExpertsAndKda. Left at the tp=1
    // identity (full width, zero offsets) under every other policy, so the forward
    // reads it unconditionally and the unsharded path is the same code with the
    // same values it always had — not a branch that can drift.
    tp::KdaShardDims kda;

    // This rank's MLA head slice, under ShardPolicy::ExpertsAndMla. Same
    // discipline as `kda`: left at the tp=1 identity under every other policy so
    // the forward reads it unconditionally and the replicated path is the same
    // code with the same values, not a branch that can drift out of step.
    tp::MlaShardDims mla;

    // Every device buffer this struct owns, for bulk free in the destructor.
    std::vector<void*> owned;
};

// Load every tensor a layer RANGE needs from the GGUF, uploading native bytes.
// `first_layer`/`last_layer` (inclusive) let a caller load only what a partial
// shard set actually has — see kimi_k3_layer_coverage() in
// kimi_k3_gguf_manifest.h to find out which layers that is. Layers outside the
// range are left as default-constructed (all KimiK3Tensor::ok() == false).
//
// Returns false on a real error (cudaMalloc/cudaMemcpy failure, or a required
// tensor within the requested range that turned out absent) — never on a tensor
// outside the range, since that is expected under partial loading.
bool kimi_k3_load_weights(const GGUF& g, const KimiK3Config& cfg,
                         const K3PlanOptions& opt, KimiK3Weights& out,
                         int first_layer, int last_layer);

// Same, but with control over the NON-layer tensors. A pipeline stage needs this:
// token_embd belongs on the FIRST stage's device and output_norm/output/
// output_res_score on the LAST stage's, so the middle stages must load neither.
// Loading them everywhere would work but waste ~2.5 GiB per stage (token_embd and
// output are each 163840 x 7168) on tensors that stage can never use.
bool kimi_k3_load_weights_scoped(const GGUF& g, const KimiK3Config& cfg,
                                const K3PlanOptions& opt, KimiK3Weights& out,
                                int first_layer, int last_layer,
                                bool load_embed, bool load_head);

void kimi_k3_free_weights(KimiK3Weights& w);

// ---------------------------------------------------------------------------
// Runtime state: everything that persists ACROSS decode steps and must reset at
// position 0. Three kinds, one per K3 mechanism that isn't stateless matmuls:
//
//   KDA recurrent state    conv window (x3: Q/K/V) + delta-rule state matrix,
//                           one slot per KDA layer
//   MLA KV cache            a flat [key_length, max_ctx] buffer per MLA layer —
//                           NOT a paged pool; this executor targets correctness
//                           validation against short prompts, not production
//                           context lengths. max_ctx bounds it.
//   Cross-layer residual   a growing bank of raw hidden-width checkpoints,
//   ring buffer             at most ceil(n_layers / block_size) entries for the
//                           whole model (shared across ALL layers, not per-layer)
struct KimiK3RuntimeState {
    int max_ctx = 0;
    int position = 0;   // next token's position; incremented by forward_token

    // THE DEVICE'S COPY OF `position`, and why both exist.
    //
    // Kernels must read the position from memory or a captured graph replays the one
    // frozen at capture time. But the LAUNCH PLAN (grid size, which MLA kernel) is a host
    // decision a graph cannot change, and that decision needs the position too. So the
    // host mirror above stays authoritative for planning, d_pos is authoritative for
    // arithmetic, and they advance together — the host increments, and bump_pos runs
    // inside the captured region. kimi_k3_check_pos_sync() exists so a test can prove
    // they never drift; drift here is invisible for thousands of tokens and then wrong.
    int* d_pos = nullptr;   // device int, 4 bytes, owned by kimi_k3_alloc_state()

    // Sizes needed to zero each buffer correctly in kimi_k3_reset_state() — stored
    // here rather than re-threading a KimiK3Config through that call. Populated by
    // kimi_k3_alloc_state().
    int conv_state_elems = 0;    // (kda_conv_kernel-1) * qkv, per KDA layer
    int delta_state_elems = 0;   // kda_head_dim^2 * n_q_heads, per KDA layer
    int kv_cache_elems = 0;      // key_length * max_ctx, per MLA layer
    int res_bank_row_elems = 0;  // hidden, per banked checkpoint

    // Indexed by KDA-layer ordinal (0..n_kda_layers-1), NOT the global layer index —
    // see kda_layer_ordinal() below.
    std::vector<float*> conv_state_q, conv_state_k, conv_state_v;   // [d_conv-1, qkv]
    std::vector<float*> delta_state;                               // [head_dim, head_dim, n_head]

    // Indexed by MLA-layer ordinal (0..n_mla_layers-1).
    //
    // Element type is f16 when kv_f16 is set (the default — llama.cpp's own
    // type_k, and the largest byte item a decode token moves), f32 when
    // SPARKINFER_K3_KV_F16=0 restores the historical layout. The pointers stay
    // float* so every existing null-check and container stays as-is; the three
    // touch points (alloc size, reset memset, store/attend in the forward) all
    // consult kv_f16, and nothing else dereferences these buffers.
    std::vector<float*> mla_kv_cache;   // [key_length, max_ctx]
    bool kv_f16 = false;                // set by kimi_k3_alloc_state, fixed for life

    // Shared across the whole model — one bank, not one per layer.
    float* res_bank = nullptr;   // [hidden, max_ckpt]
    int    n_ckpt = 0;           // live count
    int    max_ckpt = 0;

    std::vector<void*> owned;
};

// Allocate recurrent state. By default (first_layer/last_layer defaulted) sizes for
// the WHOLE model. A pipeline stage passes its own [first_layer, last_layer] so it
// allocates state ONLY for the layers it owns — the state vectors keep their
// full-model length and global-ordinal indexing, but out-of-range slots stay null
// and are never touched. This is what lets the layer-split pipeline fit: the MLA KV
// cache is per-MLA-layer and grows with context (~58 GB total at 1M), so an
// every-stage-allocates-everything sizing would need that PER STAGE.
// `kda` sizes the KDA recurrent state to THIS RANK's heads under
// ShardPolicy::ExpertsAndKda. Null (the default) means full width, which is what
// every unsharded caller wants and what tp=1 must keep getting. Passing a sharded
// KdaShardDims while the forward still indexes at full width would read past the
// end of the allocation, so the two are changed together or not at all.
bool kimi_k3_alloc_state(const KimiK3Config& cfg, int max_ctx, KimiK3RuntimeState& out,
                        int first_layer = -1, int last_layer = -1,
                        const tp::KdaShardDims* kda = nullptr);
// Move the decode position. Writes BOTH the host mirror (which picks the MLA launch plan)
// and the device copy (which the KV store and attention index with). Never set
// state.position directly — the two must not drift, and a caller that sets only the host
// side produces a benchmark that runs fast because it is attending over one position.
bool kimi_k3_set_position(KimiK3RuntimeState& s, int pos);

void kimi_k3_reset_state(KimiK3RuntimeState& s);   // zero everything, position=0, n_ckpt=0
void kimi_k3_free_state(KimiK3RuntimeState& s);

// Map a global layer index to its ordinal within its own type (KDA or MLA), so
// per-type state vectors can be indexed contiguously rather than one slot per
// layer with most of them unused.
int kimi_k3_kda_ordinal(const KimiK3Config& cfg, int layer);   // -1 if layer is MLA
int kimi_k3_mla_ordinal(const KimiK3Config& cfg, int layer);   // -1 if layer is KDA

// ---------------------------------------------------------------------------
// The executor.
//
// Optional debug callback, invoked at named checkpoints with a DEVICE pointer and
// element count — mirrors the reference's `cb(tensor, "tag", il)` calls closely
// enough that a tag/layer pair here should name-match what a debug build of
// llama.cpp would report for the same graph node, which is what makes per-layer
// comparison possible before the whole model is validated end to end.
//
// Two tiers of tags. The coarse ones ("attn_res_mix", "attn_norm", "kda_out"/
// "mla_out", "ffn_norm", "ffn_out", "l_out") are the ones a real cross-check
// against llama.cpp's own graph would target. The "dbg_*" tags inside the KDA
// branch (dbg_conv_q, dbg_conv_v, dbg_l2_q, dbg_decay_g, dbg_beta, dbg_delta_out,
// dbg_gate_out) and the dense-FFN branch (dbg_dense_gate/up/situ) exist because
// bisecting kimi_k3_layer0_ref_check.cpp's first real divergence against an
// independent float64 reference needed them — the coarse tags alone narrowed the
// bug to "somewhere in the KDA branch", not to the specific step. Kept rather than
// stripped back out: the next layer this executor gets extended to validate (an
// MLA layer, or a MoE one) will need the same kind of bisection, and these cost
// nothing when fwd.debug is unset.
using KimiK3DebugFn = std::function<void(const char* tag, int layer, const float* dev_ptr, int64_t n)>;

struct KimiK3Forward {
    const KimiK3Config* cfg = nullptr;
    const KimiK3Weights* w = nullptr;
    KimiK3RuntimeState* state = nullptr;
    K3PlanOptions opt;
    cudaStream_t stream = nullptr;   // null = the default stream
    KimiK3DebugFn debug;             // may be empty

    // Scratch buffers, sized once and reused every call. Allocated by
    // kimi_k3_forward_alloc_scratch(); forward_token() assumes they exist.
    struct Scratch;
    Scratch* s = nullptr;

    // PREFILL TILE, or null for decode. When set and its `layer` matches the layer being
    // run, the KDA q/k/v/g projections have ALREADY been computed for a tile of tokens and
    // this call takes row `prefill_tok` instead of projecting again.
    //
    // Null for every decode call, which is what keeps the decode guard safe by
    // construction rather than by care: decode never reads these fields, so a bug here
    // cannot move the 128k number that #132 makes fatal to the whole round.
    //
    // The copy this enables is 4 x qkv floats (24 KB at K3's per-rank qkv=1536) against a
    // projection that streams 1536 x 7168 Q8_0 = ~11.7 MB. Two thousandths of the cost it
    // replaces, so pointing the layer at the tile row would save nothing worth the aliasing
    // risk of handing `s.qkv_*` a buffer it does not own.
    const void* prefill_tile = nullptr;   // K3PrefillTile*, opaque here to avoid the include
    int prefill_tok = 0;                  // which row of the tile this call is
};

bool kimi_k3_forward_alloc_scratch(const KimiK3Config& cfg, KimiK3Forward& fwd);
void kimi_k3_forward_free_scratch(KimiK3Forward& fwd);

// Run one decode step. `out_logits` is host-visible (a cudaMemcpy is done into it)
// and must have room for cfg.vocab floats. Advances state->position. Every layer
// referenced must have its weights loaded (KimiK3Tensor::ok()) or this returns
// false rather than reading a null pointer.
bool kimi_k3_forward_token(KimiK3Forward& fwd, int token_id, float* out_logits);

// Run a single layer only, for per-layer validation against llama.cpp before the
// whole model's weights are available. `hidden_in`/`hidden_out` are device
// pointers, [hidden] each; hidden_in is NOT modified. Consumes/produces the same
// state slot forward_token would for this layer, so a sequence of single-layer
// calls at increasing positions is equivalent to forward_token calling this layer
// as part of the full loop — the two are meant to be cross-checked against each
// other, not just against llama.cpp.
bool kimi_k3_forward_layer(KimiK3Forward& fwd, int layer, const float* hidden_in,
                          float* hidden_out);

// ---------------------------------------------------------------------------
// The same layer, split at the two points where TP has to insert a collective.
//
// WHY A PHASE SPLIT RATHER THAN A REDUCE CALLBACK. The obvious design — hand the
// forward a `reduce(buf, n)` callback it invokes in place — CANNOT WORK here. The
// collective is a GROUP operation: NCCL requires every rank's ncclAllReduce to be
// enqueued inside one ncclGroupStart/End before any of them launches, and
// sparkinfer drives all ranks from a SINGLE host thread. A callback fired from
// inside rank r's forward would block waiting for ranks that thread has not reached
// yet. That is not hypothetical — it is the deadlock recorded in collective.h,
// measured on an 8x H200 node, which hung at the first collective with no output.
//
// So the driver must own the interleaving: run phase N on EVERY rank, issue ONE
// group collective, then run phase N+1 on every rank.
//
//   Attn        res_mix, bank push, attn_norm, the attention branch, attn_output.
//               attn_output is ColShard, so this leaves a FULL-WIDTH PARTIAL SUM.
//                 -> reduce at hidden
//   FfnPartial  attention residual combine, ffn_res_mix, ffn_norm, then either the
//               dense FFN (down is ColShard -> partial at hidden) or the MoE path
//               (router, routed_down, expert dispatch -> partial at expert_latent,
//               because the experts are ExpertShard and this rank ran only its own).
//                 -> reduce at expert_latent (MoE) or hidden (leading dense)
//   FfnFinish   routed_norm, routed_up, shared experts, the FFN residual add.
//               Everything here is Replicate and reads the ALREADY-REDUCED value,
//               which is exactly why the collective cannot be deferred past it:
//               routed_norm is an rms_norm and rms_norm is not linear.
//
// `All` runs all three back to back and is what kimi_k3_forward_layer calls, so the
// tp_size 1 path is unchanged. Running All must be bit-identical to running the
// three phases in sequence — asserted by kimi_k3_tp_phase_check.
enum class K3LayerPhase { All = 0, Attn = 1, FfnPartial = 2, FfnFinish = 3 };

bool kimi_k3_forward_layer_phase(KimiK3Forward& fwd, int layer, K3LayerPhase phase,
                                const float* hidden_in, float* hidden_out);

// The buffer holding THIS RANK'S partial sum after `phase`, and its element count.
// Returns nullptr with *count = 0 for a phase that produces no partial (FfnFinish).
//
// Exists because Scratch is opaque to callers: the TP driver has to hand the exact
// device pointer and width to the collective, and guessing either — reducing
// hidden width after the MoE dispatch, say, when the partial is expert_latent wide —
// reduces the wrong bytes and still returns success.
float* kimi_k3_partial_buffer(KimiK3Forward& fwd, int layer, K3LayerPhase phase,
                             int* count);

// Repoint the buffer kimi_k3_partial_buffer() exposes for `phase`; returns the
// previous pointer. The TP driver uses this to aim a phase's partial straight at
// a Mode-B collective's reduce_in() before the phase runs, then at reduce_out()
// after the reduce — the downstream phase reads the sum where the collective
// wrote it and the staging copies disappear. Safe against teardown: scratch
// frees via its owned-pointer list, never via these fields. Attn swaps the
// attention partial; FfnPartial swaps the MoE accumulator (the leading dense
// layer's FFN partial is replicated, never reduced, and stays where it is).
float* kimi_k3_swap_partial_buffer(KimiK3Forward& fwd, K3LayerPhase phase,
                                  float* buf);

// ---------------------------------------------------------------------------
// Chunked prompt ingestion.
//
// Prompt ingestion is prefill, and prefill runs the DECODE step once per token:
// every one of a 32k prompt's tokens pays 93 layers of launches, 185 collectives
// and one output head. The work per token is fixed, but almost none of it is
// per-token BOUND — the projections are GEMVs that want to be GEMMs, the
// collectives are 185 rendezvous that could be 185/B, and the head is needed once.
//
// A chunked walk carries B tokens through layer L before starting layer L+1. What
// makes that legal is that the only cross-token dependencies in this model run
// along the RECURRENT axis, not the layer axis: the KDA conv/delta state and the
// MLA KV cache are consumed in token order WITHIN a layer, and nothing in layer L
// reads a later layer's output. So the per-token phases still run in token order
// inside each layer, and the batching is in what surrounds them.
//
//   kimi_k3_forward_alloc_batch     allocate B slots (call after alloc_scratch,
//                                   and after fwd.state is set — the residual
//                                   bank is sized from state->max_ckpt).
//   kimi_k3_forward_batch_begin     start a chunk: clear every slot's checkpoint
//                                   count.
//   kimi_k3_forward_select_slot     address slot t. Pointer arithmetic only; the
//                                   whole of forward_layer_phase then runs on
//                                   that token unmodified.
//   kimi_k3_forward_batch_end       give the decode path its own residual bank
//                                   back. MUST be called before the next
//                                   forward_token.
//   kimi_k3_batch_partial_buffer    the base and TOTAL element count of a phase's
//                                   partial across n_slots, so the driver reduces
//                                   B tokens in one collective instead of B.
//
// B == 1 takes the same path with a one-slot arena and is bit-identical to the
// token loop — which is what makes it a usable control: any divergence at B=8 is
// the batching, not the batched code.
bool kimi_k3_forward_alloc_batch(const KimiK3Config& cfg, KimiK3Forward& fwd,
                                 int batch);
int  kimi_k3_forward_batch_capacity(const KimiK3Forward& fwd);
void kimi_k3_forward_batch_begin(KimiK3Forward& fwd);
void kimi_k3_forward_select_slot(KimiK3Forward& fwd, int slot);
void kimi_k3_forward_batch_end(KimiK3Forward& fwd);
float* kimi_k3_batch_partial_buffer(KimiK3Forward& fwd, int layer,
                                    K3LayerPhase phase, int n_slots, int* count);

// Print the SPARKINFER_K3_PROFILE=1 phase breakdown to stderr (no-op when unset).
void kimi_k3_profile_report();

// ---------------------------------------------------------------------------
// Layer-split pipeline across multiple GPUs.
//
// WHY LAYER-SPLIT RATHER THAN TENSOR-PARALLEL, for this model on this hardware:
// UD-IQ1_S is 594 GB and 8x H200 is 1128 GB, so the model FITS with each GPU
// owning a contiguous range of layers. That needs no collectives at all — just a
// device-to-device handoff of a few hundred KB at each stage boundary — where TP
// would need 186 all-reduces per token (2 per layer x 93). The TP shard rules and
// reduce points in kimi_k3_decode_plan.h remain correct and are what a TP path
// would use; this is simply the cheaper answer when the model fits.
//
// It also reuses kimi_k3_forward_layer completely unchanged: a stage is just a
// KimiK3Weights loaded over a layer range plus its own KimiK3RuntimeState, both of
// which the single-GPU code already supports.
//
// THE HANDOFF IS NOT JUST THE HIDDEN STATE. The cross-layer residual bank is READ
// AT EVERY LAYER and written at every attn_res.block_size-th one, so it spans the
// whole 93-layer pass. A stage boundary at layer 40 means layer 41 mixes over
// checkpoints banked back on layer 36 — on a different GPU. So each boundary
// transfers the bank (and n_ckpt) alongside the hidden vector. Transferring only
// the hidden state is the obvious implementation and it silently corrupts every
// mix after the first boundary: the model stays fluent, the residual scale drifts.
// Per boundary that is hidden (7168 f32 = 28 KiB) plus up to
// ceil(n_layers/block_size) = 8 bank rows (229 KiB) — trivial next to the ~6 GiB
// of weights each stage already holds.
struct KimiK3PipelineStage {
    int device = 0;
    int first_layer = 0;
    int last_layer = -1;
    KimiK3Weights weights;
    KimiK3RuntimeState state;
    KimiK3Forward fwd;
    float* hidden = nullptr;    // [hidden], on this stage's device
    float* logits = nullptr;    // [vocab], last stage only
};

struct KimiK3Pipeline {
    KimiK3Config cfg;
    K3PlanOptions opt;
    std::vector<KimiK3PipelineStage> stages;
    std::vector<float> host_hidden;   // staging for cross-device handoff
    std::vector<float> host_bank;     // staging for the residual bank
};

// Split cfg.n_layers across `devices`, balancing by each layer's real byte cost
// (the leading dense layer is ~1.2 GiB against ~6 GiB for a MoE layer, so an even
// layer-count split would leave stage 0 badly under-loaded). Pass the same device
// twice to simulate a multi-stage split on one GPU — which is how the handoff
// logic gets tested without 8 GPUs.
bool kimi_k3_pipeline_init(const GGUF& g, const KimiK3Config& cfg,
                          const K3PlanOptions& opt,
                          const std::vector<int>& devices, int max_ctx,
                          KimiK3Pipeline& out);

// One decode step through every stage in order. Equivalent to
// kimi_k3_forward_token on a single device, and that equivalence is the test.
bool kimi_k3_pipeline_forward_token(KimiK3Pipeline& p, int token_id, float* out_logits);

void kimi_k3_pipeline_free(KimiK3Pipeline& p);

}  // namespace sparkinfer
