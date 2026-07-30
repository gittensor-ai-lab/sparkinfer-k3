#pragma once
// The Kimi K3 decode step as an explicit, inspectable SCHEDULE: which op runs, in
// what order, on what width, consuming which GGUF tensor, and where the two
// collectives fall. DELIBERATELY CUDA-FREE — building the schedule launches nothing,
// so the whole thing is unit-testable on a laptop (runtime/tests/
// kimi_k3_decode_plan_cpu_test.cpp).
//
// WHY A PLAN OBJECT INSTEAD OF JUST WRITING THE FORWARD PASS
//
// K3's decode has three properties that make a hand-written forward pass hard to
// trust, and all three are checkable if the schedule is data:
//
//  1. TWO LAYER TYPES. 93 layers, 24 MLA and 69 KDA, interleaved by a per-layer GGUF
//     array. Running an MLA layer's op sequence on a KDA layer's weights produces
//     well-formed activations of the right width. It is not a crash.
//
//  2. TWO WIDTHS. The routed experts do NOT live at hidden width. They live at
//     expert_latent = 3584, reached via ffn_routed_down and left via ffn_routed_up,
//     with the per-expert intermediate at 3072. Four widths in one FFN block
//     (7168 -> 3584 -> 3072 -> 3584 -> 7168), and a mixup is again just a wrong
//     number of the right shape.
//
//  3. EXACTLY TWO COLLECTIVES PER LAYER, after attn_output and after the MoE's
//     return to hidden width. One missing all-reduce leaves each rank holding a
//     partial sum that looks like a plausible activation. One extra all-reduce
//     multiplies an already-complete tensor by tp_size. Neither is a crash, and at
//     186 reduces per token there is no chance of spotting either by eye.
//
//  4. THE CROSS-LAYER RESIDUAL MIX RUNS BEFORE EACH NORM, NOT AFTER — and a
//     checkpointed layer REPLACES the residual stream rather than adding to it.
//     Read directly off unslothai/llama.cpp's graph builder
//     (llama_model_kimi_k3::graph, src/models/kimi-k3.cpp), the per-layer order is:
//
//       cur = res_mix(prefix_sum, attn_res_score)      <- BEFORE attn_norm
//       if il % block_size == 0: bank(prefix_sum)      <- RAW pre-mix value, attn side only
//       cur = attn_norm(cur); cur = attention(cur)
//       prefix_sum = banked ? cur : prefix_sum + cur    <- REPLACE on a banked layer
//       cur = res_mix(prefix_sum, ffn_res_score)        <- BEFORE ffn_norm, no bank push
//       cur = ffn_norm(cur); cur = ffn_or_moe(cur)
//       prefix_sum = prefix_sum + cur                   <- FFN side ALWAYS adds
//
//     An earlier version of this plan had res_mix running AFTER attention/FFN and
//     the residual combine as an unconditional add on both sides — plausible, and
//     wrong on both counts. The mix must precede the norm because it is choosing
//     what to attend/FFN over, not what to fold in after; and the attention-side
//     replace exists specifically so a banked checkpoint is not double-counted
//     (once in the bank, again in the running sum this layer would otherwise have
//     kept adding it into).
//
// So the schedule is emitted as a list, and the tests assert the properties that
// would otherwise only show up as degraded output quality: widths chain, layer types
// use their own ops, the reduce count is exactly 2*n_layers, and — the strongest
// check available without a GPU — EVERY TENSOR THE PLAN CONSUMES IS A TENSOR THE
// MANIFEST REQUIRES, and vice versa. Plan and manifest are written from different
// sources (the kernel signatures vs the reference loader), so agreement between them
// is real evidence rather than a restatement.
//
// This is a schedule, not an executor. It carries no buffers and no streams; the GPU
// path consumes it. That split is the point: the ordering and the widths are decided
// and tested here, where a mistake is a failing assert instead of a fluent model that
// scores badly three weeks later.

#include <string>
#include <vector>

#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/tp/shard.h"
#include "sparkinfer/tp/weight_plan.h"

namespace sparkinfer {

// One op in the decode schedule. Named after the kernel that implements it where a
// kernel exists, so plan coverage and kernel coverage are the same question.
enum class K3Op {
    EmbedLookup,      // token_embd row gather
    RmsNorm,          // f32 rms norm, elementwise over hidden
    MatMul,           // generic GEMV against a GGUF weight
    AddResidual,      // x += y
    AllReduce,        // the collective; only after a ColShard consumer

    // situ activation
    Situ,             // situ_f32

    // KDA (linear / recurrent) path
    KdaConvStep,      // kda_conv_step_f32     depthwise causal short conv
    KdaDecayGate,     // kda_decay_gate_f32    per-channel decay from a/dt
    L2NormHeads,      // l2_norm_heads_f32     per-head L2 on q,k
    KdaDecodeStep,    // kda_decode_step_f32   gated delta rule, one token
    KdaGateOut,       // kda_gate_out_f32      output gate

    // MLA (full attention) path
    MlaAbsorbQ,       // mla_absorb_q_f32      fold k_b into q
    MlaDecodeAttn,    // mla_decode_attn_f32   NoPE-only, latent kv
    MlaGateOut,       // mla_gate_out_f32      sigmoid output gate

    // cross-layer residual attention
    AttnResMix,       // attn_res_mix_f32
    ResBankPush,      // bank the raw pre-mix residual into the checkpoint ring buffer;
                      // bookkeeping only (a device-side copy), no dedicated kernel

    // MoE
    MoeRouter,        // moe_router_noaux_tc_f32
    MoeDispatch,      // moe_expert_ffn_iq2xs_f32
};

const char* k3_op_name(K3Op op);

// True for ops backed by a dedicated kernel in kernels/include/sparkinfer/kernels/
// kimi_k3.h. Used to assert the plan reaches every kernel that was written — an
// unreached kernel is either dead code or a step the forward pass forgot.
bool k3_op_has_dedicated_kernel(K3Op op);

struct K3Step {
    K3Op        op = K3Op::MatMul;
    const char* label = "";
    int         layer = -1;        // -1 for pre/post-layer steps
    bool        is_kda_layer = false;

    // Widths in ELEMENTS, full (unsharded) logical width. The per-rank width is
    // derived by the executor from the tensor's shard rule; keeping the logical
    // width here is what makes the chaining assertions meaningful.
    int in_dim  = 0;
    int out_dim = 0;

    // The GGUF tensor consumed, "" for ops that consume none (Situ, AddResidual,
    // AllReduce). Fully qualified, so it can be looked up directly.
    std::string tensor;
    tp::Rule    rule = tp::Rule::Replicate;

    // What the GGUF tensor's dims MUST be, for k3_validate_plan_shapes(). Zero means
    // "not asserted" — used where the width is inferred from the kernel signatures
    // rather than verified against the file, so an unpinned dim shows up as a
    // reported unknown instead of a silent assumption. Running the validator once
    // against real weights is what turns them into pinned values.
    int expect_ne0 = 0;   // contracted / fastest dim
    int expect_ne1 = 0;
    int expect_ne2 = 0;

    // Set on the step that must be followed by a collective. Kept as a flag on the
    // producing step as well as an explicit AllReduce step so the two can be checked
    // against each other.
    bool needs_reduce = false;

    // Set on the ATTENTION-side residual-combine step (label "add_attn" /
    // "replace_attn") when this layer is a checkpoint layer (i % block_size == 0).
    // true means REPLACE the residual stream with the attention output; false means
    // the ordinary add. Never set on the FFN-side combine, which always adds — see
    // the header comment on why the two sides differ.
    bool residual_replace = false;

    std::string note;
};

struct K3DecodePlan {
    std::vector<K3Step> steps;
    int n_reduces = 0;
    int n_kda_layers = 0;
    int n_mla_layers = 0;

    // Every distinct tensor the plan consumes, sorted. The manifest cross-check
    // compares this against the required set.
    std::vector<std::string> tensors() const;
};

// Build the decode schedule for one token.
//
// `d` supplies tp_size / rank so the plan can attach shard rules; it does not change
// the op sequence. TP is a placement decision, not a different graph — if changing
// tp_size changed the schedule, that alone would be a bug.
//
// `has_q_lora` / `has_attn_gate` / `has_fused_kv_b` select between the layouts the
// reference loader accepts (q_a+q_b vs dense q; k_b/v_b split vs fused kv_b;
// attn_gate present or absent). The GGUF decides these, so they are inputs rather
// than assumptions.
struct K3PlanOptions {
    bool has_q_lora     = true;    // attn_q_a + attn_q_b rather than a dense attn_q
    bool has_attn_gate  = false;   // attn_gate is TENSOR_NOT_REQUIRED upstream
    bool has_fused_kv_b = false;   // attn_kv_b rather than attn_k_b + attn_v_b
    bool has_shared_experts = false;
    bool has_routed_norm    = false;
};

K3DecodePlan build_k3_decode_plan(const KimiK3Config& cfg, const tp::ShardDims& d,
                                  const K3PlanOptions& opt = {});

// ---------------------------------------------------------------------------
// Check the plan's declared widths against a real GGUF.
//
// The plan is written from the kernel signatures and the config; the GGUF is ground
// truth. Any step whose expect_ne* disagrees with the file is a width the forward
// pass would get wrong — and for a GEMV a wrong contracted dim reads adjacent
// weights rather than faulting, so it must be caught here.
//
// Steps with expect_ne* == 0 are UNPINNED: the width was inferred, not verified. The
// report lists their actual dims so they can be pinned. That is the intended workflow
// — run this once where the weights live, then hard-code what it found.
struct K3PlanShapeReport {
    int checked = 0;
    int mismatched = 0;
    int missing = 0;
    int unpinned = 0;
    std::vector<std::string> problems;   // capped
    std::vector<std::string> unpinned_shapes;
};

// Needs the GGUF reader; declared here and defined in the .cpp so this header stays
// dependency-light for callers that only build the plan.
class GGUF;
bool k3_validate_plan_shapes(const K3DecodePlan& plan, const GGUF& g,
                             K3PlanShapeReport* report = nullptr);

// Human-readable dump of one layer's steps — for a load-time log and for eyeballing
// a diff when the schedule changes.
std::string k3_plan_layer_dump(const K3DecodePlan& plan, int layer);

}  // namespace sparkinfer
