#include "sparkinfer/models/kimi_k3_decode_plan.h"

#include "sparkinfer/gguf.h"

#include <algorithm>
#include <map>
#include <cstdio>
#include <sstream>

namespace sparkinfer {

const char* k3_op_name(K3Op op) {
    switch (op) {
        case K3Op::EmbedLookup:   return "embed_lookup";
        case K3Op::RmsNorm:       return "rms_norm";
        case K3Op::MatMul:        return "matmul";
        case K3Op::AddResidual:   return "add_residual";
        case K3Op::AllReduce:     return "ALL_REDUCE";
        case K3Op::Situ:          return "situ";
        case K3Op::KdaConvStep:   return "kda_conv_step";
        case K3Op::KdaDecayGate:  return "kda_decay_gate";
        case K3Op::L2NormHeads:   return "l2_norm_heads";
        case K3Op::KdaDecodeStep: return "kda_decode_step";
        case K3Op::KdaGateOut:    return "kda_gate_out";
        case K3Op::MlaAbsorbQ:    return "mla_absorb_q";
        case K3Op::MlaDecodeAttn: return "mla_decode_attn";
        case K3Op::MlaGateOut:    return "mla_gate_out";
        case K3Op::AttnResMix:    return "attn_res_mix";
        case K3Op::MoeRouter:     return "moe_router_noaux_tc";
        case K3Op::MoeDispatch:   return "moe_expert_ffn_iq2xs";
    }
    return "?";
}

bool k3_op_has_dedicated_kernel(K3Op op) {
    switch (op) {
        // Backed by a validated kernel in kernels/csrc/cuda/kimi_k3/k3_kernels.cu.
        case K3Op::Situ:
        case K3Op::KdaConvStep:
        case K3Op::KdaDecayGate:
        case K3Op::L2NormHeads:
        case K3Op::KdaDecodeStep:
        case K3Op::KdaGateOut:
        case K3Op::MlaAbsorbQ:
        case K3Op::MlaDecodeAttn:
        case K3Op::MlaGateOut:
        case K3Op::AttnResMix:
        case K3Op::MoeRouter:
        case K3Op::MoeDispatch:
            return true;
        // Generic / shared with the existing runtime, or not a kernel at all.
        case K3Op::EmbedLookup:
        case K3Op::RmsNorm:
        case K3Op::MatMul:
        case K3Op::AddResidual:
        case K3Op::AllReduce:
            return false;
    }
    return false;
}

std::vector<std::string> K3DecodePlan::tensors() const {
    std::vector<std::string> out;
    out.reserve(steps.size());
    for (const auto& s : steps)
        if (!s.tensor.empty()) out.push_back(s.tensor);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

namespace {

std::string blk(int i, const char* suffix) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "blk.%d.%s", i, suffix);
    return std::string(buf);
}

struct Builder {
    K3DecodePlan plan;
    const KimiK3Config& cfg;

    explicit Builder(const KimiK3Config& c) : cfg(c) {}

    // A step consuming a GGUF tensor: the rule comes from weight_plan, never from a
    // guess here, so there is exactly one table deciding placement.
    //
    // `pinned` controls whether the declared widths are ASSERTED against the file by
    // k3_validate_plan_shapes(). Pin a width when it is derived from a config value
    // that itself came out of the GGUF (hidden, n_q_heads, kv_lora_rank, expert_latent
    // …) — then a mismatch is a real finding. Leave it unpinned when the tensor's
    // internal structure was inferred from the kernel signatures rather than read from
    // a file, so it is reported as unknown instead of asserted as fact.
    void mm(int layer, bool kda, const char* label, const std::string& tensor,
            int in_dim, int out_dim, const char* note = "", bool pinned = true) {
        K3Step s;
        s.op = K3Op::MatMul;
        s.label = label;
        s.layer = layer;
        s.is_kda_layer = kda;
        s.in_dim = in_dim;
        s.out_dim = out_dim;
        s.tensor = tensor;
        s.rule = tp::rule_for(tensor);
        s.needs_reduce = tp::rule_needs_reduce(s.rule);
        s.note = note;
        if (pinned) { s.expect_ne0 = in_dim; s.expect_ne1 = out_dim; }
        plan.steps.push_back(s);
    }

    // Same, but the widths are inferred and must not be asserted. See mm().
    void mm_unpinned(int layer, bool kda, const char* label, const std::string& tensor,
                     int in_dim, int out_dim, const char* note = "") {
        mm(layer, kda, label, tensor, in_dim, out_dim, note, /*pinned=*/false);
    }

    // A 1-D weight (norm scale, bias): only ne0 is meaningful.
    void vec(K3Op o, int layer, bool kda, const char* label, int width,
             const std::string& tensor, const char* note = "", bool pinned = true) {
        K3Step s;
        s.op = o;
        s.label = label;
        s.layer = layer;
        s.is_kda_layer = kda;
        s.in_dim = width;
        s.out_dim = width;
        s.tensor = tensor;
        s.rule = tensor.empty() ? tp::Rule::Replicate : tp::rule_for(tensor);
        s.note = note;
        if (pinned) s.expect_ne0 = width;
        plan.steps.push_back(s);
    }

    void op(K3Op o, int layer, bool kda, const char* label, int in_dim, int out_dim,
            const std::string& tensor = "", const char* note = "") {
        K3Step s;
        s.op = o;
        s.label = label;
        s.layer = layer;
        s.is_kda_layer = kda;
        s.in_dim = in_dim;
        s.out_dim = out_dim;
        s.tensor = tensor;
        s.rule = tensor.empty() ? tp::Rule::Replicate : tp::rule_for(tensor);
        s.note = note;
        plan.steps.push_back(s);
    }

    // Pin the dims of the step just pushed. Used where a shape was read off the real
    // file rather than derived, so it belongs in the assertion set.
    void pin3(int ne0, int ne1, int ne2) {
        if (plan.steps.empty()) return;
        K3Step& s = plan.steps.back();
        s.expect_ne0 = ne0; s.expect_ne1 = ne1; s.expect_ne2 = ne2;
    }
    void pin1(int ne0) {
        if (plan.steps.empty()) return;
        plan.steps.back().expect_ne0 = ne0;
    }

    void reduce(int layer, bool kda, const char* label, int width) {
        K3Step s;
        s.op = K3Op::AllReduce;
        s.label = label;
        s.layer = layer;
        s.is_kda_layer = kda;
        s.in_dim = width;
        s.out_dim = width;
        s.needs_reduce = true;
        plan.steps.push_back(s);
        ++plan.n_reduces;
    }

    // ---- KDA attention block -------------------------------------------------
    // Order follows the kernel signatures in kimi_k3.h: project, short conv (silu
    // AFTER the conv, not before — asserted by kimi_k3_numeric_test), per-head L2 on
    // q/k, per-channel decay, gated delta rule, norm, output gate, then Wo.
    void kda_block(int i, const K3PlanOptions&) {
        const int H = cfg.hidden;
        const int qkv = cfg.n_q_heads * cfg.kda_head_dim;

        vec(K3Op::RmsNorm, i, true, "attn_norm", H, blk(i, "attn_norm.weight"));
        mm(i, true, "q_proj", blk(i, "attn_q.weight"), H, qkv);
        mm(i, true, "k_proj", blk(i, "attn_k.weight"), H, qkv);
        mm(i, true, "v_proj", blk(i, "attn_v.weight"), H, qkv);

        op(K3Op::KdaConvStep, i, true, "conv_q", qkv, qkv, blk(i, "ssm_conv1d_q.weight"),
           "depthwise causal, silu after");
        op(K3Op::KdaConvStep, i, true, "conv_k", qkv, qkv, blk(i, "ssm_conv1d_k.weight"));
        op(K3Op::KdaConvStep, i, true, "conv_v", qkv, qkv, blk(i, "ssm_conv1d_v.weight"));

        op(K3Op::L2NormHeads, i, true, "l2_qk", qkv, qkv, "", "per-head L2 on q and k");

        // Decay gate. PINNED from the reference loader's create_tensor() calls
        // (unslothai/llama.cpp src/models/kimi-k3.cpp), which are ground truth for
        // what the file must contain — if a real GGUF disagreed, llama.cpp itself
        // would fail to load it. ssm_a's shape is independently confirmed from the
        // real UD-IQ1_S file (96x1x1); the rest are not yet cross-checked against
        // file bytes, only against the loader source, so treat them as high-
        // confidence rather than file-verified until k3_validate_plan_shapes() runs
        // against real weights.
        //
        // f_a/f_b IS A LOW-RANK BOTTLENECK THROUGH head_dim, NOT qkv. An earlier
        // version of this plan assumed f_a projected hidden -> qkv directly (as if
        // decay were computed per-head-group at full width); the reference shows a
        // single [n_embd, head_dim] projection shared before f_b fans it out to
        // d_inner. Getting this backwards would silently run f_a/f_b at 96x the
        // intended compute with a shape that still happens to chain (H -> qkv ->
        // qkv), which is exactly the "well-formed activation, wrong number" trap
        // this schedule exists to catch — and did, once the real shapes were read.
        mm(i, true, "f_a", blk(i, "ssm_f_a.weight"), H, cfg.kda_head_dim);
        mm(i, true, "f_b", blk(i, "ssm_f_b.weight"), cfg.kda_head_dim, qkv);
        // Bias add BEFORE the decay gate, not after: kda_decay_gate_f32's contract is
        // that g_raw already includes +dt_bias (see the kernel header doc). An earlier
        // version of this schedule listed dt_bias as the step AFTER decay_gate, which
        // is backwards from the dataflow even though the kernel itself takes g_raw as
        // an opaque input and was never wrong.
        vec(K3Op::MatMul, i, true, "dt_bias", qkv, blk(i, "ssm_dt.bias"));
        op(K3Op::KdaDecayGate, i, true, "decay_gate", qkv, qkv, blk(i, "ssm_a"),
           "A (ssm_a) is PER-HEAD [n_head]; g_raw (f_b(f_a(x))+dt_bias) is "
           "PER-CHANNEL [d_inner]. The gate's output is per-channel because g_raw is, "
           "even though A broadcasts one scalar across every channel in its head.");
        pin1(cfg.n_q_heads);   // ssm_a is 1-D over heads — confirmed from the real file
        mm(i, true, "beta", blk(i, "ssm_beta.weight"), H, cfg.n_q_heads);

        op(K3Op::KdaDecodeStep, i, true, "delta_rule", qkv, qkv, "",
           "gated delta rule, recurrent state update");
        op(K3Op::RmsNorm, i, true, "ssm_norm", qkv, qkv, blk(i, "ssm_norm.weight"));  // unpinned
        mm(i, true, "g_proj", blk(i, "ssm_g.weight"), H, qkv);
        op(K3Op::KdaGateOut, i, true, "gate_out", qkv, qkv);

        mm(i, true, "attn_out", blk(i, "attn_output.weight"), qkv, H);
        reduce(i, true, "reduce_attn", H);
    }

    // ---- MLA attention block -------------------------------------------------
    // NoPE-only at runtime: rope_dim is present in the GGUF but the arch does not
    // apply it. kv is a single latent (MQA), absorbed into q before the dot product.
    void mla_block(int i, const K3PlanOptions& opt) {
        const int H = cfg.hidden;
        const int qh = cfg.n_q_heads;

        vec(K3Op::RmsNorm, i, false, "attn_norm", H, blk(i, "attn_norm.weight"));

        if (opt.has_q_lora) {
            mm(i, false, "q_a", blk(i, "attn_q_a.weight"), H, cfg.q_lora_rank);
            vec(K3Op::RmsNorm, i, false, "q_a_norm", cfg.q_lora_rank,
                blk(i, "attn_q_a_norm.weight"));
            mm(i, false, "q_b", blk(i, "attn_q_b.weight"), cfg.q_lora_rank,
               qh * cfg.key_length_mla);
        } else {
            mm(i, false, "q_proj", blk(i, "attn_q.weight"), H, qh * cfg.key_length_mla);
        }

        // The latent kv: one head's worth, read by every q head. Replicated, not
        // sharded — see weight_plan's note on attn_kv_a_mqa.
        mm(i, false, "kv_a_mqa", blk(i, "attn_kv_a_mqa.weight"), H, cfg.key_length);
        vec(K3Op::RmsNorm, i, false, "kv_a_norm", cfg.kv_lora_rank,
            blk(i, "attn_kv_a_norm.weight"));

        if (opt.has_fused_kv_b) {
            mm(i, false, "kv_b", blk(i, "attn_kv_b.weight"), cfg.kv_lora_rank,
               qh * (cfg.key_length_mla + cfg.value_length_mla));
        } else {
            mm(i, false, "k_b", blk(i, "attn_k_b.weight"), cfg.kv_lora_rank,
               qh * cfg.key_length_mla);
            mm(i, false, "v_b", blk(i, "attn_v_b.weight"), cfg.kv_lora_rank,
               qh * cfg.value_length_mla);
        }

        op(K3Op::MlaAbsorbQ, i, false, "absorb_q", qh * cfg.key_length_mla,
           qh * cfg.kv_lora_rank, "", "fold k_b into q so attention runs in latent space");
        op(K3Op::MlaDecodeAttn, i, false, "decode_attn", qh * cfg.kv_lora_rank,
           qh * cfg.value_length_mla, "", "NoPE-only");

        if (opt.has_attn_gate) {
            mm(i, false, "attn_gate", blk(i, "attn_gate.weight"), H,
               qh * cfg.value_length_mla);
        }
        op(K3Op::MlaGateOut, i, false, "gate_out", qh * cfg.value_length_mla,
           qh * cfg.value_length_mla, "", "sigmoid gate");

        mm(i, false, "attn_out", blk(i, "attn_output.weight"),
           qh * cfg.value_length_mla, H);
        reduce(i, false, "reduce_attn", H);
    }

    // ---- FFN / MoE block -----------------------------------------------------
    void ffn_block(int i, bool kda, const K3PlanOptions& opt) {
        const int H = cfg.hidden;
        vec(K3Op::RmsNorm, i, kda, "ffn_norm", H, blk(i, "ffn_norm.weight"));

        if (i < cfg.leading_dense) {
            mm(i, kda, "ffn_gate", blk(i, "ffn_gate.weight"), H, cfg.dense_ffn);
            mm(i, kda, "ffn_up", blk(i, "ffn_up.weight"), H, cfg.dense_ffn);
            op(K3Op::Situ, i, kda, "situ", cfg.dense_ffn, cfg.dense_ffn);
            mm(i, kda, "ffn_down", blk(i, "ffn_down.weight"), cfg.dense_ffn, H);
            reduce(i, kda, "reduce_ffn", H);
            return;
        }

        // Router scores at HIDDEN width: it must see the whole vector to rank experts,
        // which is why it is replicated rather than sharded.
        mm(i, kda, "router", blk(i, "ffn_gate_inp.weight"), H, cfg.n_experts);
        op(K3Op::MoeRouter, i, kda, "route_noaux_tc", cfg.n_experts, cfg.top_k,
           blk(i, "exp_probs_b.bias"),
           "expert bias biases SELECTION only, never the returned weights");
        pin1(cfg.n_experts);   // exp_probs_b.bias is 1-D over experts

        // THE LATENT HOP. Routed experts do not live at hidden width.
        mm(i, kda, "routed_down", blk(i, "ffn_routed_down.weight"), H, cfg.expert_latent);
        if (opt.has_routed_norm) {
            op(K3Op::RmsNorm, i, kda, "routed_norm", cfg.expert_latent, cfg.expert_latent,
               blk(i, "ffn_routed_norm.weight"));
        }

        // Pinned from a direct read of the real model (blk.1): gate/up are
        // [3584, 3072, 896] and down is [3072, 3584, 896], all IQ2_XS.
        op(K3Op::MoeDispatch, i, kda, "expert_ffn", cfg.expert_latent, cfg.expert_latent,
           blk(i, "ffn_gate_exps.weight"),
           "gate/up -> situ at moe_ffn -> down, all in latent space");
        pin3(cfg.expert_latent, cfg.moe_ffn, cfg.n_experts);
        // The up/down expert tensors are consumed by the same dispatch kernel; listed
        // so the manifest cross-check sees them.
        op(K3Op::MoeDispatch, i, kda, "expert_ffn_up", cfg.expert_latent, cfg.moe_ffn,
           blk(i, "ffn_up_exps.weight"));
        pin3(cfg.expert_latent, cfg.moe_ffn, cfg.n_experts);
        op(K3Op::MoeDispatch, i, kda, "expert_ffn_down", cfg.moe_ffn, cfg.expert_latent,
           blk(i, "ffn_down_exps.weight"));
        pin3(cfg.moe_ffn, cfg.expert_latent, cfg.n_experts);

        if (opt.has_shared_experts) {
            mm(i, kda, "shexp_gate", blk(i, "ffn_gate_shexp.weight"), H, cfg.moe_ffn);
            mm(i, kda, "shexp_up", blk(i, "ffn_up_shexp.weight"), H, cfg.moe_ffn);
            op(K3Op::Situ, i, kda, "shexp_situ", cfg.moe_ffn, cfg.moe_ffn);
            mm(i, kda, "shexp_down", blk(i, "ffn_down_shexp.weight"), cfg.moe_ffn, H);
        }

        mm(i, kda, "routed_up", blk(i, "ffn_routed_up.weight"), cfg.expert_latent, H);
        reduce(i, kda, "reduce_ffn", H);
    }

    void build(const K3PlanOptions& opt) {
        const int H = cfg.hidden;
        op(K3Op::EmbedLookup, -1, false, "embed", 1, H, "token_embd.weight");

        for (int i = 0; i < cfg.n_layers; ++i) {
            const bool kda = cfg.is_kda_layer(i);
            if (kda) { ++plan.n_kda_layers; kda_block(i, opt); }
            else     { ++plan.n_mla_layers; mla_block(i, opt); }

            // Cross-layer residual attention: a learned mix over the last block_size
            // layers' residuals, scored per layer. Runs AFTER the reduce, on the
            // full-width tensor.
            if (cfg.attn_res_block_size > 0) {
                op(K3Op::MatMul, i, kda, "attn_res_score", H, cfg.attn_res_block_size,
                   blk(i, "attn_res_score.weight"));
                op(K3Op::AttnResMix, i, kda, "attn_res_mix", H, H, "",
                   "raw residuals, NOT renormalised");
            }
            op(K3Op::AddResidual, i, kda, "add_attn", H, H);

            ffn_block(i, kda, opt);
            if (cfg.attn_res_block_size > 0) {
                op(K3Op::MatMul, i, kda, "ffn_res_score", H, cfg.attn_res_block_size,
                   blk(i, "ffn_res_score.weight"));
                op(K3Op::AttnResMix, i, kda, "ffn_res_mix", H, H);
            }
            op(K3Op::AddResidual, i, kda, "add_ffn", H, H);
        }

        op(K3Op::RmsNorm, -1, false, "output_norm", H, H, "output_norm.weight");
        if (cfg.attn_res_block_size > 0)
            op(K3Op::MatMul, -1, false, "output_res_score", H, cfg.attn_res_block_size,
               "output_res_score.weight");
        op(K3Op::MatMul, -1, false, "lm_head", H, cfg.vocab, "output.weight");
    }
};

}  // namespace

K3DecodePlan build_k3_decode_plan(const KimiK3Config& cfg, const tp::ShardDims& d,
                                  const K3PlanOptions& opt) {
    (void)d;   // TP changes placement, never the op sequence — see the header.
    Builder b(cfg);
    b.build(opt);
    return b.plan;
}


bool k3_validate_plan_shapes(const K3DecodePlan& plan, const GGUF& g,
                             K3PlanShapeReport* report) {
    K3PlanShapeReport local;
    K3PlanShapeReport& r = report ? *report : local;

    // One entry per distinct tensor: the plan names each many times (93 layers) and a
    // per-step report would be 2000 duplicate lines.
    std::map<std::string, const K3Step*> first_use;
    for (const auto& s : plan.steps)
        if (!s.tensor.empty()) first_use.emplace(s.tensor, &s);

    auto push = [&r](std::vector<std::string>& v, const std::string& msg) {
        if (v.size() < 40) v.push_back(msg);
    };

    for (const auto& kv : first_use) {
        const std::string& name = kv.first;
        const K3Step& s = *kv.second;
        const GGUFTensor* t = g.tensor(name);
        if (!t) {
            ++r.missing;
            push(r.problems, name + ": absent from the GGUF but consumed by step '" +
                                 std::string(s.label) + "'");
            continue;
        }

        char shape[96];
        std::snprintf(shape, sizeof(shape), "%ldx%ldx%ld (type %d)",
                      t->dims[0], t->dims[1], t->dims[2], t->ggml_type);

        if (s.expect_ne0 == 0 && s.expect_ne1 == 0 && s.expect_ne2 == 0) {
            ++r.unpinned;
            push(r.unpinned_shapes, name + " [" + s.label + "] = " + shape);
            continue;
        }

        ++r.checked;
        const long want[3] = {s.expect_ne0, s.expect_ne1, s.expect_ne2};
        for (int ax = 0; ax < 3; ++ax) {
            if (want[ax] == 0) continue;             // that axis is not asserted
            if (t->dims[ax] == want[ax]) continue;
            ++r.mismatched;
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "%s [%s]: ne%d is %ld, plan expects %ld  (actual %s)",
                          name.c_str(), s.label, ax, t->dims[ax], want[ax], shape);
            push(r.problems, buf);
            break;   // one line per tensor is enough to act on
        }
    }

    if (r.mismatched || r.missing) {
        std::fprintf(stderr,
                     "[kimi-k3] plan/GGUF shape check: %d mismatched, %d missing, "
                     "%d checked, %d unpinned\n",
                     r.mismatched, r.missing, r.checked, r.unpinned);
        for (const auto& p : r.problems) std::fprintf(stderr, "  %s\n", p.c_str());
        return false;
    }
    return true;
}

std::string k3_plan_layer_dump(const K3DecodePlan& plan, int layer) {
    std::ostringstream os;
    for (const auto& s : plan.steps) {
        if (s.layer != layer) continue;
        os << "  " << (s.is_kda_layer ? "KDA " : "MLA ")
           << k3_op_name(s.op) << " [" << s.label << "] "
           << s.in_dim << " -> " << s.out_dim;
        if (!s.tensor.empty())
            os << "  " << s.tensor << " (" << tp::rule_name(s.rule) << ")";
        if (s.needs_reduce && s.op != K3Op::AllReduce) os << "  +reduce";
        if (!s.note.empty()) os << "\n       # " << s.note;
        os << "\n";
    }
    return os.str();
}

}  // namespace sparkinfer
