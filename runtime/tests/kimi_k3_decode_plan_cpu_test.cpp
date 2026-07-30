// The K3 decode schedule. CPU-only: building the plan launches nothing.
//
// The check that carries the most weight here is the PLAN/MANIFEST CROSS-CHECK. The
// manifest (kimi_k3_gguf_manifest.h) lists what the reference loader requires; the
// plan lists what the forward pass consumes. They were written from different sources
// — the reference loader vs the kernel signatures — so if they name the same tensor
// set, that is evidence. A tensor in the manifest that no step consumes is a weight
// being uploaded and never used; a tensor a step consumes that the manifest does not
// require is a load that will succeed and then read a null pointer.
//
// The rest are the properties that would otherwise surface only as bad output:
// widths chain, layer types use their own ops, and the reduce count is exactly
// 2 per layer.

#include "sparkinfer/models/kimi_k3_decode_plan.h"
#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/tp/shard.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace sparkinfer;

static int g_fail = 0, g_checks = 0;
static void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}
static void check_eq(long got, long want, const std::string& what) {
    ++g_checks;
    if (got != want) {
        ++g_fail;
        std::printf("  FAIL  %s: got %ld want %ld\n", what.c_str(), got, want);
    }
}

// The real K3 layer map: 93 layers, 24 MLA and 69 KDA. A layer is KDA when the GGUF's
// per-layer head_count_kv entry is 0. Build a representative interleave — the plan
// must not care about the pattern, only about each layer's own type.
static KimiK3Config k3_config() {
    KimiK3Config cfg;
    cfg.layer_is_kda.assign(cfg.n_layers, 1);
    // Every 4th layer full-attention, giving 24 MLA over 93 layers.
    int mla = 0;
    for (int i = 0; i < cfg.n_layers && mla < 24; i += 4) { cfg.layer_is_kda[i] = 0; ++mla; }
    return cfg;
}

static tp::ShardDims dims8() {
    tp::ShardDims d;
    d.tp_size = 8; d.rank = 0; d.hidden = 7168;
    d.n_experts_total = 896; d.n_experts = 112; d.experts_sharded = true;
    d.expert_band = tp::even_band(896, 8, 0);
    return d;
}

// Mirror of kimi_k3_gguf_manifest.h's required set, as a name list. Kept as an
// independent transcription ON PURPOSE: including the manifest header and calling it
// would need a real GGUF, and re-deriving the set from the same code it is being
// compared against would make the cross-check circular.
static std::set<std::string> manifest_required(const KimiK3Config& cfg,
                                               bool q_lora, bool split_kv_b) {
    std::set<std::string> s;
    auto blk = [](int i, const char* suf) {
        char b[96]; std::snprintf(b, sizeof(b), "blk.%d.%s", i, suf); return std::string(b);
    };
    s.insert("token_embd.weight");
    s.insert("output_norm.weight");
    s.insert("output.weight");
    if (cfg.attn_res_block_size > 0) s.insert("output_res_score.weight");

    for (int i = 0; i < cfg.n_layers; ++i) {
        s.insert(blk(i, "attn_norm.weight"));
        s.insert(blk(i, "ffn_norm.weight"));
        if (cfg.attn_res_block_size > 0) {
            s.insert(blk(i, "attn_res_score.weight"));
            s.insert(blk(i, "ffn_res_score.weight"));
        }
        if (cfg.is_kda_layer(i)) {
            s.insert(blk(i, "ssm_conv1d_q.weight"));
            s.insert(blk(i, "ssm_conv1d_k.weight"));
            s.insert(blk(i, "ssm_conv1d_v.weight"));
            s.insert(blk(i, "attn_q.weight"));
            s.insert(blk(i, "attn_k.weight"));
            s.insert(blk(i, "attn_v.weight"));
            s.insert(blk(i, "ssm_f_a.weight"));
            s.insert(blk(i, "ssm_f_b.weight"));
            s.insert(blk(i, "ssm_beta.weight"));
            s.insert(blk(i, "ssm_a"));            // no .weight suffix upstream
            s.insert(blk(i, "ssm_dt.bias"));
            s.insert(blk(i, "ssm_g.weight"));
            s.insert(blk(i, "ssm_norm.weight"));
            s.insert(blk(i, "attn_output.weight"));
        } else {
            s.insert(blk(i, "attn_kv_a_norm.weight"));
            if (q_lora) {
                s.insert(blk(i, "attn_q_a.weight"));
                s.insert(blk(i, "attn_q_a_norm.weight"));
                s.insert(blk(i, "attn_q_b.weight"));
            } else {
                s.insert(blk(i, "attn_q.weight"));
            }
            s.insert(blk(i, "attn_kv_a_mqa.weight"));
            if (split_kv_b) {
                s.insert(blk(i, "attn_k_b.weight"));
                s.insert(blk(i, "attn_v_b.weight"));
            } else {
                s.insert(blk(i, "attn_kv_b.weight"));
            }
            s.insert(blk(i, "attn_output.weight"));
        }
        if (i < cfg.leading_dense) {
            s.insert(blk(i, "ffn_gate.weight"));
            s.insert(blk(i, "ffn_up.weight"));
            s.insert(blk(i, "ffn_down.weight"));
        } else {
            s.insert(blk(i, "ffn_gate_inp.weight"));
            s.insert(blk(i, "exp_probs_b.bias"));
            s.insert(blk(i, "ffn_gate_exps.weight"));
            s.insert(blk(i, "ffn_up_exps.weight"));
            s.insert(blk(i, "ffn_down_exps.weight"));
            if (cfg.expert_latent > 0) {
                s.insert(blk(i, "ffn_routed_down.weight"));
                s.insert(blk(i, "ffn_routed_up.weight"));
            }
        }
    }
    return s;
}

int main() {
    std::printf("=== Kimi K3 decode plan ===\n\n");
    const KimiK3Config cfg = k3_config();
    const tp::ShardDims d = dims8();

    K3PlanOptions opt;
    opt.has_q_lora = true;
    opt.has_fused_kv_b = false;   // K3 ships split k_b / v_b
    const K3DecodePlan plan = build_k3_decode_plan(cfg, d, opt);

    std::printf("[shape of the plan]\n");
    check_eq(plan.n_kda_layers + plan.n_mla_layers, cfg.n_layers, "every layer planned");
    check_eq(plan.n_kda_layers, 69, "69 KDA layers");
    check_eq(plan.n_mla_layers, 24, "24 MLA layers");
    std::printf("  %zu steps, %d reduces, %d KDA / %d MLA layers\n",
                plan.steps.size(), plan.n_reduces, plan.n_kda_layers, plan.n_mla_layers);

    // ------------------------------------------------------- collective count
    std::printf("\n[collectives: exactly two per layer]\n");
    {
        check_eq(plan.n_reduces, 2 * cfg.n_layers,
                 "exactly 2 all-reduces per layer (attn + ffn)");
        check_eq(plan.n_reduces, 186, "186 reduces per token at 93 layers");

        // Per layer, and in the right places: one after attn_output, one after the
        // FFN's return to hidden width.
        std::map<int, int> per_layer;
        for (const auto& s : plan.steps)
            if (s.op == K3Op::AllReduce) ++per_layer[s.layer];
        bool all_two = true;
        for (int i = 0; i < cfg.n_layers; ++i) if (per_layer[i] != 2) all_two = false;
        check(all_two, "every layer has exactly 2 reduces");

        // Every ColShard consumer must be followed by a reduce, and nothing else may
        // be. This is the check that catches a missing OR an extra collective.
        int colshard_steps = 0;
        for (size_t k = 0; k < plan.steps.size(); ++k) {
            const auto& s = plan.steps[k];
            if (s.rule != tp::Rule::ColShard || s.op == K3Op::AllReduce) continue;
            ++colshard_steps;
            // Find the next AllReduce in the same layer; nothing consuming the tensor
            // may intervene. Adjacency is the invariant the executor relies on.
            bool followed = false;
            for (size_t j = k + 1; j < plan.steps.size(); ++j) {
                if (plan.steps[j].op == K3Op::AllReduce) { followed = true; break; }
                if (plan.steps[j].layer != s.layer) break;
                // A norm or matmul on the partial sum before reducing is wrong.
                if (plan.steps[j].op == K3Op::MatMul || plan.steps[j].op == K3Op::RmsNorm)
                    break;
            }
            check(followed, std::string("ColShard '") + s.tensor +
                                "' is followed by an all-reduce before anything consumes it");
        }
        check(colshard_steps >= cfg.n_layers,
              "at least one ColShard per layer (attn_output)");
        std::printf("  %d ColShard consumers, each reduce-adjacent\n", colshard_steps);

        // THE EXPERT-PARALLEL PARTIAL SUM. ColShard is not the only rule that leaves a
        // rank holding a fraction of the answer: the routed experts are ExpertShard, so
        // rank r's dispatch accumulator covers only the selected experts r owns.
        //
        // rule_needs_reduce() cannot express this — ExpertShard is also used for
        // attn_k_b/attn_v_b, which band the HEAD axis, and heads are concatenated
        // rather than summed, so those genuinely need no collective. The distinction is
        // the OP, not the rule, which is why it is asserted here on MoeDispatch.
        //
        // This is the check that would have caught the plan reducing after routed_up
        // instead of after the dispatch: routed_norm is an rms_norm and rms_norm is not
        // linear, so normalising a partial sum and reducing afterwards is wrong, and
        // wrong in a way that stays fluent.
        int dispatch_layers = 0;
        for (size_t k = 0; k < plan.steps.size(); ++k) {
            const auto& s = plan.steps[k];
            if (s.op != K3Op::MoeDispatch) continue;
            // Only assert once per layer, on the LAST dispatch step of that layer.
            if (k + 1 < plan.steps.size() && plan.steps[k + 1].op == K3Op::MoeDispatch &&
                plan.steps[k + 1].layer == s.layer)
                continue;
            ++dispatch_layers;
            bool followed = false;
            int reduce_width = -1;
            for (size_t j = k + 1; j < plan.steps.size(); ++j) {
                if (plan.steps[j].layer != s.layer) break;
                if (plan.steps[j].op == K3Op::AllReduce) {
                    followed = true;
                    reduce_width = plan.steps[j].in_dim;
                    break;
                }
                // A norm or a projection consuming the PARTIAL expert sum is the bug.
                if (plan.steps[j].op == K3Op::MatMul || plan.steps[j].op == K3Op::RmsNorm)
                    break;
            }
            check(followed,
                  "MoE dispatch is followed by an all-reduce before routed_norm/routed_up "
                  "consume the partial expert sum");
            // And it must reduce the LATENT vector, not the hidden one — reducing at
            // hidden width means it was placed after routed_up, where every rank
            // already holds the complete tensor and summing multiplies it by tp_size.
            check_eq(reduce_width, cfg.expert_latent,
                     "the MoE collective is at expert_latent width, not hidden");
        }
        check_eq(dispatch_layers, cfg.n_layers - cfg.leading_dense,
                 "every MoE layer's dispatch is reduce-covered");

        // Nothing may reduce AFTER routed_up: by then routed_norm/routed_up/shexp are
        // all Replicate, so the tensor is already complete and identical on every rank.
        for (size_t k = 0; k < plan.steps.size(); ++k) {
            if (plan.steps[k].op != K3Op::MatMul) continue;
            if (std::string(plan.steps[k].label) != "routed_up") continue;
            bool spurious = false;
            for (size_t j = k + 1; j < plan.steps.size(); ++j) {
                if (plan.steps[j].layer != plan.steps[k].layer) break;
                if (plan.steps[j].op == K3Op::AllReduce) { spurious = true; break; }
            }
            check(!spurious,
                  "no all-reduce after routed_up (it would multiply the FFN output by tp_size)");
        }
        std::printf("  %d MoE dispatches, each reduced at latent width %d\n",
                    dispatch_layers, cfg.expert_latent);
    }

    // ------------------------------------------------------------ width flow
    std::printf("\n[widths: the latent MoE hop is the trap]\n");
    {
        // The routed experts live at expert_latent, not hidden. Assert the full hop
        // for a MoE layer: 7168 -> 3584 -> (3072) -> 3584 -> 7168.
        const int L = 5;   // a KDA MoE layer in this interleave
        check(cfg.is_kda_layer(L), "layer 5 is KDA in this map");
        int seen_down = 0, seen_up = 0, seen_disp = 0;
        for (const auto& s : plan.steps) {
            if (s.layer != L) continue;
            if (s.tensor == "blk.5.ffn_routed_down.weight") {
                ++seen_down;
                check_eq(s.in_dim, cfg.hidden, "routed_down consumes hidden");
                check_eq(s.out_dim, cfg.expert_latent, "routed_down produces expert_latent");
            }
            if (s.tensor == "blk.5.ffn_routed_up.weight") {
                ++seen_up;
                check_eq(s.in_dim, cfg.expert_latent, "routed_up consumes expert_latent");
                check_eq(s.out_dim, cfg.hidden, "routed_up produces hidden");
            }
            if (s.op == K3Op::MoeDispatch && s.tensor == "blk.5.ffn_gate_exps.weight") {
                ++seen_disp;
                check_eq(s.in_dim, cfg.expert_latent,
                         "expert dispatch runs at expert_latent, NOT hidden");
            }
            if (s.tensor == "blk.5.ffn_down_exps.weight")
                check_eq(s.in_dim, cfg.moe_ffn, "expert down consumes moe_ffn");
        }
        check_eq(seen_down, 1, "one routed_down per MoE layer");
        check_eq(seen_up, 1, "one routed_up per MoE layer");
        check_eq(seen_disp, 1, "one expert dispatch per MoE layer");

        // The router scores at HIDDEN width and returns top_k, not a width.
        for (const auto& s : plan.steps) {
            if (s.layer == L && s.op == K3Op::MoeRouter) {
                check_eq(s.in_dim, cfg.n_experts, "router scores all experts");
                check_eq(s.out_dim, cfg.top_k, "router returns top_k");
            }
            if (s.layer == L && s.tensor == "blk.5.ffn_gate_inp.weight") {
                check_eq(s.in_dim, cfg.hidden,
                         "router weight consumes the whole hidden vector");
                check(s.rule == tp::Rule::Replicate,
                      "router is replicated — it must see all of hidden");
            }
        }

        // The leading dense layer is the ONLY one at dense_ffn width.
        int dense_layers = 0;
        for (const auto& s : plan.steps)
            if (s.tensor.find("ffn_down.weight") != std::string::npos) {
                ++dense_layers;
                check_eq(s.in_dim, cfg.dense_ffn, "dense ffn_down consumes dense_ffn");
                check(s.layer < cfg.leading_dense, "dense FFN only in the leading block");
            }
        check_eq(dense_layers, cfg.leading_dense, "exactly leading_dense dense FFNs");
    }

    // ------------------------------------------------------- layer-type purity
    std::printf("\n[layer types must not borrow each other's ops]\n");
    {
        const std::set<K3Op> kda_only = {K3Op::KdaConvStep, K3Op::KdaDecayGate,
                                         K3Op::KdaDecodeStep, K3Op::KdaGateOut};
        const std::set<K3Op> mla_only = {K3Op::MlaAbsorbQ, K3Op::MlaDecodeAttn,
                                         K3Op::MlaGateOut};
        bool ok = true;
        for (const auto& s : plan.steps) {
            if (s.layer < 0) continue;
            const bool kda = cfg.is_kda_layer(s.layer);
            if (kda_only.count(s.op) && !kda) ok = false;
            if (mla_only.count(s.op) && kda) ok = false;
            check_eq(s.is_kda_layer ? 1 : 0, kda ? 1 : 0,
                     "step's layer-type flag matches the config");
        }
        check(ok, "no KDA op in an MLA layer and no MLA op in a KDA layer");

        // Each type's signature op appears exactly once per layer of that type.
        int deltas = 0, attns = 0;
        for (const auto& s : plan.steps) {
            if (s.op == K3Op::KdaDecodeStep) ++deltas;
            if (s.op == K3Op::MlaDecodeAttn) ++attns;
        }
        check_eq(deltas, 69, "one gated-delta-rule step per KDA layer");
        check_eq(attns, 24, "one MLA attention step per MLA layer");
    }

    // ------------------------------------------------------ kernel coverage
    std::printf("\n[kernel coverage: every kernel written is a kernel reached]\n");
    {
        std::set<K3Op> reached;
        for (const auto& s : plan.steps)
            if (k3_op_has_dedicated_kernel(s.op)) reached.insert(s.op);

        const std::vector<K3Op> all_kernels = {
            K3Op::Situ, K3Op::KdaConvStep, K3Op::KdaDecayGate, K3Op::L2NormHeads,
            K3Op::KdaDecodeStep, K3Op::KdaGateOut, K3Op::MlaAbsorbQ,
            K3Op::MlaDecodeAttn, K3Op::MlaGateOut, K3Op::AttnResMix,
            K3Op::MoeRouter, K3Op::MoeDispatch,
        };
        for (K3Op op : all_kernels)
            check(reached.count(op) == 1,
                  std::string("kernel '") + k3_op_name(op) + "' is reached by the plan");
        check_eq((long)reached.size(), (long)all_kernels.size(),
                 "no dedicated kernel is unreachable (dead code)");
        std::printf("  %zu/%zu dedicated kernels reached\n", reached.size(),
                    all_kernels.size());
    }

    // -------------------------------------------- plan vs manifest cross-check
    std::printf("\n[plan vs manifest: two independently written tensor sets]\n");
    {
        const std::set<std::string> required =
            manifest_required(cfg, opt.has_q_lora, !opt.has_fused_kv_b);
        const std::vector<std::string> consumed_v = plan.tensors();
        const std::set<std::string> consumed(consumed_v.begin(), consumed_v.end());

        std::vector<std::string> loaded_never_used, used_never_loaded;
        for (const auto& n : required)
            if (!consumed.count(n)) loaded_never_used.push_back(n);
        for (const auto& n : consumed)
            if (!required.count(n)) used_never_loaded.push_back(n);

        // A tensor the plan consumes but the manifest does not require would load as a
        // null pointer — the worse direction, so report it loudly.
        for (size_t i = 0; i < used_never_loaded.size() && i < 12; ++i)
            std::printf("    consumed but NOT required: %s\n", used_never_loaded[i].c_str());
        for (size_t i = 0; i < loaded_never_used.size() && i < 12; ++i)
            std::printf("    required but NOT consumed: %s\n", loaded_never_used[i].c_str());

        check_eq((long)used_never_loaded.size(), 0,
                 "every tensor the plan consumes is one the manifest requires");
        check_eq((long)loaded_never_used.size(), 0,
                 "every tensor the manifest requires is one the plan consumes");
        std::printf("  %zu tensors, both sets agree\n", consumed.size());
    }

    // ------------------------------------------------- pinned vs inferred widths
    std::printf("\n[shape expectations: pin what the file told us, report what it did not]\n");
    {
        std::map<std::string, const K3Step*> first;
        for (const auto& s : plan.steps)
            if (!s.tensor.empty()) first.emplace(s.tensor, &s);

        // Verified by a direct read of the real model, so all three axes are asserted.
        for (const char* n : {"blk.5.ffn_gate_exps.weight", "blk.5.ffn_up_exps.weight",
                              "blk.5.ffn_down_exps.weight"}) {
            auto it = first.find(n);
            check(it != first.end(), std::string(n) + " is in the plan");
            if (it == first.end()) continue;
            check(it->second->expect_ne2 == cfg.n_experts,
                  std::string(n) + " pins the expert axis (read off the real file)");
            check(it->second->expect_ne0 > 0 && it->second->expect_ne1 > 0,
                  std::string(n) + " pins both matrix dims");
        }
        // gate/up are [latent, moe_ffn]; down is the transpose pair.
        check(first["blk.5.ffn_gate_exps.weight"]->expect_ne0 == cfg.expert_latent &&
              first["blk.5.ffn_gate_exps.weight"]->expect_ne1 == cfg.moe_ffn,
              "gate_exps pinned to [expert_latent, moe_ffn, n_experts]");
        check(first["blk.5.ffn_down_exps.weight"]->expect_ne0 == cfg.moe_ffn &&
              first["blk.5.ffn_down_exps.weight"]->expect_ne1 == cfg.expert_latent,
              "down_exps pinned to [moe_ffn, expert_latent, n_experts]");

        // PINNED from the reference loader's create_tensor() calls (kimi-k3.cpp) —
        // ssm_f_a/f_b were WRONG in an earlier version of this plan (see the
        // comment at the build site) until this reading corrected them.
        auto fa = first.find("blk.5.ssm_f_a.weight");
        check(fa != first.end() && fa->second->expect_ne0 == cfg.hidden &&
                  fa->second->expect_ne1 == cfg.kda_head_dim,
              "ssm_f_a pins [hidden, kda_head_dim] — NOT [hidden, qkv]");
        auto fb = first.find("blk.5.ssm_f_b.weight");
        check(fb != first.end() && fb->second->expect_ne0 == cfg.kda_head_dim &&
                  fb->second->expect_ne1 == cfg.n_q_heads * cfg.kda_head_dim,
              "ssm_f_b pins [kda_head_dim, qkv] — the low-rank bottleneck is "
              "kda_head_dim wide, not qkv wide");
        auto beta_t = first.find("blk.5.ssm_beta.weight");
        check(beta_t != first.end() && beta_t->second->expect_ne0 == cfg.hidden &&
                  beta_t->second->expect_ne1 == cfg.n_q_heads,
              "ssm_beta pins [hidden, n_q_heads]");
        auto dtb = first.find("blk.5.ssm_dt.bias");
        check(dtb != first.end() && dtb->second->expect_ne0 == cfg.n_q_heads * cfg.kda_head_dim &&
                  dtb->second->expect_ne1 == 0,
              "ssm_dt.bias pins ne0=d_inner (qkv) as a 1-D bias");
        auto ssma = first.find("blk.5.ssm_a");
        check(ssma != first.end() && ssma->second->expect_ne0 == cfg.n_q_heads &&
                  ssma->second->expect_ne1 == 0,
              "ssm_a pins ne0=n_head — CONFIRMED against the real UD-IQ1_S file (96x1x1)");

        // ssm_norm.weight is [head_dim] (128), NOT [d_inner]/qkv (12288) — an earlier
        // version of this plan pinned it at qkv width AND scheduled a separate
        // RmsNorm step for it, when kda_gate_out_f32 already does rms_norm internally
        // and consumes ssm_norm.weight itself. There is no standalone "ssm_norm" step.
        int idx_ssmnorm_step = -1;
        for (size_t k = 0; k < plan.steps.size(); ++k)
            if (plan.steps[k].layer == 5 && plan.steps[k].op == K3Op::RmsNorm &&
                plan.steps[k].tensor == "blk.5.ssm_norm.weight")
                idx_ssmnorm_step = (int)k;
        check(idx_ssmnorm_step < 0,
              "no standalone RmsNorm step consumes ssm_norm.weight — it is folded "
              "into kda_gate_out");
        auto ssmnorm = first.find("blk.5.ssm_norm.weight");
        check(ssmnorm != first.end() && ssmnorm->second->op == K3Op::KdaGateOut,
              "ssm_norm.weight is consumed by the KdaGateOut step");
        check(ssmnorm != first.end() && ssmnorm->second->expect_ne0 == cfg.kda_head_dim &&
                  ssmnorm->second->expect_ne1 == 0,
              "ssm_norm.weight pins ne0=kda_head_dim (128), not qkv (12288)");

        // wk_b/wv_b: 3-D, head-major, consumed directly by MlaAbsorbQ/MlaDecodeAttn —
        // an earlier version of this plan modeled them as 2-D GEMV projections
        // (mm() calls) that don't correspond to anything in the reference. Layer 0
        // is MLA in this test's synthetic layer map.
        auto kb = first.find("blk.0.attn_k_b.weight");
        const int qk_nope = cfg.key_length_mla - cfg.rope_dim;
        check(kb != first.end() && kb->second->op == K3Op::MlaAbsorbQ,
              "attn_k_b.weight is consumed by MlaAbsorbQ, not a separate MatMul");
        check(kb != first.end() && kb->second->expect_ne0 == qk_nope &&
                  kb->second->expect_ne1 == cfg.kv_lora_rank &&
                  kb->second->expect_ne2 == cfg.n_q_heads,
              "attn_k_b.weight pins [qk_nope_head_dim, kv_lora_rank, n_head]");
        auto vb = first.find("blk.0.attn_v_b.weight");
        check(vb != first.end() && vb->second->op == K3Op::MlaDecodeAttn,
              "attn_v_b.weight is consumed by MlaDecodeAttn, not a separate MatMul");
        check(vb != first.end() && vb->second->expect_ne0 == cfg.kv_lora_rank &&
                  vb->second->expect_ne1 == cfg.value_length_mla &&
                  vb->second->expect_ne2 == cfg.n_q_heads,
              "attn_v_b.weight pins [kv_lora_rank, n_embd_head_v, n_head]");

        // MlaAbsorbQ's OUTPUT (and so MlaDecodeAttn's INPUT) is qh*key_length
        // (kv_lora + rope_dim concatenated per head = 576-wide), not qh*kv_lora_rank
        // (512-wide) — per mla_absorb_q_f32's own doc: "Q[h] = concat(q_nope_absorbed
        // [h], q_pe[h]) // length key_length". Only caught while sizing the actual
        // executor buffers, not by any earlier schedule-level check — worth pinning
        // down explicitly so it can't regress silently.
        int idx_absorb = -1, idx_decode = -1;
        for (size_t k = 0; k < plan.steps.size(); ++k) {
            if (plan.steps[k].layer != 0) continue;
            if (plan.steps[k].op == K3Op::MlaAbsorbQ) idx_absorb = (int)k;
            if (plan.steps[k].op == K3Op::MlaDecodeAttn) idx_decode = (int)k;
        }
        check(idx_absorb >= 0 &&
                  plan.steps[idx_absorb].out_dim == cfg.n_q_heads * cfg.key_length,
              "MlaAbsorbQ out_dim is qh*key_length (kv_lora+rope_dim), not "
              "qh*kv_lora_rank alone");
        check(idx_decode >= 0 &&
                  plan.steps[idx_decode].in_dim == cfg.n_q_heads * cfg.key_length,
              "MlaDecodeAttn in_dim matches MlaAbsorbQ's real output width");

        // dt_bias must precede decay_gate in the STEP ORDER, not just in the tensor
        // set: kda_decay_gate_f32's contract is that g_raw already includes +dt_bias.
        // An earlier version of this plan had the order backwards.
        int idx_dt = -1, idx_gate = -1, idx_fb = -1;
        for (size_t k = 0; k < plan.steps.size(); ++k) {
            if (plan.steps[k].layer != 5) continue;
            if (plan.steps[k].label == std::string("dt_bias")) idx_dt = (int)k;
            if (plan.steps[k].op == K3Op::KdaDecayGate) idx_gate = (int)k;
            if (plan.steps[k].label == std::string("f_b")) idx_fb = (int)k;
        }
        check(idx_fb >= 0 && idx_dt >= 0 && idx_gate >= 0 &&
                  idx_fb < idx_dt && idx_dt < idx_gate,
              "order is f_b -> dt_bias -> decay_gate, matching the reference dataflow");

        // PINNED: GGUFTensor defaults unlisted dims to 1, so the reference's 3-D vs
        // 4-D conv1d storage (trailing 1 squeezed or not) is identical on ne0..ne2 —
        // there was no real ambiguity here, just an earlier overcautious guess.
        auto conv = first.find("blk.5.ssm_conv1d_q.weight");
        check(conv != first.end() && conv->second->expect_ne0 == cfg.kda_conv_kernel &&
                  conv->second->expect_ne1 == 1 &&
                  conv->second->expect_ne2 == cfg.n_q_heads * cfg.kda_head_dim,
              "ssm_conv1d_q pins [kda_conv_kernel, 1, qkv]");

        // A 1-D norm asserts its width and nothing else — pinning ne1 on a 1-D tensor
        // would fail against every real file, where ne1 is 1.
        auto nrm = first.find("blk.5.attn_norm.weight");
        check(nrm != first.end() && nrm->second->expect_ne0 == cfg.hidden &&
                  nrm->second->expect_ne1 == 0,
              "a 1-D norm pins ne0 only");

        // A plain projection derived from GGUF-read config values IS pinned.
        auto q = first.find("blk.5.attn_q.weight");
        check(q != first.end() && q->second->expect_ne0 == cfg.hidden &&
                  q->second->expect_ne1 == cfg.n_q_heads * cfg.kda_head_dim,
              "attn_q pins [hidden, n_q_heads*head_dim]");

        int pinned = 0, unpinned = 0;
        for (const auto& kv : first)
            (kv.second->expect_ne0 ? pinned : unpinned)++;
        std::printf("  %d distinct tensors pinned, %d awaiting a read from real weights\n",
                    pinned, unpinned);
        check(pinned > unpinned, "most tensors are pinned");
    }

    // -------------------------------------- cross-layer residual: order + banking
    // Read directly off unslothai/llama.cpp's graph builder (src/models/kimi-k3.cpp):
    // res_mix runs BEFORE each norm, not after; a checkpointed layer REPLACES the
    // residual stream on the attention side instead of adding; the bank push happens
    // only on the attention side, using the RAW pre-mix value. An earlier version of
    // this plan had all of this backwards (mix after, unconditional add) — plausible
    // and wrong, exactly the class of bug this project is built to catch before it
    // reaches a kernel.
    std::printf("\n[cross-layer residual: pre-norm mix, attention-side replace]\n");
    {
        auto idx_of = [&](int layer, K3Op op, const char* label) -> int {
            for (size_t k = 0; k < plan.steps.size(); ++k) {
                const auto& s = plan.steps[k];
                if (s.layer == layer && s.op == op &&
                    (label == nullptr || s.label == std::string(label)))
                    return (int)k;
            }
            return -1;
        };

        // Layer 0 is a checkpoint layer (0 % 12 == 0) in every configuration —
        // block_size divides 0 regardless of its value.
        //
        // attn_res_score.weight is NOT a separate MatMul step — it is a 1-D scoring
        // vector consumed directly by AttnResMix (see the build-site comment), so the
        // check here is on AttnResMix's own tensor field, not a preceding step.
        const int L = 0;
        const int i_mix    = idx_of(L, K3Op::AttnResMix, "attn_res_mix");
        const int i_push   = idx_of(L, K3Op::ResBankPush, "res_bank_push");
        const int i_norm   = idx_of(L, K3Op::RmsNorm, "attn_norm");
        const int i_replace = idx_of(L, K3Op::AddResidual, "replace_attn");
        const int i_add_attn = idx_of(L, K3Op::AddResidual, "add_attn");

        check(i_mix >= 0 && i_push >= 0 && i_norm >= 0,
              "layer 0 has attn_res_mix, res_bank_push, attn_norm");
        check(i_mix >= 0 && plan.steps[i_mix].tensor == "blk.0.attn_res_score.weight",
              "attn_res_mix's own tensor is attn_res_score.weight, not a preceding "
              "MatMul's output");
        check(i_mix >= 0 && plan.steps[i_mix].expect_ne0 == cfg.hidden &&
                  plan.steps[i_mix].expect_ne1 == 0,
              "attn_res_score.weight pins ne0=hidden only — it is 1-D, not a "
              "[hidden, block_size] matrix");
        check(i_mix < i_push && i_push < i_norm,
              "order is attn_res_mix -> res_bank_push -> attn_norm "
              "(mix and bank BEFORE the norm, not after)");
        check(i_replace >= 0, "layer 0's attention residual combine is labelled "
                              "'replace_attn' (it is a checkpoint layer)");
        check(i_add_attn < 0, "layer 0 has NO 'add_attn' step — checkpoint layers "
                              "replace, they do not also add");
        if (i_replace >= 0)
            check(plan.steps[i_replace].residual_replace,
                  "layer 0's residual-combine step has residual_replace = true");

        // Layer 5 is NOT a checkpoint layer (5 % 12 != 0): ordinary add, no bank push.
        const int L2 = 5;
        check(idx_of(L2, K3Op::ResBankPush, nullptr) < 0,
              "layer 5 (not a checkpoint layer) has no res_bank_push");
        const int i_add5 = idx_of(L2, K3Op::AddResidual, "add_attn");
        check(i_add5 >= 0, "layer 5's attention residual combine is labelled 'add_attn'");
        if (i_add5 >= 0)
            check(!plan.steps[i_add5].residual_replace,
                  "layer 5's residual-combine step has residual_replace = false");

        // The FFN side NEVER replaces, checkpoint layer or not — and never gets a
        // bank push either.
        for (int layer : {0, 5}) {
            const int i_ffn_mix = idx_of(layer, K3Op::AttnResMix, "ffn_res_mix");
            const int i_ffn_norm = idx_of(layer, K3Op::RmsNorm, "ffn_norm");
            const int i_ffn_add = idx_of(layer, K3Op::AddResidual, "add_ffn");
            check(i_ffn_mix >= 0 && i_ffn_norm >= 0 && i_ffn_mix < i_ffn_norm,
                  "layer " + std::to_string(layer) +
                      ": ffn_res_mix runs before ffn_norm");
            check(i_ffn_add >= 0 && !plan.steps[i_ffn_add].residual_replace,
                  "layer " + std::to_string(layer) +
                      ": FFN-side residual combine is always 'add_ffn', never replace");
        }

        // Exactly one bank push per res_bs-layer cycle, i.e. ceil(n_layers/res_bs).
        int n_pushes = 0;
        for (const auto& s : plan.steps) if (s.op == K3Op::ResBankPush) ++n_pushes;
        const int want_pushes = (cfg.attn_res_block_size > 0)
            ? (cfg.n_layers + cfg.attn_res_block_size - 1) / cfg.attn_res_block_size
            : 0;
        check_eq(n_pushes, want_pushes,
                 "one bank push per res_block_size-layer cycle");
    }

    // ---------------------------------------------------- routed_norm + shexp order
    // Read directly off the reference's build_latent_moe: routed_norm runs on the
    // DISPATCH OUTPUT (right before routed_up), not on routed_down's output (the
    // dispatch's input). An earlier version of this plan had it backwards — the
    // same class of bug as the ssm_norm fix, caught the same way, by reading the
    // actual control flow rather than assuming a norm sits next to its same-named
    // projection. Also checks n_ff_shexp = moe_ffn * n_shared, confirmed against
    // the real UD-IQ1_S file (blk.3.ffn_gate_shexp.weight is 7168x6144).
    std::printf("\n[routed_norm normalises the dispatch OUTPUT, not its input]\n");
    {
        K3PlanOptions opt2;
        opt2.has_q_lora = true;
        opt2.has_routed_norm = true;
        opt2.has_shared_experts = true;
        const K3DecodePlan p2 = build_k3_decode_plan(cfg, d, opt2);

        const int L = 5;   // a KDA MoE layer
        int idx_down = -1, idx_dispatch_last = -1, idx_norm = -1, idx_up = -1;
        int idx_shexp_gate = -1, idx_shexp_down = -1;
        for (size_t k = 0; k < p2.steps.size(); ++k) {
            const auto& s = p2.steps[k];
            if (s.layer != L) continue;
            if (s.label == std::string("routed_down")) idx_down = (int)k;
            if (s.op == K3Op::MoeDispatch) idx_dispatch_last = (int)k;   // last wins
            if (s.op == K3Op::RmsNorm && s.label == std::string("routed_norm"))
                idx_norm = (int)k;
            if (s.label == std::string("routed_up")) idx_up = (int)k;
            if (s.label == std::string("shexp_gate")) idx_shexp_gate = (int)k;
            if (s.label == std::string("shexp_down")) idx_shexp_down = (int)k;
        }
        check(idx_down >= 0 && idx_dispatch_last >= 0 && idx_norm >= 0 && idx_up >= 0,
              "layer 5 has routed_down, a MoeDispatch step, routed_norm, routed_up");
        check(idx_down < idx_dispatch_last && idx_dispatch_last < idx_norm &&
                  idx_norm < idx_up,
              "order is routed_down -> dispatch -> routed_norm -> routed_up "
              "(norm AFTER the dispatch, not between routed_down and it)");
        check(idx_norm >= 0 && p2.steps[idx_norm].expect_ne0 == cfg.expert_latent,
              "routed_norm pins ne0=expert_latent (it normalises a latent-width "
              "tensor either way)");

        const int want_shexp = cfg.moe_ffn * cfg.n_shared;
        check(idx_shexp_gate >= 0 &&
                  p2.steps[idx_shexp_gate].out_dim == want_shexp &&
                  p2.steps[idx_shexp_gate].expect_ne1 == want_shexp,
              "shexp_gate out_dim/pin is moe_ffn*n_shared (6144 for K3), not moe_ffn "
              "(3072) alone");
        check(idx_shexp_down >= 0 && p2.steps[idx_shexp_down].in_dim == want_shexp,
              "shexp_down in_dim matches the same derived width");
    }

    // ------------------------------------------------------------ TP invariance
    std::printf("\n[TP changes placement, never the graph]\n");
    {
        tp::ShardDims one; one.tp_size = 1; one.rank = 0; one.hidden = cfg.hidden;
        const K3DecodePlan p1 = build_k3_decode_plan(cfg, one, opt);
        check_eq((long)p1.steps.size(), (long)plan.steps.size(),
                 "tp=1 and tp=8 produce the same number of steps");
        bool same = p1.steps.size() == plan.steps.size();
        for (size_t i = 0; same && i < p1.steps.size(); ++i)
            if (p1.steps[i].op != plan.steps[i].op ||
                p1.steps[i].tensor != plan.steps[i].tensor ||
                p1.steps[i].in_dim != plan.steps[i].in_dim) same = false;
        check(same, "the op sequence is identical at tp=1 and tp=8");
    }

    std::printf("\n[layer 0 (MLA, leading dense) as scheduled]\n%s",
                k3_plan_layer_dump(plan, 0).c_str());
    std::printf("\n[layer 5 (KDA, MoE) as scheduled]\n%s",
                k3_plan_layer_dump(plan, 5).c_str());

    std::printf("\n%s: %d checks, %d failures\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
