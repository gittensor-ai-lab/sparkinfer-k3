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

        // Still genuinely unpinned: the conv1d weight's exact axis layout (3-D vs 4-D
        // storage, trailing dim of 1 possibly squeezed) has ambiguity a naive pin
        // could get wrong, so it stays reported rather than asserted.
        auto conv = first.find("blk.5.ssm_conv1d_q.weight");
        check(conv != first.end() && conv->second->expect_ne0 == 0,
              "ssm_conv1d_q is still UNPINNED (axis-order ambiguity, not yet resolved)");

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
