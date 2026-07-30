// CPU-only test for multi-shard GGUF open + Kimi K3 tensor manifest validation.
//
// Writes a 2-shard tiny GGUF (split.count=2) with tensors distributed across
// shards, opens the first shard, and checks that GGUF::open merges both tensor
// tables and that kimi_k3_validate_tensors passes for a 4-layer toy config
// (3 KDA + 1 MLA, 1 dense lead + MoE).

#include "sparkinfer/models/kimi_k3_gguf_manifest.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
enum { VT_U16 = 2, VT_U32 = 4, VT_I32 = 5, VT_F32 = 6, VT_STR = 8, VT_ARR = 9 };

template <typename T>
void put(std::vector<uint8_t>& b, T v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, (uint64_t)s.size());
    b.insert(b.end(), s.begin(), s.end());
}

struct Meta {
    std::string key;
    int type = VT_U32;
    uint32_t u = 0;
    uint16_t u16 = 0;
    float f = 0.f;
    std::string s;
    std::vector<int32_t> arr;
};

struct Tensor {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0;  // F32
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

uint64_t tensor_bytes(const Tensor& t) {
    uint64_t n = 1;
    for (uint64_t d : t.dims) n *= d;
    return n * 4;
}

bool write_shard(const std::string& path, const std::vector<Meta>& meta,
                 std::vector<Tensor> tensors) {
    uint64_t off = 0;
    for (Tensor& t : tensors) {
        t.offset = off;
        t.bytes = tensor_bytes(t);
        off += t.bytes;
    }
    std::vector<uint8_t> b;
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    put<uint32_t>(b, 3);
    put<uint64_t>(b, tensors.size());
    put<uint64_t>(b, meta.size());
    for (const Meta& m : meta) {
        put_str(b, m.key);
        put<uint32_t>(b, (uint32_t)m.type);
        if (m.type == VT_STR) put_str(b, m.s);
        else if (m.type == VT_F32) put<float>(b, m.f);
        else if (m.type == VT_U16) put<uint16_t>(b, m.u16);
        else if (m.type == VT_ARR) {
            put<uint32_t>(b, VT_I32);
            put<uint64_t>(b, (uint64_t)m.arr.size());
            for (int32_t v : m.arr) put<int32_t>(b, v);
        } else put<uint32_t>(b, m.u);
    }
    for (const Tensor& t : tensors) {
        put_str(b, t.name);
        put<uint32_t>(b, (uint32_t)t.dims.size());
        for (uint64_t d : t.dims) put<uint64_t>(b, d);
        put<uint32_t>(b, t.type);
        put<uint64_t>(b, t.offset);
    }
    while (b.size() % 32) b.push_back(0);
    for (const Tensor& t : tensors) b.insert(b.end(), (size_t)t.bytes, 0);
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(b.data()), (std::streamsize)b.size());
    return out.good();
}

// 4 layers: 0,1,2 KDA; 3 MLA. leading_dense=1.
constexpr int N_LAYERS = 4;

std::vector<Meta> base_meta(uint16_t split_no, uint16_t split_count, int n_tensors_total) {
    // head_count_kv: 0,0,0,1
    std::vector<Meta> meta = {
        {"general.name", VT_STR, 0, 0, 0.f, "Kimi-K3-toy"},
        {"general.architecture", VT_STR, 0, 0, 0.f, "kimi-k3"},
        {"general.alignment", VT_U32, 32},
        {"split.no", VT_U16, 0, split_no},
        {"split.count", VT_U16, 0, split_count},
        {"split.tensors.count", VT_I32, (uint32_t)n_tensors_total},
        {"kimi-k3.block_count", VT_U32, N_LAYERS},
        {"kimi-k3.embedding_length", VT_U32, 64},
        {"kimi-k3.vocab_size", VT_U32, 128},
        {"kimi-k3.attention.head_count", VT_U32, 4},
        {"kimi-k3.attention.key_length", VT_U32, 16},
        {"kimi-k3.attention.key_length_mla", VT_U32, 12},
        {"kimi-k3.attention.value_length_mla", VT_U32, 8},
        {"kimi-k3.attention.q_lora_rank", VT_U32, 16},
        {"kimi-k3.attention.kv_lora_rank", VT_U32, 8},
        {"kimi-k3.rope.dimension_count", VT_U32, 4},
        {"kimi-k3.attention.layer_norm_rms_epsilon", VT_F32, 0, 0, 1e-5f},
        {"kimi-k3.expert_count", VT_U32, 8},
        {"kimi-k3.expert_used_count", VT_U32, 2},
        {"kimi-k3.expert_shared_count", VT_U32, 1},
        {"kimi-k3.expert_feed_forward_length", VT_U32, 32},
        {"kimi-k3.expert_latent_length", VT_U32, 32},
        {"kimi-k3.attn_res.block_size", VT_U32, 2},
        {"kimi-k3.activation.situ_beta", VT_F32, 0, 0, 4.0f},
        {"kimi-k3.activation.situ_linear_beta", VT_F32, 0, 0, 25.0f},
        {"kimi-k3.kda.head_dim", VT_U32, 8},
        {"kimi-k3.ssm.conv_kernel", VT_U32, 4},
        {"kimi-k3.kda.gate_lower_bound", VT_F32, 0, 0, -5.0f},
        {"kimi-k3.leading_dense_block_count", VT_U32, 1},
        {"kimi-k3.feed_forward_length", VT_U32, 64},
    };
    Meta hckv;
    hckv.key = "kimi-k3.attention.head_count_kv";
    hckv.type = VT_ARR;
    hckv.arr = {0, 0, 0, 1};
    meta.push_back(hckv);
    return meta;
}

std::vector<Tensor> kda_layer_tensors(int i) {
    auto n = [i](const char* s) {
        char b[64]; snprintf(b, sizeof(b), "blk.%d.%s", i, s); return std::string(b);
    };
    return {
        {n("attn_norm.weight"), {64}},
        {n("ffn_norm.weight"), {64}},
        {n("attn_res_score.weight"), {64}},
        {n("ffn_res_score.weight"), {64}},
        {n("ssm_conv1d_q.weight"), {4, 32}},
        {n("ssm_conv1d_k.weight"), {4, 32}},
        {n("ssm_conv1d_v.weight"), {4, 32}},
        {n("attn_q.weight"), {64, 32}},
        {n("attn_k.weight"), {64, 32}},
        {n("attn_v.weight"), {64, 32}},
        {n("ssm_f_a.weight"), {64, 8}},
        {n("ssm_f_b.weight"), {8, 32}},
        {n("ssm_beta.weight"), {64, 4}},
        {n("ssm_a"), {4}},
        {n("ssm_dt.bias"), {32}},
        {n("ssm_g.weight"), {64, 32}},
        {n("ssm_norm.weight"), {8}},
        {n("attn_output.weight"), {32, 64}},
    };
}

std::vector<Tensor> mla_layer_tensors(int i) {
    auto n = [i](const char* s) {
        char b[64]; snprintf(b, sizeof(b), "blk.%d.%s", i, s); return std::string(b);
    };
    return {
        {n("attn_norm.weight"), {64}},
        {n("ffn_norm.weight"), {64}},
        {n("attn_res_score.weight"), {64}},
        {n("ffn_res_score.weight"), {64}},
        {n("attn_kv_a_norm.weight"), {8}},
        {n("attn_q_a.weight"), {64, 16}},
        {n("attn_q_b.weight"), {16, 48}},
        {n("attn_kv_a_mqa.weight"), {64, 12}},
        {n("attn_k_b.weight"), {8, 8, 4}},
        {n("attn_v_b.weight"), {8, 8, 4}},
        {n("attn_output.weight"), {32, 64}},
    };
}

std::vector<Tensor> dense_ffn(int i) {
    auto n = [i](const char* s) {
        char b[64]; snprintf(b, sizeof(b), "blk.%d.%s", i, s); return std::string(b);
    };
    return {
        {n("ffn_gate.weight"), {64, 64}},
        {n("ffn_up.weight"), {64, 64}},
        {n("ffn_down.weight"), {64, 64}},
    };
}

std::vector<Tensor> moe_ffn(int i) {
    auto n = [i](const char* s) {
        char b[64]; snprintf(b, sizeof(b), "blk.%d.%s", i, s); return std::string(b);
    };
    return {
        {n("ffn_gate_inp.weight"), {64, 8}},
        {n("exp_probs_b.bias"), {8}},
        {n("ffn_gate_exps.weight"), {32, 32, 8}},
        {n("ffn_up_exps.weight"), {32, 32, 8}},
        {n("ffn_down_exps.weight"), {32, 32, 8}},
        {n("ffn_routed_down.weight"), {64, 32}},
        {n("ffn_routed_up.weight"), {32, 64}},
    };
}

void append_all(std::vector<Tensor>& dst, const std::vector<Tensor>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

#define CHECK(x) do { if (!(x)) { std::printf("FAIL: %s line %d\n", #x, __LINE__); return 1; } } while (0)
} // namespace

int main() {
    const std::string dir = "/tmp";
    const std::string prefix = dir + "/sparkinfer_kimi_k3_split_toy";
    const std::string shard0 = prefix + "-00001-of-00002.gguf";
    const std::string shard1 = prefix + "-00002-of-00002.gguf";

    // Shard 0: globals + layer 0 (KDA dense) + layer 1 (KDA MoE)
    // Shard 1: layer 2 (KDA MoE) + layer 3 (MLA MoE)
    std::vector<Tensor> t0 = {
        {"token_embd.weight", {64, 128}},
        {"output_norm.weight", {64}},
        {"output.weight", {64, 128}},
        {"output_res_score.weight", {64}},
    };
    append_all(t0, kda_layer_tensors(0));
    append_all(t0, dense_ffn(0));
    append_all(t0, kda_layer_tensors(1));
    append_all(t0, moe_ffn(1));

    std::vector<Tensor> t1;
    append_all(t1, kda_layer_tensors(2));
    append_all(t1, moe_ffn(2));
    append_all(t1, mla_layer_tensors(3));
    append_all(t1, moe_ffn(3));

    const int n_total = (int)(t0.size() + t1.size());
    auto m0 = base_meta(0, 2, n_total);
    auto m1 = base_meta(1, 2, n_total);
    // Later shards only need split bookkeeping + alignment for the reader;
    // architecture keys may be absent. Keep a minimal set.
    m1 = {
        {"general.alignment", VT_U32, 32},
        {"split.no", VT_U16, 0, 1},
        {"split.count", VT_U16, 0, 2},
        {"split.tensors.count", VT_I32, (uint32_t)n_total},
    };

    CHECK(write_shard(shard0, m0, t0));
    CHECK(write_shard(shard1, m1, t1));

    sparkinfer::GGUF g;
    CHECK(g.open(shard0));
    CHECK(g.n_shards() == 2);
    CHECK(g.n_tensors() == (size_t)n_total);
    CHECK(g.meta_int("split.count") == 2);
    CHECK(g.meta_int("split.no") == 0);

    // Tensors from both shards are visible.
    CHECK(g.tensor("token_embd.weight") != nullptr);
    CHECK(g.tensor("blk.0.ssm_g.weight") != nullptr);
    CHECK(g.tensor("blk.2.ssm_g.weight") != nullptr);
    CHECK(g.tensor("blk.3.attn_q_a.weight") != nullptr);
    CHECK(g.tensor("blk.0.ssm_g.weight")->shard == 0);
    CHECK(g.tensor("blk.3.attn_q_a.weight")->shard == 1);

    sparkinfer::KimiK3Config cfg;
    CHECK(sparkinfer::kimi_k3_config_from_gguf(g, cfg));
    CHECK(cfg.n_layers == 4);
    CHECK(cfg.n_kda_layers() == 3);
    CHECK(cfg.n_mla_layers() == 1);
    CHECK(cfg.is_kda_layer(0));
    CHECK(!cfg.is_kda_layer(3));

    sparkinfer::KimiK3ManifestReport rep;
    CHECK(sparkinfer::kimi_k3_validate_tensors(g, cfg, &rep));
    CHECK(rep.missing == 0);
    CHECK(rep.checked > 0);

    // Opening shard 1 alone must fail (split.no != 0).
    sparkinfer::GGUF bad;
    CHECK(!bad.open(shard1));

    std::printf("kimi_k3_manifest_cpu_test: OK  (%d tensors across 2 shards, %d checks)\n",
                n_total, rep.checked);
    return 0;
}
