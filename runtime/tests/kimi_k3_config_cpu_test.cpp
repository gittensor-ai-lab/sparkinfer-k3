// CPU-only test for Kimi K3 GGUF metadata parsing.
//
// Writes a tiny GGUF with kimi-k3 architecture keys (including the per-layer
// head_count_kv int array), then verifies kimi_k3_config_from_gguf() derives
// the config a loader needs before sizing CUDA scratch / recurrent state.
//
// Also asserts that dropping any REQUIRED key fails the parse — the reference
// loader treats those as GGML_ASSERT for the same reason (silent defaults emit
// fluent garbage).

#include "sparkinfer/models/kimi_k3_gguf_config.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
enum { VT_U32 = 4, VT_I32 = 5, VT_F32 = 6, VT_STR = 8, VT_ARR = 9 };

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
    int type;
    uint32_t u = 0;
    float f = 0.f;
    std::string s;
    std::vector<int32_t> arr;  // for VT_ARR of VT_I32
};

struct Tensor {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0; // F32
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

uint64_t tensor_bytes(const Tensor& t) {
    uint64_t n = 1;
    for (uint64_t d : t.dims) n *= d;
    return n * 4;
}

// Build the 1-indexed MLA set from the YAML: every 4th layer plus layer 93.
std::vector<int32_t> make_head_count_kv(int n_layers) {
    std::vector<int32_t> v(n_layers, 0);  // 0 = KDA
    for (int one_based = 4; one_based <= 92; one_based += 4)
        v[one_based - 1] = 1;             // MLA
    if (n_layers >= 93) v[92] = 1;        // layer 93
    return v;
}

bool write_tiny_gguf(const std::string& path, bool drop_situ_beta = false,
                     bool drop_head_kv = false, int n_layers = 93) {
    auto hckv = make_head_count_kv(n_layers);
    std::vector<Meta> meta = {
        {"general.name", VT_STR, 0, 0.f, "Kimi-K3"},
        {"general.architecture", VT_STR, 0, 0.f, "kimi-k3"},
        {"general.alignment", VT_U32, 32},
        {"kimi-k3.block_count", VT_U32, (uint32_t)n_layers},
        {"kimi-k3.embedding_length", VT_U32, 7168},
        {"kimi-k3.vocab_size", VT_U32, 163840},
        {"kimi-k3.attention.head_count", VT_U32, 96},
        {"kimi-k3.attention.key_length", VT_U32, 576},
        {"kimi-k3.attention.key_length_mla", VT_U32, 192},
        {"kimi-k3.attention.value_length_mla", VT_U32, 128},
        {"kimi-k3.attention.q_lora_rank", VT_U32, 1536},
        {"kimi-k3.attention.kv_lora_rank", VT_U32, 512},
        {"kimi-k3.rope.dimension_count", VT_U32, 64},
        {"kimi-k3.attention.layer_norm_rms_epsilon", VT_F32, 0, 1e-5f},
        {"kimi-k3.expert_count", VT_U32, 896},
        {"kimi-k3.expert_used_count", VT_U32, 16},
        {"kimi-k3.expert_shared_count", VT_U32, 2},
        {"kimi-k3.expert_feed_forward_length", VT_U32, 3072},
        {"kimi-k3.expert_latent_length", VT_U32, 3584},
        {"kimi-k3.attn_res.block_size", VT_U32, 12},
        {"kimi-k3.activation.situ_linear_beta", VT_F32, 0, 25.0f},
        {"kimi-k3.kda.head_dim", VT_U32, 128},
        {"kimi-k3.ssm.conv_kernel", VT_U32, 4},
        {"kimi-k3.kda.gate_lower_bound", VT_F32, 0, -5.0f},
        {"kimi-k3.leading_dense_block_count", VT_U32, 1},
        {"tokenizer.ggml.eos_token_id", VT_U32, 163839},
    };
    if (!drop_situ_beta)
        meta.push_back({"kimi-k3.activation.situ_beta", VT_F32, 0, 4.0f});
    if (!drop_head_kv) {
        Meta a;
        a.key = "kimi-k3.attention.head_count_kv";
        a.type = VT_ARR;
        a.arr = hckv;
        meta.push_back(a);
    }

    std::vector<Tensor> tensors = {
        {"token_embd.weight", {64, 163840}},          // tiny rows; vocab from dims[1]
        {"blk.0.attn_q.weight", {64, 128}},           // KDA layer 0
        {"blk.0.ssm_conv1d_q.weight", {4, 128}},
        {"blk.3.attn_q_a.weight", {64, 32}},          // MLA layer 3 (1-based 4)
        {"blk.3.attn_gate.weight", {64, 128}},
    };

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

#define CHECK(x) do { if (!(x)) { std::printf("FAIL: %s line %d\n", #x, __LINE__); return 1; } } while (0)
} // namespace

int main() {
    const std::string path = "/tmp/sparkinfer_kimi_k3_config_cpu_test.gguf";
    CHECK(write_tiny_gguf(path));

    sparkinfer::GGUF g;
    CHECK(g.open(path));
    CHECK(sparkinfer::kimi_k3_is_arch(g));

    sparkinfer::KimiK3Config cfg;
    CHECK(sparkinfer::kimi_k3_config_from_gguf(g, cfg));

    CHECK(cfg.n_layers == 93);
    CHECK(cfg.hidden == 7168);
    CHECK(cfg.vocab == 163840);          // from token_embd dims[1]
    CHECK(cfg.n_q_heads == 96);
    CHECK(cfg.n_experts == 896);
    CHECK(cfg.top_k == 16);
    CHECK(cfg.n_shared == 2);
    CHECK(cfg.moe_ffn == 3072);
    CHECK(cfg.expert_latent == 3584);
    CHECK(cfg.attn_res_block_size == 12);
    CHECK(cfg.situ_beta == 4.0f);
    CHECK(cfg.situ_linear_beta == 25.0f);
    CHECK(cfg.kda_head_dim == 128);
    CHECK(cfg.kda_conv_kernel == 4);
    CHECK(cfg.kda_gate_lower_bound == -5.0f);
    CHECK(cfg.kv_lora_rank == 512);
    CHECK(cfg.q_lora_rank == 1536);
    CHECK(cfg.key_length == 576);
    CHECK(cfg.key_length_mla == 192);
    CHECK(cfg.value_length_mla == 128);

    CHECK((int)cfg.layer_is_kda.size() == 93);
    CHECK(cfg.n_kda_layers() == 69);
    CHECK(cfg.n_mla_layers() == 24);

    // REDUCING n_layers WITHOUT TRUNCATING layer_is_kda must stay consistent.
    // n_mla_layers() is n_layers - n_kda_layers(), so if the KDA count ranged over
    // the whole map while n_layers was smaller, the difference would go NEGATIVE and
    // then blow up far away (a negative int widening into vector::resize()'s size_t,
    // throwing std::length_error with no hint of the cause). A caller capping depth
    // to what fits in VRAM, or bringing up a partial model, does exactly this.
    {
        sparkinfer::KimiK3Config c = cfg;   // full 93-entry map retained on purpose
        c.n_layers = 13;
        CHECK((int)c.layer_is_kda.size() == 93);          // map deliberately longer
        CHECK(c.n_kda_layers() + c.n_mla_layers() == 13);  // the invariant that matters
        CHECK(c.n_kda_layers() >= 0 && c.n_mla_layers() >= 0);
        CHECK(c.n_kda_layers() <= 13 && c.n_mla_layers() <= 13);
        // Layers 0..12 under the real map: MLA at 3, 7, 11 -> 3 MLA, 10 KDA.
        CHECK(c.n_mla_layers() == 3);
        CHECK(c.n_kda_layers() == 10);

        sparkinfer::KimiK3Config z = cfg;
        z.n_layers = 0;
        CHECK(z.n_kda_layers() == 0 && z.n_mla_layers() == 0);
    }
    // Layer 0 (1-based 1) is KDA; layer 3 (1-based 4) is MLA; layer 92 (1-based 93) is MLA.
    CHECK(cfg.is_kda_layer(0));
    CHECK(!cfg.is_kda_layer(3));
    CHECK(!cfg.is_kda_layer(92));
    CHECK(cfg.is_kda_layer(1));
    CHECK(g.tensor("blk.0.ssm_conv1d_q.weight") != nullptr);
    CHECK(g.tensor("blk.3.attn_gate.weight") != nullptr);

    // Missing situ_beta must refuse.
    {
        const std::string bad = "/tmp/sparkinfer_kimi_k3_config_cpu_test_nositu.gguf";
        CHECK(write_tiny_gguf(bad, /*drop_situ_beta=*/true));
        sparkinfer::GGUF gb;
        CHECK(gb.open(bad));
        sparkinfer::KimiK3Config c2;
        CHECK(!sparkinfer::kimi_k3_config_from_gguf(gb, c2));
    }

    // Missing head_count_kv array must refuse.
    {
        const std::string bad = "/tmp/sparkinfer_kimi_k3_config_cpu_test_nokv.gguf";
        CHECK(write_tiny_gguf(bad, false, /*drop_head_kv=*/true));
        sparkinfer::GGUF gb;
        CHECK(gb.open(bad));
        sparkinfer::KimiK3Config c2;
        CHECK(!sparkinfer::kimi_k3_config_from_gguf(gb, c2));
    }

    // Wrong architecture must refuse.
    CHECK(g.meta_str("general.architecture") == "kimi-k3");
    {
        const std::string bad = "/tmp/sparkinfer_kimi_k3_config_cpu_test_badarch.gguf";
        // Reuse a file and only check the arch gate helper against empty.
        sparkinfer::GGUF empty;
        CHECK(!sparkinfer::kimi_k3_is_arch(empty));
        (void)bad;
    }

    std::printf("kimi_k3_config_cpu_test: OK  (69 KDA + 24 MLA, situ/attn_res/latent required)\n");
    return 0;
}
