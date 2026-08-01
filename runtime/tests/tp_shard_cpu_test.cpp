// CPU-only tests for the tensor-parallel shard math (no GPU needed).
//
// The collectives are a solved problem — NCCL is correct. What is NOT solved is
// the shard arithmetic, and that is what this file pins down, because every bug
// in it produces a model that loads cleanly, runs at full speed, and emits
// subtly wrong tokens. Verifying it here costs a second; discovering it on a
// rented 8x H200 node costs an afternoon and a plausible-looking wrong number.
//
// Build/run:
//   g++ -std=c++17 -I runtime/include runtime/tests/tp_shard_cpu_test.cpp
//       runtime/src/tp/shard.cpp -o /tmp/tp_shard_cpu_test && /tmp/tp_shard_cpu_test

#include "sparkinfer/tp/shard.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

using namespace sparkinfer::tp;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                                        \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            std::printf("  FAIL %s:%d: %s\n        ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                                           \
            std::printf("\n");                                                   \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

#define CHECK_EQ(a, b) CHECK((a) == (b), "got %lld want %lld", (long long)(a), (long long)(b))

// Real shapes, so the tests fail on the configs that will actually be run.
static ModelShape qwen36() {
    ModelShape s;
    s.hidden = 2048; s.n_q_heads = 16; s.n_kv_heads = 2; s.head_dim = 256;
    s.vocab = 248320; s.n_experts = 256; s.moe_ffn = 512; s.dense_ffn = 0;
    return s;
}

// Kimi K3 as a GGUF presents it: MLA stored as MQA with a single large kv head,
// 96 query heads, 896 routed experts, 163840 vocab.
static ModelShape kimi_k3() {
    ModelShape s;
    s.hidden = 7168; s.n_q_heads = 96; s.n_kv_heads = 1; s.head_dim = 128;
    s.vocab = 163840; s.n_experts = 896; s.moe_ffn = 3072; s.dense_ffn = 0;
    return s;
}

static Config cfg_of(int tp) {
    Config c;
    c.tp_size = tp;
    for (int i = 0; i < tp; ++i) c.devices.push_back(i);
    return c;
}

// ---------------------------------------------------------------------------

static void test_tp1_is_the_identity_shard() {
    std::printf("tp1_is_the_identity_shard\n");
    // TP=1 must be indistinguishable from the pre-TP path: nothing sharded,
    // nothing replicated. This is what makes landing TP safe before validation.
    for (ModelShape s : {qwen36(), kimi_k3()}) {
        ShardDims d;
        ShardError e = shard_dims(s, cfg_of(1), 0, &d);
        CHECK(e.ok(), "unexpected error: %s", e.message.c_str());
        CHECK_EQ(d.n_q_heads, s.n_q_heads);
        CHECK_EQ(d.n_kv_heads, s.n_kv_heads);
        CHECK(!d.kv_replicated, "tp=1 must not report replication");
        CHECK_EQ(d.n_experts, s.n_experts);
        CHECK_EQ(d.moe_ffn, s.moe_ffn);
        CHECK_EQ(d.vocab, s.vocab);
        CHECK_EQ(d.vocab_band.offset, 0);
    }
    Config c;   // default-constructed: tp_size 1, no devices listed
    CHECK(!c.enabled(), "default Config must be disabled");
}

static void test_qwen36_tp8() {
    std::printf("qwen36_tp8\n");
    ShardDims d;
    ShardError e = shard_dims(qwen36(), cfg_of(8), 3, &d);
    CHECK(e.ok(), "unexpected error: %s", e.message.c_str());
    CHECK_EQ(d.n_q_heads, 2);                 // 16 / 8
    CHECK_EQ(d.q_rows, 2 * 256);
    // 2 kv heads at tp 8: fewer heads than ranks, so REPLICATE, do not divide.
    // If this ever reports 0 the model has no keys on any rank.
    CHECK_EQ(d.n_kv_heads, 2);
    CHECK(d.kv_replicated, "2 kv heads at tp=8 must replicate");
    CHECK_EQ(d.kv_rows, 2 * 256);
    CHECK_EQ(d.n_experts, 32);                // 256 / 8
    CHECK(d.experts_sharded, "256 experts / 8 should shard by expert");
    CHECK_EQ(d.moe_ffn, 512);                 // full width, fewer experts
    CHECK_EQ(d.expert_band.offset, 96);       // rank 3 * 32
    CHECK_EQ(d.vocab, 31040);                 // 248320 / 8
    CHECK_EQ(d.vocab_band.offset, 3 * 31040);
    CHECK_EQ(d.hidden, 2048);                 // never sharded
}

static void test_kimi_k3_tp8() {
    std::printf("kimi_k3_tp8\n");
    ShardDims d;
    ShardError e = shard_dims(kimi_k3(), cfg_of(8), 0, &d);
    CHECK(e.ok(), "unexpected error: %s", e.message.c_str());
    CHECK_EQ(d.n_q_heads, 12);                // 96 / 8
    CHECK_EQ(d.n_kv_heads, 1);
    CHECK(d.kv_replicated, "MLA-as-MQA single kv head must replicate at tp=8");
    CHECK_EQ(d.n_experts, 112);               // 896 / 8
    CHECK(d.experts_sharded, "896 experts / 8 should shard by expert");
    CHECK_EQ(d.vocab, 20480);                 // 163840 / 8
    CHECK_EQ(d.hidden, 7168);
}

static void test_kimi_k3_tp4_for_b300() {
    std::printf("kimi_k3_tp4_for_b300\n");
    // M3's node is 4x B300. 96/4, 896/4 and 163840/4 all divide, so TP=4 is
    // available without falling back to ffn-width sharding.
    ShardDims d;
    CHECK(shard_dims(kimi_k3(), cfg_of(4), 2, &d).ok(), "tp4 should be valid");
    CHECK_EQ(d.n_q_heads, 24);
    CHECK_EQ(d.n_experts, 224);
    CHECK_EQ(d.expert_band.offset, 448);      // rank 2 * 224
    CHECK_EQ(d.vocab, 40960);
}

static void test_shape_must_divide_on_every_axis() {
    std::printf("shape_must_divide_on_every_axis\n");
    // A shard is only usable if EVERY sharded axis divides — it is not enough for
    // the experts to divide. K3 has 896 experts, and 896 % 7 == 0, so a naive
    // expert-only check would happily accept tp=7 — but K3 has 96 query heads and
    // 96 % 7 != 0, so rank 0 would get 13 heads and rank 6 would get 13 with one
    // left over. shard_dims must reject the whole shape, naming the axis at fault.
    ModelShape s = kimi_k3();
    CHECK_EQ(s.n_experts % 7, 0);        // the tempting-but-irrelevant fact
    for (int tp : {7, 14}) {
        ShardDims d;
        ShardError e = shard_dims(s, cfg_of(tp), 0, &d);
        CHECK(!e.ok(), "tp=%d must be rejected for K3 even though experts divide", tp);
        CHECK(e.message.find("n_q_heads") != std::string::npos,
              "tp=%d error should name n_q_heads, got: %s", tp, e.message.c_str());
    }
}

static void test_expert_bands_tile_exactly() {
    std::printf("expert_bands_tile_exactly\n");
    // Every routed expert must be owned by exactly one rank. A gap silently
    // drops an expert's contribution; an overlap double-counts it. Either way
    // the all-reduce combine is wrong and the output is plausible but off.
    //
    // Only tp values that divide EVERY axis of the shape (see the test above):
    // 96 q heads, 896 experts and 163840 vocab all divide by 2/4/8/16/32.
    for (int tp : {2, 4, 8, 16, 32}) {
        ModelShape s = kimi_k3();
        std::set<int> seen;
        int prev_end = 0;
        for (int r = 0; r < tp; ++r) {
            ShardDims d;
            CHECK(shard_dims(s, cfg_of(tp), r, &d).ok(), "tp=%d rank=%d", tp, r);
            CHECK_EQ(d.expert_band.offset, prev_end);          // no gap
            prev_end = d.expert_band.end();
            for (int x = d.expert_band.offset; x < d.expert_band.end(); ++x) {
                CHECK(seen.insert(x).second, "expert %d owned twice at tp=%d", x, tp);
            }
        }
        CHECK_EQ(prev_end, s.n_experts);                        // full coverage
        CHECK_EQ((int)seen.size(), s.n_experts);
    }
}

static void test_vocab_bands_tile_exactly() {
    std::printf("vocab_bands_tile_exactly\n");
    // Same argument for the lm_head row shard: a gap means tokens that can never
    // be predicted by any rank.
    const int tp = 8;
    ModelShape s = kimi_k3();
    int prev_end = 0;
    for (int r = 0; r < tp; ++r) {
        ShardDims d;
        CHECK(shard_dims(s, cfg_of(tp), r, &d).ok(), "rank %d", r);
        CHECK_EQ(d.vocab_band.offset, prev_end);
        prev_end = d.vocab_band.end();
    }
    CHECK_EQ(prev_end, s.vocab);
}

static void test_indivisible_shapes_are_rejected_with_a_reason() {
    std::printf("indivisible_shapes_are_rejected_with_a_reason\n");
    // Failing loudly is the whole point — a ragged shard would need per-rank
    // launch geometry and would break CUDA-graph capture.
    ModelShape s = qwen36();
    s.n_q_heads = 15;                       // prime-ish, will not divide by 8
    ShardDims d;
    ShardError e = shard_dims(s, cfg_of(8), 0, &d);
    CHECK(!e.ok(), "15 q heads at tp=8 must be rejected");
    CHECK(e.message.find("n_q_heads") != std::string::npos,
          "error should name the offending field, got: %s", e.message.c_str());

    // Multiple problems are reported together, not one-at-a-time.
    ModelShape bad = qwen36();
    bad.n_q_heads = 15;
    bad.vocab = 248321;
    e = shard_dims(bad, cfg_of(8), 0, &d);
    CHECK(!e.ok(), "should reject");
    CHECK(e.message.find("n_q_heads") != std::string::npos &&
          e.message.find("vocab") != std::string::npos,
          "both problems should be reported, got: %s", e.message.c_str());
}

static void test_expert_fallback_to_ffn_width() {
    std::printf("expert_fallback_to_ffn_width\n");
    // Experts do not divide but the intermediate width does: replicate experts,
    // split each one's ffn. Correct, just more weight traffic per rank.
    ModelShape s = kimi_k3();
    s.n_experts = 100;                      // 100 / 8 does not divide
    s.moe_ffn = 3072;                       // 3072 / 8 does
    ShardDims d;
    ShardError e = shard_dims(s, cfg_of(8), 5, &d);
    CHECK(e.ok(), "should fall back, not fail: %s", e.message.c_str());
    CHECK(!d.experts_sharded, "should have fallen back to ffn-width sharding");
    CHECK_EQ(d.n_experts, 100);             // all experts on every rank
    CHECK_EQ(d.moe_ffn, 384);               // 3072 / 8
}

static void test_bad_rank_and_device_list() {
    std::printf("bad_rank_and_device_list\n");
    ShardDims d;
    CHECK(!shard_dims(qwen36(), cfg_of(8), 8, &d).ok(), "rank == tp_size is out of range");
    CHECK(!shard_dims(qwen36(), cfg_of(8), -1, &d).ok(), "negative rank");
    Config c;
    c.tp_size = 8;
    c.devices = {0, 1, 2};                  // mismatched length
    CHECK(!shard_dims(qwen36(), c, 0, &d).ok(), "devices/tp_size mismatch must be rejected");
    CHECK(!shard_dims(qwen36(), cfg_of(0), 0, &d).ok(), "tp_size 0");
}

static void test_row_shard_slices_are_contiguous_and_tile() {
    std::printf("row_shard_slices_are_contiguous_and_tile\n");
    const int rows = 1024, cols = 7168, tp = 8;
    std::size_t total = 0;
    std::size_t expect_off = 0;
    for (int r = 0; r < tp; ++r) {
        Slice s = row_shard(rows, cols, tp, r);
        CHECK_EQ(s.rows, rows / tp);
        CHECK_EQ(s.cols, cols);
        CHECK_EQ(s.src_offset, expect_off);           // contiguous, no gap
        CHECK_EQ(s.count, (std::size_t)(rows / tp) * cols);
        expect_off += s.count;
        total += s.count;
    }
    CHECK_EQ(total, (std::size_t)rows * cols);        // exactly covers the matrix
    CHECK_EQ(row_shard(1023, cols, tp, 0).count, 0);  // indivisible -> empty
}

static void test_col_shard_is_strided_and_tiles_per_row() {
    std::printf("col_shard_is_strided_and_tiles_per_row\n");
    const int rows = 4, cols = 16, tp = 4;
    CHECK_EQ(col_shard_row_extent(cols, tp), 4);

    // Every element of the full matrix belongs to exactly one rank's band.
    std::vector<int> owner(rows * cols, -1);
    for (int r = 0; r < tp; ++r) {
        Slice s = col_shard(rows, cols, tp, r);
        CHECK_EQ(s.rows, rows);
        CHECK_EQ(s.cols, cols / tp);
        // count is the SHARD size, not the bounding span — a caller that memcpys
        // `count` from `src_offset` under-copies rather than reading OOB.
        CHECK_EQ(s.count, (std::size_t)rows * (cols / tp));
        for (int row = 0; row < rows; ++row) {
            std::size_t off = col_shard_row_offset(row, cols, tp, r);
            for (int k = 0; k < cols / tp; ++k) {
                std::size_t idx = off + k;
                CHECK(idx < owner.size(), "index %zu out of range", idx);
                CHECK(owner[idx] == -1, "element %zu claimed twice", idx);
                owner[idx] = r;
            }
        }
    }
    for (std::size_t i = 0; i < owner.size(); ++i) {
        CHECK(owner[i] != -1, "element %zu owned by nobody", i);
    }
    CHECK_EQ(col_shard_row_extent(15, 4), 0);        // indivisible -> 0
}

static void test_row_and_col_shard_compose() {
    std::printf("row_and_col_shard_compose\n");
    // The reason TP works with no redistribution mid-layer: Wq is row-sharded by
    // output, Wo is col-sharded by input, and rank r's Wq output rows line up
    // exactly with rank r's Wo input columns.
    const int tp = 8, heads = 96, hd = 128, hidden = 7168;
    const int q_rows_total = heads * hd;             // Wq output rows
    for (int r = 0; r < tp; ++r) {
        Slice wq = row_shard(q_rows_total, hidden, tp, r);
        Slice wo = col_shard(hidden, q_rows_total, tp, r);
        CHECK_EQ(wq.rows, wo.cols);                  // the seam matches
        CHECK_EQ(wq.rows, q_rows_total / tp);
        std::size_t wq_row_start = (std::size_t)(q_rows_total / tp) * r;
        std::size_t wo_col_start = col_shard_row_offset(0, q_rows_total, tp, r);
        CHECK_EQ(wq_row_start, wo_col_start);        // and at the same offset
    }
}

static void test_reduce_accounting() {
    std::printf("reduce_accounting\n");
    ShardDims d;
    CHECK(shard_dims(kimi_k3(), cfg_of(8), 0, &d).ok(), "shard");
    // Both collectives carry FULL hidden width — that is what makes them
    // all-reduces rather than all-gathers.
    CHECK_EQ(reduce_elems(ReducePoint::AfterAttnOut, d, 1), 7168u);
    CHECK_EQ(reduce_elems(ReducePoint::AfterFfnDown, d, 1), 7168u);
    CHECK_EQ(reduce_elems(ReducePoint::AfterAttnOut, d, 64), 7168u * 64);
    CHECK_EQ(reduce_elems(ReducePoint::AfterAttnOut, d, 0), 0u);
    // K3: 93 layers x 2 = 186 collectives per decoded token, 14 KiB each at bf16.
    // Latency-bound, not bandwidth-bound — which is why the collective mechanism
    // matters more than its throughput.
    CHECK_EQ(reduce_count_per_token(93), 186);
    CHECK_EQ(reduce_count_per_token(0), 0);
}

static void test_even_band_edges() {
    std::printf("even_band_edges\n");
    CHECK_EQ(even_band(896, 8, 0).offset, 0);
    CHECK_EQ(even_band(896, 8, 7).offset, 784);
    CHECK_EQ(even_band(896, 8, 7).extent, 112);
    CHECK(even_band(896, 8, 8).empty(), "rank out of range -> empty");
    CHECK(even_band(896, 8, -1).empty(), "negative rank -> empty");
    CHECK(even_band(896, 0, 0).empty(), "tp_size 0 -> empty");
    CHECK(divisible(896, 8), "896/8");
    CHECK(!divisible(895, 8), "895/8");
    CHECK(!divisible(896, 0), "div by zero");
    CHECK(!divisible(0, 8), "zero total");
}

// ---------------------------------------------------------------------------
// Kimi K3 KDA — head-parallel shard math
// ---------------------------------------------------------------------------

// K3's real KDA dimensions, from KimiK3Config.
static KdaShape k3_kda() {
    KdaShape s;
    s.hidden = 7168; s.n_heads = 96; s.head_dim = 128; s.conv_kernel = 4;
    return s;
}

static void test_kda_tp8_real_dims() {
    std::printf("kda_tp8_real_dims\n");
    KdaShardDims d;
    KdaShape sh = k3_kda();
    ShardError e = kda_shard_dims(sh, cfg_of(8), 3, &d);
    CHECK(e.ok(), "unexpected error: %s", e.message.c_str());
    CHECK_EQ(d.n_heads_total, 96);
    CHECK_EQ(d.n_heads, 12);                  // 96 / 8
    CHECK_EQ(d.head_band.offset, 36);         // rank 3 * 12
    CHECK_EQ(d.qkv_total, 12288);             // 96 * 128
    CHECK_EQ(d.qkv, 1536);                    // 12 * 128
    CHECK_EQ(d.qkv_band.offset, 4608);        // 36 * 128
    CHECK_EQ(d.head_dim, 128);                // never sharded
    CHECK_EQ(d.hidden, 7168);                 // never sharded
    // Per-rank recurrent state. Allocating the full-width version and indexing it
    // with a rank-local head id is the silent-corruption failure this pins.
    CHECK_EQ(d.conv_state_elems, 3 * 1536);           // (4-1) * qkv
    CHECK_EQ(d.delta_state_elems, 128 * 128 * 12);    // head_dim^2 * n_heads
}

static void test_kda_bands_tile_and_hold_the_invariant() {
    std::printf("kda_bands_tile_and_hold_the_invariant\n");
    const int tp = 8;
    KdaShape sh = k3_kda();
    int heads_seen = 0, qkv_seen = 0;
    int next_head = 0, next_qkv = 0;
    long conv_total = 0, delta_total = 0;
    for (int r = 0; r < tp; ++r) {
        KdaShardDims d;
        CHECK(kda_shard_dims(sh, cfg_of(tp), r, &d).ok(), "rank %d", r);
        // Contiguous, gapless, non-overlapping — checked by walking the seam
        // rather than by set membership, so an overlap cannot hide as a duplicate.
        CHECK_EQ(d.head_band.offset, next_head);
        CHECK_EQ(d.qkv_band.offset, next_qkv);
        next_head = d.head_band.end();
        next_qkv = d.qkv_band.end();
        heads_seen += d.n_heads;
        qkv_seen += d.qkv;
        conv_total += d.conv_state_elems;
        delta_total += d.delta_state_elems;
        // THE INVARIANT the header names: the two addressings agree exactly.
        CHECK_EQ(d.qkv_band.offset, d.head_band.offset * sh.head_dim);
        CHECK_EQ(d.qkv_band.extent, d.head_band.extent * sh.head_dim);
    }
    CHECK_EQ(next_head, 96);
    CHECK_EQ(next_qkv, 12288);
    CHECK_EQ(heads_seen, 96);
    CHECK_EQ(qkv_seen, 12288);
    // The state divides exactly too: no rank carries a remainder, so every rank
    // gets the identical launch geometry.
    CHECK_EQ(conv_total, 3L * 12288);
    CHECK_EQ(delta_total, 128L * 128 * 96);
}

static void test_kda_tp1_is_the_identity_shard() {
    std::printf("kda_tp1_is_the_identity_shard\n");
    KdaShardDims d;
    CHECK(kda_shard_dims(k3_kda(), cfg_of(1), 0, &d).ok(), "tp1");
    CHECK_EQ(d.n_heads, 96);
    CHECK_EQ(d.qkv, 12288);
    CHECK_EQ(d.head_band.offset, 0);
    CHECK_EQ(d.head_band.extent, 96);
    CHECK_EQ(d.qkv_band.extent, 12288);
    CHECK_EQ(d.conv_state_elems, 3L * 12288);
    CHECK_EQ(d.delta_state_elems, 128L * 128 * 96);
}

static void test_kda_rejects_what_it_cannot_split() {
    std::printf("kda_rejects_what_it_cannot_split\n");
    KdaShardDims d;
    // 96 % 7 != 0. A KDA head owns a private recurrent state, so there is no
    // replicate-the-remainder fallback the way there is for kv heads.
    ShardError e = kda_shard_dims(k3_kda(), cfg_of(7), 0, &d);
    CHECK(!e.ok(), "96 heads over 7 ranks must be rejected");
    CHECK(e.message.find("recurrent state") != std::string::npos,
          "the reason must name why a head is atomic, got: %s", e.message.c_str());

    // A zeroed shape must report the SHAPE problem, not a divisibility one —
    // "0 not divisible by 8" reads like a config error when it is plumbing.
    KdaShape empty;
    ShardError z = kda_shard_dims(empty, cfg_of(8), 0, &d);
    CHECK(!z.ok(), "default-constructed shape must be rejected");
    CHECK(z.message.find("n_heads must be > 0") != std::string::npos,
          "got: %s", z.message.c_str());

    CHECK(!kda_shard_dims(k3_kda(), cfg_of(8), 8, &d).ok(), "rank == tp_size");
    CHECK(!kda_shard_dims(k3_kda(), cfg_of(8), -1, &d).ok(), "negative rank");
    CHECK(!kda_shard_dims(k3_kda(), cfg_of(8), 0, nullptr).ok(), "null out");
}

static void test_kda_replicate_and_offset_bands() {
    std::printf("kda_replicate_and_offset_bands\n");
    // Four KDA tensors stay REPLICATED at full width and are indexed at this
    // rank's band instead (see the header). The offsets are pinned HERE rather
    // than left to the call site, because a forward that offsets by anything else
    // reads another rank's heads and the model stays fluent.
    const int tp = 8;
    KdaShape sh = k3_kda();
    for (int r = 0; r < tp; ++r) {
        KdaShardDims d;
        CHECK(kda_shard_dims(sh, cfg_of(tp), r, &d).ok(), "rank %d", r);

        // ssm_dt.bias, f32 [qkv] -> + qkv_band.offset
        CHECK_EQ(d.qkv_band.offset, r * 1536);
        // ssm_a, f32 [n_heads] -> + head_band.offset
        CHECK_EQ(d.head_band.offset, r * 12);
        // ssm_conv1d_{q,k,v}, f32 [conv_kernel, 1, qkv], CHANNEL-MAJOR:
        // kda_conv_step_kernel reads w + c * d_conv, so a channel band is
        // contiguous and the offset scales by conv_kernel.
        CHECK_EQ((long)d.qkv_band.offset * d.conv_kernel, (long)r * 1536 * 4);
        // The recurrent state uses rank-LOCAL channel ids against a per-rank
        // allocation, so it carries no offset at all — only a smaller size.
        CHECK_EQ(d.conv_state_elems, 3L * 1536);

        // The offset must land on this rank's OWN heads, not a neighbour's:
        // the last channel of the band is one short of the next band's first.
        CHECK_EQ(d.qkv_band.end(), (r + 1) * 1536);
        CHECK_EQ(d.head_band.end(), (r + 1) * 12);
    }
    // At tp=1 every offset is zero, so the replicate-and-offset path is a no-op
    // and the single-GPU forward is byte-identical to the pre-shard one.
    KdaShardDims one;
    CHECK(kda_shard_dims(sh, cfg_of(1), 0, &one).ok(), "tp1");
    CHECK_EQ(one.qkv_band.offset, 0);
    CHECK_EQ(one.head_band.offset, 0);
}

static void test_kda_q8_0_traffic_divides_by_eight() {
    std::printf("kda_q8_0_traffic_divides_by_eight\n");
    // The claim this shard exists to make good on, in bytes, checkable with no GPU.
    //
    // Under ShardPolicy::ExpertsOnly the KDA projections are REPLICATED, so all
    // eight ranks stream the identical Q8_0 weights every token. These are the two
    // kernels nsys puts at 25.9% of a 221 ms token at ctx 128k on main
    // (proj_q8_0_multirow 17.0% + proj_q8_0_fused4 8.9%).
    auto q8_0_bytes = [](long rows, long cols) -> long {
        return rows * (cols / 32) * 34;      // 34 bytes per 32 weights
    };
    const long H = 7168, QKV = 12288;
    const long full_layer = 4 * q8_0_bytes(QKV, H)      // attn_q/k/v + ssm_g
                          + q8_0_bytes(H, QKV);          // attn_output
    CHECK_EQ(full_layer, 467927040L);                    // 468 MB per KDA layer

    KdaShardDims d;
    CHECK(kda_shard_dims(k3_kda(), cfg_of(8), 5, &d).ok(), "shard");
    const long shard_layer = 4 * q8_0_bytes(d.qkv, H)    // rows split
                           + q8_0_bytes(H, d.qkv);        // cols split
    // Exact, not approximate: 96 heads over 8 ranks leaves no remainder, so no
    // rank reads a byte more than any other.
    CHECK_EQ(shard_layer * 8, full_layer);
    CHECK_EQ(shard_layer, 58490880L);

    const long kda_layers = 69;                          // 93 - 24 MLA
    CHECK_EQ(full_layer * kda_layers, 32286965760L);     // 32.3 GB per rank per token
    CHECK_EQ(shard_layer * kda_layers, 4035870720L);     // 4.04 GB after sharding
}

static void test_kda_collective_accounting() {
    std::printf("kda_collective_accounting\n");
    // K3 today (ExpertsOnly): one all-reduce per MoE layer, at expert_latent.
    CHECK_EQ(k3_reduce_count_per_token(69, 92, false), 92);
    // With KDA sharded: one more per KDA layer, at hidden, after attn_output.
    CHECK_EQ(k3_reduce_count_per_token(69, 92, true), 161);
    // The COST of the change is exactly the KDA layer count — the number that has
    // to be multiplied by the measured per-collective latency to price it.
    CHECK_EQ(k3_reduce_count_per_token(69, 92, true) -
             k3_reduce_count_per_token(69, 92, false), 69);
    // MLA layers contribute none: they stay replicated. 69 + 24 = 93, and the
    // sharded count must not move when the MLA count changes.
    CHECK_EQ(k3_reduce_count_per_token(69, 0, true), 69);
    CHECK_EQ(k3_reduce_count_per_token(0, 0, true), 0);
    CHECK_EQ(k3_reduce_count_per_token(-1, -1, true), 0);

    // The payload is full hidden width — a col-shard yields a partial sum over the
    // WHOLE output, which is what makes it an all-reduce and not an all-gather.
    ShardDims sd;
    CHECK(shard_dims(kimi_k3(), cfg_of(8), 0, &sd).ok(), "shard");
    CHECK_EQ(reduce_elems(ReducePoint::AfterAttnOut, sd, 1), 7168u);
}

int main() {
    test_tp1_is_the_identity_shard();
    test_qwen36_tp8();
    test_kimi_k3_tp8();
    test_kimi_k3_tp4_for_b300();
    test_shape_must_divide_on_every_axis();
    test_expert_bands_tile_exactly();
    test_vocab_bands_tile_exactly();
    test_indivisible_shapes_are_rejected_with_a_reason();
    test_expert_fallback_to_ffn_width();
    test_bad_rank_and_device_list();
    test_row_shard_slices_are_contiguous_and_tile();
    test_col_shard_is_strided_and_tiles_per_row();
    test_row_and_col_shard_compose();
    test_reduce_accounting();
    test_even_band_edges();

    test_kda_tp8_real_dims();
    test_kda_bands_tile_and_hold_the_invariant();
    test_kda_tp1_is_the_identity_shard();
    test_kda_rejects_what_it_cannot_split();
    test_kda_replicate_and_offset_bands();
    test_kda_q8_0_traffic_divides_by_eight();
    test_kda_collective_accounting();

    std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
