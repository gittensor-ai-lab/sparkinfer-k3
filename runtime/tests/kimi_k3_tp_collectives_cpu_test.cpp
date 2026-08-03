// CPU-only tests for the expected TP collective count (no GPU needed).
//
// The count is a correctness gate — kimi_k3_tp_check fails a run whose reduce count does
// not match — so a wrong expectation costs one of two ways, and both have already happened:
//
//   too low   a real missing reduce is indistinguishable from the noise, and a partial
//             expert sum reaches the logits looking like a plausible number
//   too high  the gate fails on unmodified main, which is issue #75: 45 collectives seen
//             against 21 expected, with every logit matching the tp=1 reference BITWISE
//
// The second is the more expensive failure in practice. A check that cries wolf on `main`
// gets read as broken and then gets ignored, which quietly disarms the first case too.
//
// Build/run:
//   g++ -std=c++17 -I runtime/include runtime/tests/kimi_k3_tp_collectives_cpu_test.cpp
//       -o /tmp/kimi_k3_tp_collectives_cpu_test && /tmp/kimi_k3_tp_collectives_cpu_test

#include "sparkinfer/models/kimi_k3_tp_collectives.h"

#include <cstdio>
#include <vector>

using namespace sparkinfer;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                                          \
    do {                                                                          \
        ++g_checks;                                                               \
        if (!(cond)) {                                                            \
            std::printf("  FAIL %s:%d: %s\n        ", __FILE__, __LINE__, #cond); \
            std::printf(__VA_ARGS__);                                             \
            std::printf("\n");                                                    \
            ++g_failures;                                                         \
        }                                                                         \
    } while (0)

#define CHECK_EQ(a, b) CHECK((a) == (b), "got %lld want %lld", (long long)(a), (long long)(b))

// The real per-layer type map: period 4, full attention LAST in each group
// (k k k M | k k k M | ...), as recorded in KimiK3Config. Built rather than baked so a
// short slice is expressed the same way the loader would hand one over.
static KimiK3Config k3(int n_layers, int leading_dense = 1) {
    KimiK3Config cfg;
    cfg.n_layers = n_layers;
    cfg.leading_dense = leading_dense;
    cfg.layer_is_kda.assign((size_t)n_layers, 0);
    for (int i = 0; i < n_layers; ++i) cfg.layer_is_kda[(size_t)i] = (i % 4 != 3) ? 1 : 0;
    return cfg;
}

int main() {
    std::printf("kimi_k3_tp_collectives_cpu_test\n");

    // ---- tp=1 reduces nothing, whatever the flags say ----------------------------------
    // Every reduce site in kimi_k3_tp_forward_token is guarded by tp_size > 1. This is the
    // invariant the check's reference run asserts.
    {
        const KimiK3Config cfg = k3(93);
        for (int kda = 0; kda <= 1; ++kda)
            for (int mla = 0; mla <= 1; ++mla) {
                const auto c = kimi_k3_expected_collectives(cfg, 1, kda != 0, mla != 0);
                CHECK_EQ(c.attn, 0);
                CHECK_EQ(c.moe, 0);
                CHECK_EQ(c.per_token(), 0);
            }
        // tp_size 0 / negative are not reachable from the CLI, but the guard is <=, so
        // pin it: an unset tp_size must not be read as "shard everything".
        CHECK_EQ(kimi_k3_expected_collectives(cfg, 0, true, true).per_token(), 0);
    }

    // ---- the full model, one policy at a time ------------------------------------------
    {
        const KimiK3Config cfg = k3(93);              // 93 layers, leading_dense 1
        const long moe = 92;                          // 93 - 1
        const long mla_layers = 23;                   // i % 4 == 3 over the map built above
        const long kda_layers = 93 - mla_layers;      // 70

        // ExpertsOnly — the policy the old formula described. Still reachable with
        // SPARKINFER_K3_SHARD_KDA=0 SPARKINFER_K3_SHARD_MLA=0, which is the configuration
        // #75 used to make the check pass.
        const auto only = kimi_k3_expected_collectives(cfg, 8, false, false);
        CHECK_EQ(only.attn, 0);
        CHECK_EQ(only.moe, moe);
        CHECK_EQ(only.per_token(), 92);

        const auto kda_only = kimi_k3_expected_collectives(cfg, 8, true, false);
        CHECK_EQ(kda_only.attn, kda_layers);
        CHECK_EQ(kda_only.per_token(), kda_layers + moe);

        const auto mla_only = kimi_k3_expected_collectives(cfg, 8, false, true);
        CHECK_EQ(mla_only.attn, mla_layers);
        CHECK_EQ(mla_only.per_token(), mla_layers + moe);

        // Both bands — today's default. 185/token is the number the 2-D MoE sharding work
        // quotes for this shape, and it is the one this repo actually runs.
        const auto both = kimi_k3_expected_collectives(cfg, 8, true, true);
        CHECK_EQ(both.attn, 93);
        CHECK_EQ(both.moe, moe);
        CHECK_EQ(both.per_token(), 185);

        // Sharding one band and then the other has to add up to sharding both — a layer is
        // either KDA or MLA, never neither and never both.
        CHECK_EQ(kda_only.attn + mla_only.attn, both.attn);
    }

    // ---- the property that the old formula lost ----------------------------------------
    // With both bands sharded, EVERY layer reduces its attention output, so attn == n_layers
    // regardless of how the type map is laid out. Asserted over several depths and a couple
    // of maps so it cannot be satisfied by the period-4 pattern alone.
    {
        for (int n : {1, 2, 3, 4, 8, 17, 93}) {
            const auto c = kimi_k3_expected_collectives(k3(n), 2, true, true);
            CHECK_EQ(c.attn, n);
        }
        KimiK3Config all_kda = k3(11);
        all_kda.layer_is_kda.assign(11, 1);
        CHECK_EQ(kimi_k3_expected_collectives(all_kda, 2, true, true).attn, 11);

        KimiK3Config all_mla = k3(11);
        all_mla.layer_is_kda.assign(11, 0);
        CHECK_EQ(kimi_k3_expected_collectives(all_mla, 2, true, true).attn, 11);

        // A map SHORTER than n_layers must not be read as "shard fewer layers": is_kda_layer
        // reports false past the end, so those layers are MLA to the forward and to us
        // alike. Same source of truth, same answer — that agreement is the whole point.
        KimiK3Config truncated = k3(11);
        truncated.layer_is_kda.resize(4);
        const auto t = kimi_k3_expected_collectives(truncated, 2, true, true);
        CHECK_EQ(t.attn, 11);
        CHECK_EQ(kimi_k3_expected_collectives(truncated, 2, true, false).attn, 3);
    }

    // ---- the exact run reported in issue #75 -------------------------------------------
    // 2x A100, tp=2, first 8 layers, 3 tokens. Measured on hardware: 45 collectives under
    // the default policy, 21 with both bands disabled. The old formula predicted 21 for
    // both, which is why the check failed on a decode that was bitwise correct.
    {
        const KimiK3Config slice = k3(8);
        const int n_tokens = 3;

        const auto def = kimi_k3_expected_collectives(slice, 2, true, true);
        CHECK_EQ(def.attn, 8);
        CHECK_EQ(def.moe, 7);
        CHECK_EQ(def.per_token() * n_tokens, 45);      // matches the observed count

        const auto unsharded = kimi_k3_expected_collectives(slice, 2, false, false);
        CHECK_EQ(unsharded.per_token() * n_tokens, 21);  // matches the second run

        // And the formula that shipped: (n_layers - leading_dense) * n_tokens. Kept as an
        // explicit inequality so nobody reintroduces it by simplification.
        const long old_formula = (long)(slice.n_layers - slice.leading_dense) * n_tokens;
        CHECK(def.per_token() * n_tokens != old_formula,
              "the corrected count must differ from the stale one under the default policy");
    }

    std::printf("%s: %d checks, %d failure(s)\n", g_failures ? "FAIL" : "PASS",
                g_checks, g_failures);
    return g_failures ? 1 : 0;
}
