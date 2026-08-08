#include "sparkinfer/kernels/k3_proj_rowbudget.h"

#include <cstdlib>

namespace sparkinfer::kernels::k3 {

namespace {

// 1792 warps = 448 CTAs of 128 threads, which is where ffn_routed_down and
// ffn_routed_up already sit and where this kernel family measures 1.8 TB/s on an
// 8x H200. It is 13.6 warps per SM over 132 SMs — deep enough that a CTA's misses
// are covered by other CTAs rather than by nothing. Not a tuning constant pulled
// from the air: it is the operating point of the two shapes in this exact kernel
// that already reach bandwidth, used as the target for the four that do not.
// SPARKINFER_K3_WARP_TARGET overrides the 1792 default: it was the
// best-OBSERVED operating point, never a measured optimum. Raising it lowers
// ROWS (fewer loads/thread, more CTAs) — also the decisive H1-vs-H3 experiment.
static int target_warps() {
    static const int v = [] {
        const char* e = std::getenv("SPARKINFER_K3_WARP_TARGET");
        const int n = e ? atoi(e) : 0;
        // MEASURED: 3584 (ROWS halved, more resident CTAs) beats the shipped
        // 1792 at the scored shape; 7168 regresses (one shape crosses a wave).
        return n > 0 ? n : 3584;
    }();
    return v;
}

bool rowbudget_on() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_PROJ_ROWBUDGET");
        return !(e && e[0] == '0');
    }();
    return on;
}

}  // namespace

int k3_proj_rows_for_budget(int N, int block_threads, int legacy_rows) {
    if (!rowbudget_on()) return legacy_rows;
    if (N <= 0 || block_threads < 32) return legacy_rows;

    const int warps_per_cta = block_threads / 32;
    // Largest ROWS whose grid still meets the budget. Warp count falls as ROWS
    // rises, so this walks down from the widest row-blocking to the narrowest and
    // stops at the first one that fills the machine — the most activation reuse
    // that does not cost occupancy.
    static const int kLadder[] = {16, 8, 4, 2, 1};
    for (int r : kLadder) {
        const long grid  = ((long)N + r - 1) / r;
        const long warps = grid * warps_per_cta;
        if (warps >= (long)target_warps()) {
            // Never widen a CTA's row block beyond what the shipped tier chose.
            // The budget can only ask for MORE parallelism here; a shape that the
            // old rule already ran narrow (ssm_f_b at 1536 rows cannot reach 1792
            // warps at any ROWS) must not be pushed the other way by a rule that
            // happens to be satisfied at a higher ROWS on some future geometry.
            // WAVE FLOOR: a target that pushes the grid past ONE wave costs
            // more than the warps it buys — ffn_down_shexp at target 7168 went
            // to 7168 one-warp CTAs = 1.70 waves against the 32-CTA/SM cap and
            // 100% activation re-read, and ate the gains the other shapes made.
            // Also floor ROWS at 2: re-read is exactly 1/ROWS.
            const long cap = 132L * (warps_per_cta >= 2 ? 32 / warps_per_cta : 32);
            if (grid > cap && r < legacy_rows) continue;
            if (r < 2 && legacy_rows >= 2) continue;
            return r < legacy_rows ? r : legacy_rows;
        }
    }
    // Nothing reaches the budget — take the widest grid available, which is what
    // the machine is short of. Still capped by the shipped tier for the same
    // reason as above.
    return 1 < legacy_rows ? 1 : legacy_rows;
}

}  // namespace sparkinfer::kernels::k3
