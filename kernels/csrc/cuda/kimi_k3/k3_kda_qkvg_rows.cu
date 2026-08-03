#include "sparkinfer/kernels/k3_kda_qkvg_rows.h"

#include <cstdlib>

namespace sparkinfer::kernels::k3 {

namespace {

// The same 1792-warp target k3_proj_rowbudget.cu uses — 448 CTAs of 128 threads,
// where the projection family measures 1.8 TB/s on an 8x H200. Repeated as a
// constant rather than shared because the two files answer different questions:
// that one widens where nothing is traded, this one buys warps with an activation
// re-read and has to be allowed to lose on its own.
constexpr int kTargetWarps = 1792;

// 0 = off (keep the shipped ROWS). 1/2/4 = pin that tier for an ablation. Anything
// else, including unset, = let the warp budget below choose.
int qkvg_rows_env() {
    static const int v = [] {
        const char* e = std::getenv("SPARKINFER_K3_KDA_QKVG_ROWS");
        if (!e || !e[0]) return -1;
        const int n = std::atoi(e);
        return (n == 0 || n == 1 || n == 2 || n == 4) ? n : -1;
    }();
    return v;
}

}  // namespace

int k3_kda_qkvg_rows_for_budget(int N, int block_threads, int legacy_rows) {
    const int env = qkvg_rows_env();
    if (env == 0) return legacy_rows;                          // shipped geometry
    if (env > 0) return env < legacy_rows ? env : legacy_rows;  // pinned tier
    if (N <= 0 || block_threads < 32) return legacy_rows;

    const int warps_per_cta = block_threads / 32;
    // Largest ROWS whose grid still meets the budget, walking down from the most
    // activation reuse to the least and stopping at the first that fills the
    // machine. Only ever LOWERS the shipped ROWS: the trade being tested is
    // "more warps for more activation re-read", never the reverse.
    static const int kLadder[] = {4, 2, 1};
    for (int r : kLadder) {
        const long grid  = ((long)N + r - 1) / r;
        const long warps = grid * warps_per_cta;
        if (warps >= (long)kTargetWarps) return r < legacy_rows ? r : legacy_rows;
    }
    return 1 < legacy_rows ? 1 : legacy_rows;
}

}  // namespace sparkinfer::kernels::k3
