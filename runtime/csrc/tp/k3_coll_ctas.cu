#include "sparkinfer/tp/k3_coll_ctas.h"

#include <cstdlib>

namespace sparkinfer::tp {

namespace {

// 0 = off (keep 512). 32/64/128/256/512 = pin. Anything else, including unset, = auto.
int coll_cta_env() {
    static const int v = [] {
        const char* e = std::getenv("SPARKINFER_TP_COLL_CTA");
        if (!e || !e[0]) return -1;
        const int n = std::atoi(e);
        if (n == 0 || n == 32 || n == 64 || n == 128 || n == 256 || n == 512) return n;
        return -1;
    }();
    return v;
}

}  // namespace

int k3_coll_block_for(int n_vec, int max_blocks) {
    const int env = coll_cta_env();
    if (env == 0) return 512;         // shipped geometry, on the same binary
    if (env > 0) return env;          // pinned for an ablation

    if (n_vec <= 0 || max_blocks <= 0) return 512;

    // Smallest CTA (most SMs) that still keeps every CTA a full warp AND does not need
    // more flag slots than Signal carries. Walking UP from one warp stops at the
    // first width whose grid fits, so the clamp in grid_for is never what decides the
    // geometry — a clamped grid means CTAs beyond max_blocks would have been dropped
    // onto the grid-stride tail, which is correct but re-serialises what this is
    // trying to spread.
    static const int kWidths[] = {32, 64, 128, 256, 512};
    for (int b : kWidths) {
        const long grid = ((long)n_vec + b - 1) / b;
        if (grid <= (long)max_blocks) return b;
    }
    return 512;
}

}  // namespace sparkinfer::tp
