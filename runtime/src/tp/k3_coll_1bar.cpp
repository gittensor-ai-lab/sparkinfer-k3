#include "sparkinfer/tp/k3_coll_1bar.h"

#include <cstdlib>

namespace sparkinfer::tp {

bool k3_coll_1bar_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_COLL_1BAR");
        return !(e && e[0] == '0');
    }();
    return on;
}

bool k3_coll_1bar_ok(int collectives_per_token, int slots) {
    if (!k3_coll_1bar_enabled()) return false;
    if (slots < 2 || collectives_per_token < 2) return false;
    // Minimum reuse distance of 2, including across the token boundary where the
    // rotation restarts. Within a token consecutive indices always differ for
    // slots >= 2; the wrap is the case that can fail, and it is this one.
    return ((collectives_per_token - 1) % slots) >= 1;
}

}  // namespace sparkinfer::tp
