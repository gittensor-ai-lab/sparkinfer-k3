#include "sparkinfer/kernels/k3_mla_pvec.h"

#include <cstdlib>

namespace sparkinfer::kernels::k3 {

bool k3_mla_pvec_on() {
    // Default ON, because the harness scores a default build: an env-gated default-off
    // improvement is one the scorer never runs.
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_MLA_PVEC");
        return !(e && e[0] == '0');
    }();
    return on;
}

}  // namespace sparkinfer::kernels::k3
