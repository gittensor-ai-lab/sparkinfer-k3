#include "sparkinfer/models/k3_moe_band.h"

#include <cstdlib>

namespace sparkinfer {

bool k3_moe_band_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_MOE_BAND");
        return !(e && e[0] == '0');
    }();
    return on;
}

bool k3_moe_band_router_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_MOE_BAND_ROUTER");
        return !(e && e[0] == '0');
    }();
    return on;
}

bool k3_moe_band_active(int n_experts, int expert_latent, int hidden, int tp_size) {
    return k3_moe_band_enabled() && tp_size > 1 &&
           n_experts > 0 && expert_latent > 0 && hidden > 0 &&
           n_experts % tp_size == 0 &&
           expert_latent % tp_size == 0 &&
           hidden % tp_size == 0;
}

bool k3_moe_band(int rows_total, size_t weight_bytes, int tp_size, int rank,
                 K3MoeBand* out) {
    if (!out) return false;
    if (!k3_moe_band_enabled()) return false;
    if (tp_size <= 1 || rank < 0 || rank >= tp_size) return false;
    if (rows_total <= 0 || rows_total % tp_size != 0) return false;
    if (weight_bytes == 0 || weight_bytes % (size_t)rows_total != 0) return false;

    const int rows = rows_total / tp_size;
    // The rows of a GGUF 2-D tensor are contiguous and equal-sized, so the band is
    // one memory range and the offset is exact integer arithmetic. Both facts are
    // checked above rather than assumed: a stride built from a rounded
    // weight_bytes / rows_total reads real bytes belonging to neighbouring rows at
    // full speed, which is fluent and wrong rather than slow or crashing.
    out->offset   = rank * rows;
    out->rows     = rows;
    out->byte_off = (size_t)out->offset * (weight_bytes / (size_t)rows_total);
    return true;
}

}  // namespace sparkinfer
