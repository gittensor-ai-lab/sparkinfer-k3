#include "sparkinfer/models/k3_head_band.h"

#include <cstdlib>

namespace sparkinfer {

bool k3_head_band_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SPARKINFER_K3_HEAD_BAND");
        return !(e && e[0] == '0');
    }();
    return on;
}

bool k3_head_band(int vocab, size_t weight_bytes, int tp_size, int rank,
                  K3HeadBand* out) {
    if (!out) return false;
    if (!k3_head_band_enabled()) return false;
    if (tp_size <= 1 || rank < 0 || rank >= tp_size) return false;
    if (vocab <= 0 || vocab % tp_size != 0) return false;
    if (weight_bytes == 0 || weight_bytes % (size_t)vocab != 0) return false;

    const int rows = vocab / tp_size;
    // The rows of a GGUF 2-D tensor are contiguous and equal-sized, so the band is
    // one memory range and the offset is exact integer arithmetic. Both facts are
    // checked above rather than assumed, because getting either wrong reads real
    // bytes belonging to other rows and produces a fluent wrong token.
    out->offset   = rank * rows;
    out->rows     = rows;
    out->byte_off = (size_t)out->offset * (weight_bytes / (size_t)vocab);
    return true;
}

}  // namespace sparkinfer
