// Front door for the banded LM head.
//
// The head is the last thing a token does and the only thing in the model that ONE
// rank does alone: kimi_k3_tp.cpp runs output_norm + the 163840 x 7168 projection on
// rank 0 while the other seven H200s hold the identical hidden state and nothing to
// do with it. At Q8_0 that projection reads (7168/32) * 34 * 163840 = 1.25 GB, which
// is 45x the largest per-layer projection in the model, and it is fully exposed —
// the driver blocks on rank 0's stream immediately afterwards with nothing else in
// flight, so every microsecond of it is a microsecond of the token.
//
// Every rank already HOLDS the weight. weight_plan.cpp declares output.weight
// RowShard, but upload_sliced's short-circuit only consults the rule table for the
// expert, KDA and MLA stacks, so the head short-circuits to a full replica on all
// eight devices — VRAM they pay for and never use. Banding it is therefore a pointer
// offset and a smaller N, with no loader change, no new collective and no change to
// which numbers are computed.
//
// This header is deliberately arithmetic only: it owns the toggle and the decline
// rules and nothing else, so the band a rank projects and the band the host copies
// back can never be derived twice and disagree.
#pragma once

#include <cstddef>

namespace sparkinfer {

// Rows [offset, offset + rows) of output.weight, and the byte offset of the first of
// them. `row_bytes` is derived from the tensor rather than from its quant type: the
// head is Q8_0 today (34 B per 32 weights) and F32 in an unquantised build, and
// n_bytes / vocab is exact for both without this file having to know either.
struct K3HeadBand {
    int    offset    = 0;
    int    rows      = 0;
    size_t byte_off  = 0;
};

// SPARKINFER_K3_HEAD_BAND=0 declines on the same binary. Default ON — the harness
// scores a default build.
bool k3_head_band_enabled();

// Fills `out` and returns true when the band is usable. Declines — leaving the
// rank-0 head reachable — unless the toggle is on, tp_size > 1 divides vocab, the
// rank is in range, and the weight's byte count divides evenly into rows. A head
// whose n_bytes is not a multiple of vocab would give a fractional row stride, and
// a pointer built from a rounded one reads the wrong weights at full speed.
bool k3_head_band(int vocab, size_t weight_bytes, int tp_size, int rank,
                  K3HeadBand* out);

}  // namespace sparkinfer
