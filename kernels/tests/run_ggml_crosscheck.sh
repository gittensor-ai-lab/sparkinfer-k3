#!/usr/bin/env bash
# Cross-check sparkinfer's quant decoders against REAL ggml.
#
# Not wired into ctest: it needs a llama.cpp/ggml source tree, which CI does not carry.
# Run it whenever a decoder or a lattice table changes.
#
# The IQ1_S check needs NO GPU — it compares the kernel's flat-index arithmetic
# (one thread per 8-value group) against ggml's nested-loop reference on the CPU. That
# is where the risk is: the lattice tables are proven identical by mechanical
# extraction, but the index derivation is hand-written.
#
# Usage: run_ggml_crosscheck.sh /path/to/llama.cpp
set -euo pipefail
LLAMA="${1:-}"
if [[ -z "$LLAMA" || ! -f "$LLAMA/ggml/src/ggml-quants.c" ]]; then
  echo "usage: $0 /path/to/llama.cpp   (needs ggml/src/ggml-quants.c)" >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo ">> compiling ggml-quants.c"
gcc -std=c11 -O2 -I "$LLAMA/ggml/include" -I "$LLAMA/ggml/src" \
    -c "$LLAMA/ggml/src/ggml-quants.c" -o "$OUT/ggml-quants.o"
gcc -std=c11 -O2 -c "$HERE/ggml_quantize_stubs.c" -o "$OUT/stubs.o"

echo ">> IQ1_S: kernel index arithmetic vs ggml dequantize_row_iq1_s"
g++ -std=c++17 -O2 -I "$ROOT/kernels/include" -I "$LLAMA/ggml/include" \
    "$HERE/kimi_k3_iq1s_ggml_cpu_test.cpp" "$OUT/ggml-quants.o" "$OUT/stubs.o" \
    -lm -o "$OUT/iq1s"
"$OUT/iq1s"

echo
echo ">> IQ2_XS (GPU): build separately, needs libggml-base.so + a device"
echo "   see kernels/tests/kimi_k3_iq2xs_ggml_test.cu"
