#!/usr/bin/env bash
# Capture llama.cpp reference logits + tokenization for a K3 model, as the ground
# truth sparkinfer's executor is validated against. See dump_ref_logits.cpp.
#
# llama.cpp K3 support lives on the unslothai/llama.cpp branch kimi-k3-fullsize-vision
# (commit efc8bc38 — the same the kernels transcribe). master does NOT have it
# ("unknown model architecture: 'kimi-k3'"), so the branch must be checked out and
# llama.cpp rebuilt before this works.
#
# Runs on CPU + mmap so it fits one box: the 594 GB weights page from disk rather
# than needing 594 GB of RAM/VRAM. That makes it SLOW (a forward pass reads the whole
# model from disk), so keep the prompt short — a few reference logit vectors is all
# the acceptance test needs.
#
# Usage: dump_ref_logits.sh <llama.cpp-dir> <model.gguf> <out-prefix> "<prompt>" [n_predict]
set -euo pipefail
LLAMA="${1:?llama.cpp source+build dir on the K3 branch}"
MODEL="${2:?first-shard gguf}"
OUT="${3:?output prefix, e.g. bench/refdata/hello}"
PROMPT="${4:?prompt text}"
NPRED="${5:-0}"

INC="$LLAMA/include"
LIB="$LLAMA/build/bin"
[[ -f "$INC/llama.h" ]] || { echo "no llama.h at $INC" >&2; exit 2; }

g++ -std=c++17 -O2 -I "$INC" "$(dirname "$0")/dump_ref_logits.cpp" \
    -L "$LIB" -lllama -lggml -lggml-base -o /tmp/dump_ref_logits
LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}" /tmp/dump_ref_logits "$MODEL" "$OUT" "$PROMPT" "$NPRED"

echo
echo "Reference captured. To validate sparkinfer against it:"
echo "  1. feed the SAME ids to the executor:"
echo "       kimi_k3_generate <model.gguf> 1 \$(paste -sd' ' $OUT.ids) --logits ours.spkl"
echo "  2. compare:"
echo "       python bench/scripts/compare_logits.py $OUT.spkl ours.spkl"
echo "  Against llama.cpp on the IDENTICAL quant, agreement should be near-exact"
echo "  (mean KLD < 1e-5) — anything more is a bug in the executor, not the quant."
