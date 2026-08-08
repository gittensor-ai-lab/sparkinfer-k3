#!/usr/bin/env bash
# Capture the llama.cpp reference logits for the DEPTH-SWEEP parity suite.
#
# The correctness gate used to be one 4-token prompt graded against one 640 KB .spkl:
# a single next-token distribution, n=1, with the KV cache essentially empty -- while the
# engine is scored at 131,072 tokens. This captures the same kind of reference at
# 128…32768 tokens, as nested prefixes of one document, so parity is established across
# four orders of magnitude of context instead of at one point.
#
# Run this ONCE per (model, quant, llama.cpp pin). The outputs are committed as provenance
# — bench/refdata/longctx.ctx<L>.spkl / .ids — and the evaluator compares against the
# controller's copy, which the box never receives.
#
#   capture_parity_refs.sh <llama.cpp-dir> <model.gguf> [out-prefix] [depths-csv]
#
# NEEDS GPU. dump_ref_logits defaults to CPU + mmap, which pages the whole 554 GB per
# ubatch; a 32k prefill that way is effectively unbounded. --ngl 999 is passed below, so
# the llama.cpp build must have CUDA on and the GPUs must be free.
set -euo pipefail

LLAMA="${1:?llama.cpp source+build dir on the K3 branch}"
MODEL="${2:?first-shard gguf}"
OUT="${3:-$(dirname "$0")/../refdata/longctx}"
DEPTHS="${4:-128,256,512,1024,2048,4096,8192,16384,32768}"
CORPUS="${CORPUS:-$(dirname "$0")/../refdata/longctx.txt}"
NGL="${NGL:-999}"

[[ -f "$CORPUS" ]] || { echo "no corpus at $CORPUS" >&2; exit 2; }

INC="$LLAMA/include"
LIB="$LLAMA/build/bin"
[[ -f "$INC/llama.h" ]] || { echo "no llama.h at $INC" >&2; exit 2; }

# ggml's headers moved around between llama.cpp revisions; include both roots so this
# builds against the pinned commit without needing to know which layout it uses.
g++ -std=c++17 -O2 -I "$INC" -I "$LLAMA/ggml/include" \
    "$(dirname "$0")/dump_ref_logits.cpp" \
    -L "$LIB" -lllama -lggml -lggml-base -o /tmp/dump_ref_logits

echo "capturing depths $DEPTHS from $(wc -c < "$CORPUS") bytes of corpus, ngl=$NGL"
LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}" /tmp/dump_ref_logits \
    "$MODEL" "$OUT" "@$CORPUS" --ngl "$NGL" --prefixes "$DEPTHS"

echo
echo "captured:"
ls -la "$(dirname "$OUT")"/"$(basename "$OUT")".ctx*.spkl

cat <<NOTE

These .spkl files are the ANSWER KEY. They are compared on the controller and are removed
from the evaluation box before every build (k3_eval_bot._box_build), because a bench that
can read them can echo one back as its own --logits dump and score top1 1.0 / KL 0.0 while
computing nothing. The .ctx<L>.ids alongside them are inputs, are public, and DO ship to
the box — a 32k-token prompt is ~200 KB, far past MAX_ARG_STRLEN, so it cannot be passed
as an argument and is read from the box's own origin/main checkout instead.
NOTE
