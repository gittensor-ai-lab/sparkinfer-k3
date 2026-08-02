#!/usr/bin/env bash
# Kimi K3 end to end on one multi-GPU node: TEXT IN, TEXT OUT.
#
#   bench/scripts/kimi_k3_run.sh "Hello, world!" [n_predict] [devices]
#
#   bench/scripts/kimi_k3_run.sh "The capital of France is" 32 0,1,2,3,4,5,6,7
#
# The three stages, and why each is what it is:
#
#   1. TEXT -> IDS via llama.cpp's llama-tokenize, against THE GGUF'S OWN VOCAB.
#      K3 ships a tiktoken BPE (tiktoken.model + tokenization_kimi.py), NOT a
#      tokenizer.json, so `tokenizers.Tokenizer.from_file` cannot read it and a
#      side-loaded tokenizer would silently produce different ids. The GGUF carries
#      the vocab llama.cpp itself used, which makes this the only encoder that is
#      correct by construction. llama-tokenize opens the model vocab_only, so it
#      costs metadata, not 553 GiB.
#
#   2. IDS -> IDS via sparkinfer's kimi_k3_generate across `devices`. Greedy decode.
#
#   3. IDS -> TEXT via sparkinfer's own byte-level BPE detokenizer, built from
#      tokenizer.ggml.tokens in the same GGUF. Printed by kimi_k3_generate.
#
# NOTE ON THE DECODE PATH. kimi_k3_generate uses the LAYER-SPLIT PIPELINE, so each
# GPU owns a contiguous range of layers. That is the path with a generation loop and
# detokenization wired up. The tensor-parallel path (kimi_k3_tp_bench) is faster per
# token but is a decode benchmark, not a generator — it has no sampling loop. Both
# produce the same logits (verified: mean KLD 1.85e-09, top-1 100%).
#
# Env:
#   KIMI_K3_MODEL     first shard of the GGUF   (required unless $1 is a path)
#   LLAMA_TOKENIZE    path to llama-tokenize    (default: search the usual places)
#   SPARKINFER_BUILD  sparkinfer build dir      (default: ./build)

set -euo pipefail

PROMPT="${1:-Hello, world!}"
NPREDICT="${2:-32}"
DEVICES="${3:-0}"

MODEL="${KIMI_K3_MODEL:-}"
if [[ -z "$MODEL" ]]; then
    echo "kimi_k3_run: set KIMI_K3_MODEL to the first GGUF shard" >&2
    exit 2
fi
if [[ ! -f "$MODEL" ]]; then
    echo "kimi_k3_run: no such model: $MODEL" >&2
    exit 2
fi

BUILD="${SPARKINFER_BUILD:-build}"
GEN="$BUILD/kimi_k3_generate"
if [[ ! -x "$GEN" ]]; then
    echo "kimi_k3_run: $GEN not built. cmake --build $BUILD --target kimi_k3_generate" >&2
    exit 2
fi

# Find llama-tokenize. Not vendored: it is llama.cpp's, and the baseline harness
# already builds a pinned llama.cpp into .llamacpp-k3/ for the reference sweep.
TOK="${LLAMA_TOKENIZE:-}"
if [[ -z "$TOK" ]]; then
    for c in .llamacpp-k3/build/bin/llama-tokenize \
             ../llama.cpp/build/bin/llama-tokenize \
             /workspace/llama.cpp/build/bin/llama-tokenize \
             "$(command -v llama-tokenize || true)"; do
        [[ -n "$c" && -x "$c" ]] && { TOK="$c"; break; }
    done
fi
if [[ -z "$TOK" || ! -x "$TOK" ]]; then
    cat >&2 <<EOF
kimi_k3_run: llama-tokenize not found.

  K3's vocab is a tiktoken BPE, so the GGUF's own vocab is the only correct
  encoder. Build it from the K3-supporting llama.cpp:

    cmake --build <llama.cpp>/build -j --target llama-tokenize
    LLAMA_TOKENIZE=<llama.cpp>/build/bin/llama-tokenize $0 ...
EOF
    exit 2
fi

# --no-bos: kimi_k3_generate feeds the ids verbatim and adds nothing, so a BOS here
# would be a token the reference never saw. Matches how bench/refdata was captured.
# llama-tokenize is noisy on stderr even when it succeeds (model metadata), so its
# output is captured rather than shown. Capture rather than discard: under
# `set -o pipefail` a non-zero llama-tokenize aborts the script at this assignment,
# so the "produced no ids" check below never runs — with stderr sent to /dev/null
# that surfaced as a bare exit code and no message at all.
TOK_LOG="$(mktemp)"
trap 'rm -f "$TOK_LOG"' EXIT
if ! IDS_RAW="$(printf '%s' "$PROMPT" | "$TOK" -m "$MODEL" --stdin --ids --no-bos 2>"$TOK_LOG" | tail -1)"; then
    echo "kimi_k3_run: llama-tokenize failed on $MODEL" >&2
    [[ -s "$TOK_LOG" ]] && sed 's/^/  /' "$TOK_LOG" >&2
    exit 1
fi
# This script ends in `exec`, which replaces the process without running the EXIT
# trap — drop the log explicitly now that it is no longer needed.
rm -f "$TOK_LOG"
trap - EXIT
IDS="$(printf '%s' "$IDS_RAW" | tr -d '[]' | tr ',' ' ' | tr -s ' ')"

if [[ -z "${IDS// /}" ]]; then
    echo "kimi_k3_run: tokenizer produced no ids for: $PROMPT" >&2
    exit 1
fi

echo "prompt : $PROMPT"
echo "ids    : $IDS"
echo "devices: $DEVICES   predict: $NPREDICT"
echo

# shellcheck disable=SC2086
exec "$GEN" "$MODEL" "$NPREDICT" $IDS --devices "$DEVICES"
