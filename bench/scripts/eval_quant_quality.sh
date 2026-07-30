#!/usr/bin/env bash
# The four numbers in unsloth's quant table — mean KLD, PPL(q), top-1 agreement,
# RMS Δp — all come from ONE tool: llama-perplexity's KL-divergence mode.
#
#   pass 1   llama-perplexity -m BASE  -f corpus --kl-divergence-base logits.dat
#   pass 2   llama-perplexity -m QUANT -f corpus --kl-divergence \
#                                              --kl-divergence-base logits.dat
#
# Pass 2 reports "KL Divergence", "Δp RMS" and "Same top p" (= top-1 agreement).
# PPL on its own needs no baseline: drop both --kl-divergence flags.
#
# READ THIS BEFORE RUNNING IT ON K3
# ---------------------------------
# The baseline must be HIGHER PRECISION than the quant under test. For K3 that means
# UD-Q8_K_XL at ~1560 GB, which does not fit on one RTX PRO 6000 (96 GB) and does not
# fit on 8x H200 (1128 GB) either. So the absolute numbers in unsloth's table are NOT
# reproducible on this hardware, and there is no point trying: we inherit them.
#
# What this script is actually for is the OTHER question. Two different things get
# conflated and only one of them is ours to measure:
#
#   (A) "Is the quant good?"        baseline = a high-precision model.
#                                   Already answered by unsloth. Needs 1.5 TB.
#   (B) "Is our implementation right?"  baseline = llama.cpp ON THE IDENTICAL QUANT.
#                                   Fits on the dev box. This is the one that matters
#                                   while optimizing, and the target is near-EXACT
#                                   agreement, not "close".
#
# For (B), pass the SAME gguf as both BASE and QUANT but run the second pass through
# sparkinfer instead of llama.cpp. Any KLD above numerical noise is then a bug in our
# code, not a property of the quant.
#
# WHY (B) IS THE ONLY USABLE SIGNAL AT IQ1_S. UD-IQ1_S agrees with a high-precision
# reference on only 78.9% of top-1 tokens (mean KLD 0.5645). So roughly one token in
# five legitimately differs from a good model. Output that "looks a bit off" is
# therefore indistinguishable from output produced by a broken kernel — reading
# generations cannot tell you whether the implementation is correct. Exact agreement
# against llama.cpp on the same file, same seed, greedy, can.
#
# Usage: eval_quant_quality.sh <llama.cpp-bin-dir> <quant.gguf> <corpus.txt> [base.gguf]
#        With no base.gguf: PPL only.
set -euo pipefail

BIN="${1:?llama.cpp bin dir}"
QUANT="${2:?quant gguf}"
CORPUS="${3:?corpus text file}"
BASE="${4:-}"

PPL="$BIN/llama-perplexity"
[[ -x "$PPL" ]] || { echo "not executable: $PPL" >&2; exit 2; }
[[ -f "$CORPUS" ]] || { echo "missing corpus: $CORPUS" >&2; exit 2; }

# -ngl 99 offloads everything it can; --no-mmap keeps page-cache thrash out of the
# timing. Context 512 matches the convention these published tables use — a different
# context gives a different PPL, so it is not comparable across settings.
COMMON=(-ngl 99 --no-mmap -c 512 -b 512)

if [[ -n "$BASE" ]]; then
  LOGITS="${LOGITS:-$(dirname "$QUANT")/kld_base_logits.dat}"
  if [[ ! -s "$LOGITS" ]]; then
    echo ">> pass 1/2: dumping baseline logits from $(basename "$BASE")"
    echo "   NOTE: this file is large — roughly n_tokens * n_vocab * 4 bytes."
    echo "   K3's vocab is 163840, so a 100k-token corpus is ~65 GB. Check disk first."
    "$PPL" -m "$BASE" -f "$CORPUS" "${COMMON[@]}" --kl-divergence-base "$LOGITS"
  else
    echo ">> reusing existing baseline logits: $LOGITS"
  fi
  echo
  echo ">> pass 2/2: KLD / top-1 / RMS dp for $(basename "$QUANT")"
  "$PPL" -m "$QUANT" -f "$CORPUS" "${COMMON[@]}" \
         --kl-divergence --kl-divergence-base "$LOGITS"
else
  echo ">> PPL only (no baseline given) for $(basename "$QUANT")"
  "$PPL" -m "$QUANT" -f "$CORPUS" "${COMMON[@]}"
fi
