#!/usr/bin/env bash
# Kimi K3 EVAL — score sparkinfer against the llama.cpp reference and emit the label.
#
#   bench/scripts/kimi_k3_eval.sh [--node h200x8] [--devices 0,1,..] [--frontier TOKS]
#                                 [--llama-ref TOKS] [--layers N] [--json-out FILE]
#
# Produces the same RESULT_JSON contract the eval loop already consumes for the Qwen
# models (bench/scripts/label.py), so the K3 track needs no second scoring path.
#
# THE THREE INPUTS label.py NEEDS, AND WHERE EACH COMES FROM
#
#   value      sparkinfer decode tok/s, measured here by kimi_k3_tp_bench.
#   frontier   the best VERIFIED sparkinfer tok/s so far. 0 => the run is the first
#              on this node/quant and labels BASELINE. Pass --frontier to score a PR
#              against the merged best.
#   DIFF_REF   the llama.cpp reference tok/s — the TIER BASIS. label.py sizes a gain
#              against llama.cpp rather than against the frontier so the same tok/s of
#              real work earns the same tier whether the frontier is fast or slow.
#              Read from reference.lock (KIMI_K3_<NODE>_LLAMA_128) unless --llama-ref.
#
# plus the correctness gate: top-1 agreement and mean KLD against llama.cpp on the
# SAME weights and the SAME ids, from bench/refdata/. label.py REJECTs below
# top1 0.90 / KLD 0.20 no matter how fast the run was — a speedup that erodes parity
# with the reference is not a speedup worth having.
#
# WHY THE REFERENCE IS A CAPTURE AND NOT A LIVE llama.cpp RUN. Re-running llama.cpp
# for every eval would mean loading 553 GiB twice per PR. bench/refdata/*.spkl is the
# reference's own output on pinned weights at a pinned commit, so the comparison is
# identical and costs one sparkinfer forward.
#
# Env: KIMI_K3_MODEL (or KIMI_K3_MODELS_DIR) · SPARKINFER_BUILD · KIMI_K3_NODE

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

NODE="${KIMI_K3_NODE:-h200x8}"
DEVICES="0,1,2,3,4,5,6,7"
FRONTIER=""          # empty => read the lock's sparkinfer slot; 0 => BASELINE
LLAMA_REF=""         # empty => read the lock
LAYERS=93
TOKENS=8
JSON_OUT=""

while [[ $# -gt 0 ]]; do case "$1" in
  --node)      shift; NODE="$1" ;;
  --devices)   shift; DEVICES="$1" ;;
  --frontier)  shift; FRONTIER="$1" ;;
  --llama-ref) shift; LLAMA_REF="$1" ;;
  --layers)    shift; LAYERS="$1" ;;
  --tokens)    shift; TOKENS="$1" ;;
  --json-out)  shift; JSON_OUT="$1" ;;
  -h|--help)   sed -n '2,31p' "$0"; exit 0 ;;
  *) echo "kimi_k3_eval: unknown arg $1" >&2; exit 2 ;;
esac; shift; done

BUILD="${SPARKINFER_BUILD:-$ROOT/build}"
BENCH="$BUILD/kimi_k3_tp_bench"
[[ -x "$BENCH" ]] || { echo "kimi_k3_eval: $BENCH not built" >&2; exit 2; }

MODEL="${KIMI_K3_MODEL:-}"
if [[ -z "$MODEL" && -n "${KIMI_K3_MODELS_DIR:-}" ]]; then
    MODEL="$(ls "$KIMI_K3_MODELS_DIR"/*/*-00001-of-*.gguf 2>/dev/null | head -1 || true)"
fi
[[ -f "$MODEL" ]] || { echo "kimi_k3_eval: set KIMI_K3_MODEL to the first GGUF shard" >&2; exit 2; }

# reference.lock is the pinned source for both the llama reference and the current
# frontier. Sourcing it (rather than hardcoding) is what makes M1/M2/M3 independent.
# shellcheck source=/dev/null
[[ -f "$HERE/reference.lock" ]] && source "$HERE/reference.lock"
PFX="KIMI_K3_$(printf '%s' "$NODE" | tr 'a-z' 'A-Z')"
[[ -n "$LLAMA_REF" ]] || LLAMA_REF="$(eval "printf '%s' \"\${${PFX}_LLAMA_128:-0}\"")"
[[ -n "$FRONTIER"  ]] || FRONTIER="$(eval "printf '%s' \"\${${PFX}_SPARKINFER_128:-0}\"")"

echo ">> Kimi K3 eval — node=$NODE devices=$DEVICES layers=$LAYERS"
echo ">> llama.cpp reference (tier basis): $LLAMA_REF tok/s"
echo ">> sparkinfer frontier             : $FRONTIER tok/s$([[ "$FRONTIER" == "0" ]] && echo '  (none yet -> BASELINE)')"

# ---- 1. speed -------------------------------------------------------------
NDEV="$(printf '%s' "$DEVICES" | tr ',' '\n' | grep -c .)"
echo ">> measuring sparkinfer decode ..."
SPEED_OUT="$("$BENCH" "$MODEL" "$NDEV" "$LAYERS" "$TOKENS" 2>&1)" || {
    echo "$SPEED_OUT" | tail -20 >&2; echo "kimi_k3_eval: bench failed" >&2; exit 1; }
TPS="$(printf '%s' "$SPEED_OUT" | sed -n 's/.*ms\/token[^(]*(\([0-9.]*\) tok\/s).*/\1/p' | head -1)"
MSTOK="$(printf '%s' "$SPEED_OUT" | sed -n 's/.*ms\/token *\([0-9.]*\).*/\1/p' | head -1)"
[[ -n "$TPS" ]] || { echo "$SPEED_OUT" | tail -20 >&2; echo "kimi_k3_eval: could not parse tok/s" >&2; exit 1; }
echo ">> sparkinfer: $TPS tok/s ($MSTOK ms/token)"

# ---- 2. correctness -------------------------------------------------------
# Same ids llama.cpp was captured on. Fed verbatim, no BOS, no chat template — both
# sides must see the identical token sequence or the comparison means nothing.
REF_SPKL="$ROOT/bench/refdata/hello.spkl"
REF_IDS_FILE="$ROOT/bench/refdata/hello.ids"
if [[ -f "$REF_SPKL" && -f "$REF_IDS_FILE" ]]; then
    # hello.ids holds prompt ids THEN the reference continuations; the prompt is the
    # first 4. Taking the whole file would score a different step.
    IDS_CSV="$(head -4 "$REF_IDS_FILE" | paste -sd, -)"
    OURS="$(mktemp -t k3eval_XXXXXX.spkl)"
    trap 'rm -f "$OURS"' EXIT
    echo ">> measuring accuracy vs llama.cpp on ids $IDS_CSV ..."
    "$BENCH" "$MODEL" "$NDEV" "$LAYERS" 1 --ids "$IDS_CSV" --logits "$OURS" >/dev/null 2>&1 || true
    if [[ -s "$OURS" ]]; then
        ACC_JSON="$(python3 "$HERE/compare_logits.py" "$REF_SPKL" "$OURS" --json 2>/dev/null || true)"
    fi
fi
if [[ -z "${ACC_JSON:-}" ]]; then
    echo "kimi_k3_eval: could not measure accuracy (missing bench/refdata or logits dump)" >&2
    echo "  refusing to score: label.py's correctness gate is not optional, and passing" >&2
    echo "  a fabricated top1/kl would turn a REJECT into a merge." >&2
    exit 1
fi
TOP1="$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["top1_agreement"])' "$ACC_JSON")"
KL="$(python3 -c 'import json,sys;print(json.loads(sys.argv[1])["mean_kld"])' "$ACC_JSON")"
echo ">> accuracy: top1=$TOP1  mean_kld=$KL"

# ---- 3. label -------------------------------------------------------------
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
PROV="$(python3 - "$NODE" "$DEVICES" "$LAYERS" "$MODEL" "$MSTOK" <<'PY'
import json, sys, os
node, devs, layers, model, mstok = sys.argv[1:6]
print(json.dumps({
    "node": node, "devices": devs, "layers": int(layers),
    "quant": os.path.basename(os.path.dirname(model)),
    "ms_per_token": float(mstok) if mstok else None,
    "engine": "sparkinfer/kimi_k3_tp_bench",
    "llama_commit": os.environ.get("KIMI_K3_LLAMACPP_COMMIT", ""),
}, separators=(",", ":")))
PY
)"

# DIFF_REF is the tier basis. label.py falls back to delta/frontier when it is <= 0,
# which for K3 would mean scoring against an unmeasured reference — so say so loudly
# rather than silently emitting a tier computed off the wrong denominator.
if [[ "${LLAMA_REF%%.*}" == "0" ]]; then
    echo ">> WARN: no llama.cpp reference pinned for $NODE — the label will fall back to" >&2
    echo ">>       delta/frontier, which is NOT the llama-anchored tier this repo scores on." >&2
    echo ">>       Run: bench/scripts/kimi_k3_baseline.sh --node $NODE --decode-only" >&2
fi

RESULT="$(SPARKINFER_DIFFICULTY_REF="$LLAMA_REF" \
    python3 "$HERE/label.py" "$TPS" "$FRONTIER" 0 "$TOP1" "$KL" "$COMMIT" "$PROV")"
echo
echo "$RESULT"
if [[ -n "$JSON_OUT" ]]; then
    printf '%s\n' "${RESULT#RESULT_JSON }" > "$JSON_OUT"
    echo ">> wrote $JSON_OUT"
fi
