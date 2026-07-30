#!/usr/bin/env bash
# Kimi K3 evaluation BASELINE on a multi-GPU node (developed against 8x H200 SXM).
#
#   bench/scripts/kimi_k3_baseline.sh [--download] [--decode-only|--prefill-only]
#                                     [--reps N] [--ctx "128 512 4096 32768"] [--dry-run]
#
# Builds the pinned unsloth llama.cpp fork (the only engine that can load a 896-expert
# model — see _kimi_k3.sh), verifies the sharded weights against the pinned manifest,
# then sweeps llama-bench decode (tg) and prefill (pp) at each context and writes
# machine-readable results to bench/results/.
#
# This is the reference half of the eval only. sparkinfer has no kimi-k3 loader yet, so
# there is nothing to compare against on this branch; the point is to produce a pinned,
# reproducible number that a future native implementation is scored against on the SAME
# box with the SAME GGUF.
#
# Env: PRIMARY_QUANT (UD-IQ1_S) · KIMI_K3_MODELS_DIR · LLAMACPP_DIR · ARCH
#      KIMI_K3_NGL / _BATCH / _UBATCH / _SPLIT_MODE / _EXTRA_FLAGS · CUDA_VISIBLE_DEVICES
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$HERE/_common.sh"
# shellcheck source=_kimi_k3.sh
source "$HERE/_kimi_k3.sh"

DOWNLOAD=0; DO_DECODE=1; DO_PREFILL=1; REPS=3; DRY=0
CTXS="${KIMI_K3_CTXS:-128 512 4096 32768}"
DECODE_TOKENS="${KIMI_K3_DECODE_TOKENS:-64}"
while [ $# -gt 0 ]; do case "$1" in
  --download)     DOWNLOAD=1 ;;
  --decode-only)  DO_PREFILL=0 ;;
  --prefill-only) DO_DECODE=0 ;;
  --reps)         shift; REPS="$1" ;;
  --ctx)          shift; CTXS="$1" ;;
  --dry-run)      DRY=1 ;;
  -h|--help)      sed -n '2,17p' "$0"; exit 0 ;;
  *)              echo "!! unknown arg: $1" >&2; exit 2 ;;
esac; shift; done

ARCH="$(detect_arch)"
NGPU="$(gpu_count)"
GGUF="$(kimi_k3_gguf)"
QUANT="$(kimi_k3_quant)"
OUT_DIR="${KIMI_K3_OUT_DIR:-$ROOT/bench/results}"
STAMP="${KIMI_K3_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_JSON="$OUT_DIR/kimi_k3_${QUANT}_baseline_${STAMP}.json"

echo ">> Kimi K3 baseline — quant=$QUANT arch=sm_$ARCH gpus=$NGPU"
echo ">> baseline engine: ${LLAMACPP_REPO} @ ${LLAMACPP_COMMIT:-UNPINNED} (${LLAMACPP_REF:-no ref})"
[ -n "${LLAMACPP_COMMIT:-}" ] || echo ">> WARN: llama.cpp NOT pinned — this run is not reproducible"

if [ "$DRY" = 1 ]; then
  echo ">> dry run: would use"
  echo "   gguf     : $GGUF"
  echo "   mmproj   : $(kimi_k3_mmproj)"
  echo "   manifest : $(kimi_k3_manifest)"
  echo "   flags    : $(kimi_k3_llama_flags)"
  echo "   contexts : $CTXS  (reps=$REPS, n=$DECODE_TOKENS)"
  echo "   out      : $OUT_JSON"
  exit 0
fi

kimi_k3_check_fits

if [ "$DOWNLOAD" = 1 ]; then
  ensure_model_split "$KIMI_K3_REPO" "$KIMI_K3_MODELS_DIR" "$(kimi_k3_include_glob)" \
                     "$(kimi_k3_first_shard)" "$(kimi_k3_n_shards)"
fi
[ -f "$GGUF" ] || { echo "!! missing weights: $GGUF  (re-run with --download)" >&2; exit 1; }
verify_model_split "$KIMI_K3_MODELS_DIR" "$(kimi_k3_first_shard)" "$(kimi_k3_manifest)"

# llama-tokenize is what makes the accuracy gate possible for this model: K3 ships a
# tiktoken vocab, not an HF tokenizer.json, so prompt ids must come from the GGUF itself.
export LLAMACPP_EXTRA_TARGETS="${LLAMACPP_EXTRA_TARGETS:-llama-tokenize}"
ensure_llamacpp "$ARCH"
LLAMA_BENCH="$LLAMACPP_DIR/build/bin/llama-bench"

pin_clocks
trap 'unpin_clocks' EXIT

mkdir -p "$OUT_DIR"
ROWS_JSON="{}"

# llama-bench -o json emits one entry per (test, run). Pull the median samples_ts for the
# tg row and the pp row separately; a single -p/-n pair yields exactly those two tests.
run_one_ctx() {  # $1 = ctx ; echoes "<pp_tps> <tg_tps>" (0 for a skipped/failed half)
  local ctx="$1" out rc=0 pp=0 tg=0
  local pflag=("-p" "$ctx") nflag=("-n" "$DECODE_TOKENS")
  [ "$DO_PREFILL" = 1 ] || pflag=("-p" "0")
  [ "$DO_DECODE"  = 1 ] || nflag=("-n" "0")
  # shellcheck disable=SC2086
  out="$("$LLAMA_BENCH" -m "$GGUF" "${pflag[@]}" "${nflag[@]}" -r "$REPS" \
          $(kimi_k3_llama_flags) -o json 2>&1)" || rc=$?
  if [ "$rc" != 0 ]; then
    echo ">> WARN: llama-bench failed at ctx=$ctx (rc=$rc): ${out##*$'\n'}" >&2
    printf '0 0'; return 1
  fi
  read -r pp tg <<<"$(python3 - <<'PY' "$out"
import json, re, statistics, sys
raw = sys.argv[1]
# tolerate build/log noise around the JSON array
m = re.search(r"\[.*\]", raw, re.S)
pp = tg = 0.0
if m:
    try:
        for e in json.loads(m.group(0)):
            ts = e.get("samples_ts") or []
            if not ts:
                continue
            med = statistics.median(ts)
            if e.get("n_prompt", 0) > 0 and e.get("n_gen", 0) == 0:
                pp = med
            elif e.get("n_gen", 0) > 0:
                tg = med
    except Exception:
        pass
print(f"{pp:.4f} {tg:.4f}")
PY
)"
  printf '%s %s' "$pp" "$tg"
}

echo
echo "=== llama.cpp reference sweep (median of $REPS, n=$DECODE_TOKENS) ==="
printf '%10s  %14s  %14s\n' ctx "prefill pp/s" "decode tg/s"
for ctx in $CTXS; do
  read -r pp tg <<<"$(run_one_ctx "$ctx" || true)"
  printf '%10s  %14s  %14s\n' "$ctx" "$pp" "$tg"
  ROWS_JSON="$(python3 - "$ROWS_JSON" "$ctx" "$pp" "$tg" <<'PY'
import json, sys
d = json.loads(sys.argv[1]); d[sys.argv[2]] = {"prefill_pp": float(sys.argv[3]), "decode_tps": float(sys.argv[4])}
print(json.dumps(d, separators=(",", ":")))
PY
)"
done

python3 - > "$OUT_JSON" <<PY
import json, os, subprocess
def sh(*c):
    try: return subprocess.check_output(c, text=True).strip()
    except Exception: return ""
print(json.dumps({
    "engine": "llama.cpp (unsloth fork)",
    "engine_repo": "${LLAMACPP_REPO}",
    "engine_ref": "${LLAMACPP_REF:-}",
    "engine_commit": "${LLAMACPP_COMMIT:-}",
    "model": "Kimi K3",
    "quant": "$QUANT",
    "gguf": os.path.basename("$GGUF"),
    "arch": "sm_$ARCH",
    "gpus": int("$NGPU"),
    "gpu_name": sh("bash", "-c", "nvidia-smi --query-gpu=name --format=csv,noheader | head -1"),
    "pinned_graphics_clock_mhz": "${PINNED_GCLK:-}",
    "decode_tokens": int("$DECODE_TOKENS"),
    "reps": int("$REPS"),
    "llama_flags": "$(kimi_k3_llama_flags)",
    "contexts": json.loads('''$ROWS_JSON'''),
    "sparkinfer": None,
    "note": "Baseline only — sparkinfer has no kimi-k3 loader yet (see bench/configs/models/kimi_k3.yaml).",
}, indent=2))
PY

echo
echo ">> wrote $OUT_JSON"
echo ">> next: pin these into bench/scripts/reference.lock (KIMI_K3_LLAMA_*) so future"
echo ">>       runs are scored against a fixed, third-party-reproducible reference."
