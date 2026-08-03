#!/usr/bin/env bash
# Kimi K3 evaluation BASELINE — UD-Q2_K_XL at 1M context, on any of the milestone nodes.
#
#   bench/scripts/kimi_k3_baseline.sh [--download] [--decode-only|--prefill-only]
#                                     [--node h200x8|b200x8|b300x4] [--reps N]
#                                     [--ctx "128 512 4096 32768"] [--longctx] [--dry-run]
#
# Builds the pinned unsloth llama.cpp fork (the only engine that can load a 896-expert
# model — see _kimi_k3.sh), verifies the sharded weights against the pinned manifest,
# then sweeps llama-bench decode (tg) and prefill (pp) at each context and writes
# machine-readable results to bench/results/.
#
#   --node   which milestone's hardware this run claims to be. Selects the reference.lock
#            prefix so M1/M2/M3 cannot overwrite each other, and warns on a mismatch
#            between the claimed profile and the detected box.
#   --longctx  additionally probe 128k / 256k / 1M at 1 rep. NOT in the default sweep: a
#            1M-token prefill is minutes of wall-clock per rep, so it is a capability
#            probe, not a median. All three hardware milestones require it to pass.
#
# This is the reference half of the eval only. sparkinfer has no kimi-k3 loader yet, so
# there is nothing to compare against yet; the point is to produce a pinned, reproducible
# number that a future native implementation is scored against on the SAME box with the
# SAME GGUF.
#
# Env: PRIMARY_QUANT (UD-Q2_K_XL) · KIMI_K3_NODE · KIMI_K3_MODELS_DIR · LLAMACPP_DIR · ARCH
#      KIMI_K3_MAX_CTX / _COMPUTE_BUF_GIB · KIMI_K3_NGL / _BATCH / _UBATCH / _SPLIT_MODE
#      KIMI_K3_EXTRA_FLAGS · CUDA_VISIBLE_DEVICES
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$HERE/_common.sh"
# shellcheck source=_kimi_k3.sh
source "$HERE/_kimi_k3.sh"

DOWNLOAD=0; DO_DECODE=1; DO_PREFILL=1; REPS=3; DRY=0; LONGCTX=0
# 128k IS IN THE SCORED SWEEP, not deferred to --longctx. UD-IQ1_S at 128k is the
# configuration this repo actually runs; a number that only covers 128-token decode has
# not shown anything about the case anyone uses. 256k/1M stay in --longctx because a 1M
# prefill is minutes per rep — those are capability probes, this is a median.
CTXS="${KIMI_K3_CTXS:-128 512 4096 32768 131072}"
LONGCTXS="${KIMI_K3_LONGCTXS:-262144 1048576}"
DECODE_TOKENS="${KIMI_K3_DECODE_TOKENS:-64}"
while [ $# -gt 0 ]; do case "$1" in
  --download)     DOWNLOAD=1 ;;
  --decode-only)  DO_PREFILL=0 ;;
  --prefill-only) DO_DECODE=0 ;;
  --node)         shift; export KIMI_K3_NODE="$1" ;;
  --reps)         shift; REPS="$1" ;;
  --ctx)          shift; CTXS="$1" ;;
  --longctx)      LONGCTX=1 ;;
  --dry-run)      DRY=1 ;;
  -h|--help)      sed -n '2,22p' "$0"; exit 0 ;;
  *)              echo "!! unknown arg: $1" >&2; exit 2 ;;
esac; shift; done

ARCH="$(detect_arch)"
NGPU="$(gpu_count)"
GGUF="$(kimi_k3_gguf)"
QUANT="$(kimi_k3_quant)"
NODE="$(kimi_k3_node)"
# reference.lock prefix: h200x8 -> KIMI_K3_H200X8_*. Per-node so M1/M2/M3 cannot overwrite
# each other's reference — they run the same quant at the same context, so the node is the
# only variable and a shared slot would silently conflate them.
# THE WRITER AND THE READER MUST DERIVE THIS THE SAME WAY.
#
# _kimi_k3.sh states the convention outright: "The reference.lock slots keep the quant
# in their NAME for exactly this reason (KIMI_K3_H200X8_IQ1S_LLAMA_128), so a number
# measured under this default can never be read as a Q2_K_XL result." This script did
# not implement it -- it emitted KIMI_K3_H200X8_LLAMA_* whatever PRIMARY_QUANT said,
# while kimi_k3_eval.sh reads KIMI_K3_H200X8_IQ1S_LLAMA_128K. So a correct measurement
# landed in a slot nothing scores from, and the unqualified slots it did write are the
# all-zero ones whose emptiness once made every PR score BASELINE.
#
# Same derivation as kimi_k3_eval.sh, character for character, so the two cannot drift.
LOCK_PREFIX="KIMI_K3_$(printf '%s' "$NODE" | tr 'a-z' 'A-Z')_$(printf '%s' "${PRIMARY_QUANT:-UD-IQ1_S}" | sed 's/^UD-//' | tr -cd 'A-Za-z0-9' | tr 'a-z' 'A-Z')"
OUT_DIR="${KIMI_K3_OUT_DIR:-$ROOT/bench/results}"
STAMP="${KIMI_K3_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_JSON="$OUT_DIR/kimi_k3_${QUANT}_${NODE}_baseline_${STAMP}.json"

echo ">> Kimi K3 baseline — quant=$QUANT node=$NODE"
echo ">> baseline engine: ${LLAMACPP_REPO} @ ${LLAMACPP_COMMIT:-UNPINNED} (${LLAMACPP_REF:-no ref})"
[ -n "${LLAMACPP_COMMIT:-}" ] || echo ">> WARN: llama.cpp NOT pinned — this run is not reproducible"

if [ "$DRY" = 1 ]; then
  echo ">> dry run: would use"
  echo "   node     : $(kimi_k3_node_label)"
  echo "   expects  : sm_$(kimi_k3_node_arch), >=$(kimi_k3_node_gpus) GPU(s), ~$(kimi_k3_node_gib) GiB HBM"
  echo "   detected : sm_$ARCH, $NGPU GPU(s)"
  echo "   gguf     : $GGUF"
  echo "   mmproj   : $(kimi_k3_mmproj)  (M4 only, not loaded here)"
  echo "   manifest : $(kimi_k3_manifest)"
  echo "   flags    : $(kimi_k3_llama_flags)"
  echo "   weights  : $(kimi_k3_size_gib) GiB + $(kimi_k3_headroom_gib) GiB headroom" \
       "(KV $(kimi_k3_kv_gib) GiB @ ${KIMI_K3_MAX_CTX})"
  echo "   contexts : $CTXS  (reps=$REPS, n=$DECODE_TOKENS)"
  [ "$LONGCTX" = 1 ] && echo "   longctx  : $LONGCTXS  (reps=1, capability probe)"
  echo "   lock pfx : ${LOCK_PREFIX}_LLAMA_*"
  echo "   out      : $OUT_JSON"
  exit 0
fi

kimi_k3_check_node
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

sweep() {  # $1 = space-separated contexts, $2 = reps, $3 = "scored"|"probe"
  local ctxs="$1" reps="$2" kind="$3" ctx pp tg saved_reps="$REPS"
  REPS="$reps"
  for ctx in $ctxs; do
    read -r pp tg <<<"$(run_one_ctx "$ctx" || true)"
    printf '%10s  %14s  %14s  %s\n' "$ctx" "$pp" "$tg" "$kind"
    ROWS_JSON="$(python3 - "$ROWS_JSON" "$ctx" "$pp" "$tg" "$kind" "$reps" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
d[sys.argv[2]] = {"prefill_pp": float(sys.argv[3]), "decode_tps": float(sys.argv[4]),
                  "kind": sys.argv[5], "reps": int(sys.argv[6])}
print(json.dumps(d, separators=(",", ":")))
PY
)"
  done
  REPS="$saved_reps"
}

echo
echo "=== llama.cpp reference sweep — $(kimi_k3_node_label) ==="
printf '%10s  %14s  %14s  %s\n' ctx "prefill pp/s" "decode tg/s" kind
sweep "$CTXS" "$REPS" scored

if [ "$LONGCTX" = 1 ]; then
  echo
  echo "=== long-context capability probe (1 rep — a 1M prefill is minutes per rep) ==="
  printf '%10s  %14s  %14s  %s\n' ctx "prefill pp/s" "decode tg/s" kind
  sweep "$LONGCTXS" 1 probe
fi

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
    "node_profile": "$NODE",
    "node_label": "$(kimi_k3_node_label)",
    "node_expects_arch": "sm_$(kimi_k3_node_arch)",
    "arch": "sm_$ARCH",
    "gpus": int("$NGPU"),
    "gpu_name": sh("bash", "-c", "nvidia-smi --query-gpu=name --format=csv,noheader | head -1"),
    "hbm_gib_total": int("$(gpu_vram_gib_total)"),
    "pinned_graphics_clock_mhz": "${PINNED_GCLK:-}",
    "max_ctx_budgeted": int("$KIMI_K3_MAX_CTX"),
    "kv_gib_at_max_ctx": int("$(kimi_k3_kv_gib)"),
    "decode_tokens": int("$DECODE_TOKENS"),
    "reps": int("$REPS"),
    "llama_flags": "$(kimi_k3_llama_flags)",
    "reference_lock_prefix": "${LOCK_PREFIX}_LLAMA_",
    "contexts": json.loads('''$ROWS_JSON'''),
    "sparkinfer": None,
    "note": "Baseline only — sparkinfer has no kimi-k3 loader yet (see bench/configs/models/kimi_k3.yaml).",
}, indent=2))
PY

echo
echo ">> wrote $OUT_JSON"
echo ">> next: pin these into bench/scripts/reference.lock as ${LOCK_PREFIX}_LLAMA_* so"
echo ">>       future runs are scored against a fixed, third-party-reproducible reference."
[ "$LONGCTX" = 1 ] || echo ">> NOTE: 1M context NOT probed — re-run with --longctx to close the milestone."
