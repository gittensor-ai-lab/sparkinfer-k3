#!/usr/bin/env bash
# Kimi K3 EVAL — score sparkinfer against the llama.cpp reference and emit the label.
#
#   bench/scripts/kimi_k3_eval.sh [--node h200x8] [--devices 0,1,..] [--frontier TOKS]
#                                 [--llama-ref TOKS] [--layers N] [--json-out FILE]
#                                 [--seal]
#
# --seal additionally seals the verdict into a Polaris receipt and publishes the run to
# sparkinfer-k3-log. eval-label.yml reads the receipt FROM THAT LOG, so an unsealed run
# cannot be scored. Needs POLARIS_API_KEY and push access to the log.
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
#              Read from reference.lock (KIMI_K3_<NODE>_<QUANT>_LLAMA_128, falling back
#              to the unqualified slot) unless --llama-ref.
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
SEAL=0
EXT_TOP1=""      # --top1/--kl: accuracy measured by the CONTROLLER, off this box
EXT_KL=""

while [[ $# -gt 0 ]]; do case "$1" in
  --node)      shift; NODE="$1" ;;
  --devices)   shift; DEVICES="$1" ;;
  --frontier)  shift; FRONTIER="$1" ;;
  --llama-ref) shift; LLAMA_REF="$1" ;;
  --layers)    shift; LAYERS="$1" ;;
  --tokens)    shift; TOKENS="$1" ;;
  --json-out)  shift; JSON_OUT="$1" ;;
  --top1)      shift; EXT_TOP1="$1" ;;
  --kl)        shift; EXT_KL="$1" ;;
  --seal)      SEAL=1 ;;
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

# THE WEIGHTS ARE SHARED STATE AND THE BINARY UNDER TEST RUNS AS ROOT.
#
# reference.lock's "C2: reference quarantine" exists because a malicious build could corrupt
# the baseline weights to depress every later measurement -- but the eval path never checked.
# A full sha256 is not affordable here (553 GiB per run), so fingerprint the shard set by
# size and mtime: cheap, and enough to notice that the weights changed under us between runs.
# Pinning it is what makes it a gate rather than a note; absent, it is recorded into the
# receipt so a change is at least visible after the fact.
#
# This DETECTS tampering. It does not prevent it -- the fix for that is mounting
# KIMI_K3_MODELS_DIR read-only for the bench, which is an ops change on the box.
MODEL_DIR="$(dirname "$MODEL")"
FINGERPRINT="$(find "$MODEL_DIR" -maxdepth 1 -name '*.gguf' -printf '%f %s %T@\n' 2>/dev/null \
               | sort | sha256sum | cut -c1-16)"
PIN_FILE="$HERE/kimi_k3_$(printf '%s' "${PRIMARY_QUANT:-UD-IQ1_S}" | tr 'A-Z-' 'a-z_').fingerprint"
if [[ -f "$PIN_FILE" ]]; then
    PINNED_FP="$(tr -d '[:space:]' < "$PIN_FILE")"
    if [[ "$FINGERPRINT" != "$PINNED_FP" ]]; then
        echo "kimi_k3_eval: model fingerprint $FINGERPRINT does not match the pinned" >&2
        echo "  $PINNED_FP in $(basename "$PIN_FILE"). The shared weights changed." >&2
        echo "  Refusing to score: every tier is relative to these weights, so a run against" >&2
        echo "  different ones is not comparable to the frontier it is measured against." >&2
        exit 1
    fi
    echo ">> model fingerprint  : $FINGERPRINT (matches pin)"
else
    echo ">> model fingerprint  : $FINGERPRINT (no pin — recorded, not enforced)"
fi

# reference.lock is the pinned source for both the llama reference and the current
# frontier. Sourcing it (rather than hardcoding) is what makes M1/M2/M3 independent.
# shellcheck source=/dev/null
[[ -f "$HERE/reference.lock" ]] && source "$HERE/reference.lock"
# SLOTS ARE QUANT-QUALIFIED. reference.lock stores the measured values as
# KIMI_K3_<NODE>_<QUANT>_{LLAMA,SPARKINFER}_128 because the node alone does not identify
# a reference — UD-IQ1_S is 249 GiB smaller than UD-Q2_K_XL and decodes faster on the
# same box, so one number in the other's slot would misprice every future PR.
#
# This lookup used to read the UNQUALIFIED slots, which are all still 0. Both refs came
# back 0, and the consequences were silent and total:
#   frontier 0  -> label.py returns BASELINE for every run, so no PR could ever score.
#   llama_ref 0 -> DIFF_REF <= 0, so label.py falls back to delta/frontier — the exact
#                  un-anchored basis the llama-anchored tier exists to replace.
# Try the qualified slot first, fall back to the unqualified one for a quant that has
# not been split out yet.
PFX="KIMI_K3_$(printf '%s' "$NODE" | tr 'a-z' 'A-Z')"
QUANT="$(printf '%s' "${PRIMARY_QUANT:-UD-IQ1_S}" | sed 's/^UD-//' | tr -cd 'A-Za-z0-9' | tr 'a-z' 'A-Z')"
# THE SCORED CONTEXT IS 128k, AND THE SLOT SUFFIX FOLLOWS IT. Both references have to come
# from the same context as the measurement or the tier is arithmetic on unrelated numbers.
# Scoring used to read the _128 slots while the bench ran with max_ctx hardcoded to 64 --
# so the recorded context, the slot name and the measured context were three different
# things, and 128k, the configuration this repo actually targets, was never scored at all.
#
# 128k is where the two engines diverge. Measured on 8x H200 / UD-IQ1_S: llama.cpp gives up
# 8% from depth 0 to 131,072 (18.20 -> 16.70 tok/s) while sparkinfer gives up 90%
# (10.34 -> 1.01). Scoring at 64 hid that entirely.
SCORED_CTX="${KIMI_K3_SCORED_CTX:-131072}"
CTX_SUFFIX="${KIMI_K3_CTX_SUFFIX:-128K}"
lookup() {  # $1 = LLAMA | SPARKINFER
    local v
    v="$(eval "printf '%s' \"\${${PFX}_${QUANT}_$1_${CTX_SUFFIX}:-}\"")"
    [[ -n "$v" && "$v" != "0" ]] && { printf '%s' "$v"; return; }
    eval "printf '%s' \"\${${PFX}_$1_${CTX_SUFFIX}:-0}\""
}
[[ -n "$LLAMA_REF" ]] || LLAMA_REF="$(lookup LLAMA)"
[[ -n "$FRONTIER"  ]] || FRONTIER="$(lookup SPARKINFER)"

echo ">> Kimi K3 eval — node=$NODE devices=$DEVICES layers=$LAYERS ctx=$SCORED_CTX"
echo ">> llama.cpp reference (tier basis): $LLAMA_REF tok/s  @ ctx $SCORED_CTX"
echo ">> sparkinfer frontier             : $FRONTIER tok/s$([[ "$FRONTIER" == "0" ]] && echo '  (none yet -> BASELINE)')"

# ---- 1. speed -------------------------------------------------------------
# THE NUMBER IS TIMED FROM OUTSIDE THE BINARY, NOT READ OUT OF IT.
#
# kimi_k3_tp_bench is built from the PR's OWN runtime/ -- only bench/scripts is restored
# from the protected branch -- so anything the binary PRINTS is under the PR's control. This
# used to sed the tok/s straight out of its stdout, taking `head -1`, while the honest line
# is printed near the end. One printf at startup therefore set the tier. The --seek
# announcement this used to require was defeated the same way: by printing it.
#
# So the cost per token is derived from WALL CLOCK across two runs at different token counts.
# The model load (~150 s) and every other fixed cost cancel in the difference, leaving the
# marginal cost of TOKENS_HI - TOKENS_LO decode steps. A binary cannot print its way out of
# elapsed time.
#
# What this still cannot prove is that the work was real: a bench that returns immediately is
# indistinguishable from an infinitely fast one. That residual is bounded by the accuracy
# gate, which now runs off-box (step 2) -- but only at short context, because no 128k
# reference exists to check the seeked run against.
NDEV="$(printf '%s' "$DEVICES" | tr ',' '\n' | grep -c .)"
TOKENS_LO="$TOKENS"
TOKENS_HI="$(( TOKENS * 3 ))"

# POLARIS_API_KEY signs the attestation ledger, and `. .polaris_env` put it in the
# environment of every child -- including this binary, which the PR wrote. env -u strips it
# for the bench only; the harness keeps it for the seal step below.
run_bench() {  # $1 = n_tokens ; echoes elapsed ns on stdout, bench output on fd 3
    local t0 t1 out rc=0
    t0="$(date +%s%N)"
    out="$(env -u POLARIS_API_KEY "$BENCH" "$MODEL" "$NDEV" "$LAYERS" "$1" \
             --ctx "$SCORED_CTX" --seek 2>&1)" || rc=$?
    t1="$(date +%s%N)"
    [[ "$rc" == 0 ]] || { printf '%s\n' "$out" | tail -20 >&2; return 1; }
    printf '%s\n' "$out" >&3
    printf '%s' "$(( t1 - t0 ))"
}

echo ">> timing sparkinfer decode at ctx $SCORED_CTX (2 runs, ${TOKENS_LO} and ${TOKENS_HI} tokens) ..."
SPEED_OUT="$(mktemp -t k3speed_XXXXXX)"
trap 'rm -f "$SPEED_OUT"' EXIT
NS_LO="$(run_bench "$TOKENS_LO" 3>"$SPEED_OUT")" || { echo "kimi_k3_eval: bench failed" >&2; exit 1; }
NS_HI="$(run_bench "$TOKENS_HI" 3>>"$SPEED_OUT")" || { echo "kimi_k3_eval: bench failed" >&2; exit 1; }

read -r TPS MSTOK <<<"$(python3 - "$NS_LO" "$NS_HI" "$TOKENS_LO" "$TOKENS_HI" <<'PY'
import sys
ns_lo, ns_hi, n_lo, n_hi = (int(x) for x in sys.argv[1:5])
d_ns, d_n = ns_hi - ns_lo, n_hi - n_lo
if d_n <= 0 or d_ns <= 0:
    sys.stderr.write(f"kimi_k3_eval: non-positive time delta ({d_ns} ns over {d_n} tokens) — "
                     "the longer run was not slower, so nothing was measured\n")
    raise SystemExit(1)
ms = d_ns / d_n / 1e6
print(f"{1000.0 / ms:.2f} {ms:.2f}")
PY
)" || exit 1
echo ">> sparkinfer: $TPS tok/s ($MSTOK ms/token)  [wall-clock differential]"

# Cross-check against what the binary claimed. They measure the same thing by different
# means, so a large gap is either a lying binary or a broken measurement, and both are
# reasons to stop rather than to pick one.
SELF="$(sed -n 's/.*ms\/token[^(]*(\([0-9.]*\) tok\/s).*/\1/p' "$SPEED_OUT" | head -1)"
if [[ -n "$SELF" ]]; then
    python3 - "$SELF" "$TPS" <<'PY' || exit 1
import sys
self_tps, ext_tps = float(sys.argv[1]), float(sys.argv[2])
print(f">> self-reported {self_tps} tok/s vs wall-clock {ext_tps} tok/s", end="")
if ext_tps <= 0 or self_tps / ext_tps > 1.25:
    print()
    sys.stderr.write(
        f"kimi_k3_eval: the bench claims {self_tps} tok/s but wall clock allows {ext_tps}.\n"
        "  refusing to score: the binary is built from the PR's runtime/, and a self-report\n"
        "  faster than elapsed time is the signature of a fabricated number.\n")
    raise SystemExit(1)
print("  (agree)")
PY
fi

# ---- 2. correctness -------------------------------------------------------
# THE ANSWER KEY MUST NOT BE ON THE SAME MACHINE AS THE BINARY BEING GRADED.
#
# This used to check bench/refdata out onto the box and compare there. The reference sat at
# a fixed relative path in the very working tree the PR's binary ran in, with cwd at the
# repo root -- so a bench could open bench/refdata/hello.spkl and write it back out as its
# own --logits dump. That is top1 1.0 and KL 0.0 for a binary that computed nothing, and no
# rule rejected a suspiciously perfect result: label.py only bounds top1 >= 0.90 and
# KL <= 0.20 (honest main measures 0.004, so an exact 0.0 was a tell nothing looked for).
#
# So the controller now measures accuracy: it feeds the ids, copies the logits dump back,
# and compares against a reference the box never sees, then passes the result in as
# --top1/--kl. The box cannot fake a match to a file it does not have.
if [[ -n "$EXT_TOP1" && -n "$EXT_KL" ]]; then
    TOP1="$EXT_TOP1"; KL="$EXT_KL"
    echo ">> accuracy: top1=$TOP1  mean_kld=$KL  [measured off-box by the controller]"
else
    # LOCAL DEVELOPMENT ONLY. Kept so a maintainer can run the harness by hand on a box they
    # trust, and loud because the same path is worthless against a hostile PR.
    REF_SPKL="$ROOT/bench/refdata/hello.spkl"
    REF_IDS_FILE="$ROOT/bench/refdata/hello.ids"
    echo ">> WARN: comparing accuracy ON THIS BOX against $REF_SPKL." >&2
    echo ">>       The binary under test can read that file. Trustworthy only when you" >&2
    echo ">>       built the binary yourself; the eval bot passes --top1/--kl instead." >&2
    if [[ -f "$REF_SPKL" && -f "$REF_IDS_FILE" ]]; then
        # hello.ids holds prompt ids THEN the reference continuations; the prompt is the
        # first 4. Taking the whole file would score a different step.
        IDS_CSV="$(head -4 "$REF_IDS_FILE" | paste -sd, -)"
        OURS="$(mktemp -t k3eval_XXXXXX.spkl)"
        trap 'rm -f "$OURS" "$SPEED_OUT"' EXIT
        echo ">> measuring accuracy vs llama.cpp on ids $IDS_CSV ..."
        env -u POLARIS_API_KEY "$BENCH" "$MODEL" "$NDEV" "$LAYERS" 1 \
            --ids "$IDS_CSV" --logits "$OURS" >/dev/null 2>&1 || true
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
fi

# ---- 3. label -------------------------------------------------------------
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
PROV="$(python3 - "$NODE" "$DEVICES" "$LAYERS" "$MODEL" "$MSTOK" "$FINGERPRINT" \
        "${EXT_TOP1:+controller}" "$SCORED_CTX" <<'PY'
import json, sys, os
node, devs, layers, model, mstok, fp, acc_src, ctx = sys.argv[1:9]
print(json.dumps({
    "node": node, "devices": devs, "layers": int(layers),
    "quant": os.path.basename(os.path.dirname(model)),
    "ms_per_token": float(mstok) if mstok else None,
    "engine": "sparkinfer/kimi_k3_tp_bench",
    "llama_commit": os.environ.get("KIMI_K3_LLAMACPP_COMMIT", ""),
    # How the two numbers that decide the tier were obtained. Both used to come from the
    # binary under test; a verifier reading the log could not tell, because nothing recorded
    # it. "wall_clock" and "controller" are the trustworthy values -- anything else means
    # the run was graded by the thing it was grading.
    "speed_source": "wall_clock_differential",
    "accuracy_source": acc_src or "on_box",
    "model_fingerprint": fp,
    "scored_ctx": int(ctx),
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

RESULT="$(SPARKINFER_DIFFICULTY_REF="$LLAMA_REF" SPARKINFER_SCORED_CONTEXT="$SCORED_CTX" \
    python3 "$HERE/label.py" "$TPS" "$FRONTIER" 0 "$TOP1" "$KL" "$COMMIT" "$PROV")"
echo
echo "$RESULT"

# ---- 4. seal + publish --------------------------------------------------
# eval-label.yml reads the receipt from sparkinfer-k3-log, NOT from the PR comment, so a
# verdict that is not sealed and published cannot be scored. Doing it here rather than
# leaving it to the operator is the difference between attestation being the default and
# attestation being the thing everyone forgets.
#
# --seal is opt-in because sealing needs POLARIS_API_KEY and push access to the log, and a
# developer running this locally to check a number should not be blocked by either.
if [ "$SEAL" = 1 ]; then
    RJ="$(mktemp -t k3result_XXXXXX.json)"
    printf '%s\n' "${RESULT#RESULT_JSON }" > "$RJ"
    echo
    echo ">> sealing + publishing ..."
    if ! python3 "$HERE/kimi_k3_attest.py" --result "$RJ" --model "$MODEL" \
             --node "$NODE" --build-dir "$BUILD" --publish; then
        echo ">> FAILED to seal/publish — this verdict is NOT scorable." >&2
        echo ">>   eval-label.yml will refuse it when REQUIRE_EVAL_RECEIPT=1." >&2
        rm -f "$RJ"; exit 1
    fi
    rm -f "$RJ"
fi
if [[ -n "$JSON_OUT" ]]; then
    printf '%s\n' "${RESULT#RESULT_JSON }" > "$JSON_OUT"
    echo ">> wrote $JSON_OUT"
fi
