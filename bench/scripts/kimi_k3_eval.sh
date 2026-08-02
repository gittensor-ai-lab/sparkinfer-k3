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
# top1 0.90 / KLD 0.05 (K3's pinned bar — label.py's shared default is the Qwen track's
# 0.20) no matter how fast the run was — a speedup that erodes parity with the reference
# is not a speedup worth having.
#
# WHY THE REFERENCE IS A CAPTURE AND NOT A LIVE llama.cpp RUN. Re-running llama.cpp
# for every eval would mean loading 553 GiB twice per PR. bench/refdata/*.spkl is the
# reference's own output on pinned weights at a pinned commit, so the comparison is
# identical and costs one sparkinfer forward.
#
# Env: KIMI_K3_MODEL (or KIMI_K3_MODELS_DIR) · SPARKINFER_BUILD · KIMI_K3_NODE
#
# Speed-guard knobs (step 1). The two guards have to overlap — see the comment there:
#   KIMI_K3_MARGIN_TOKENS  128    tokens of difference between the two timed runs
#   KIMI_K3_SPEED_TOL      1.5    max claim / wall-clock differential
#   KIMI_K3_WORK_TOL       2.0    max wall-clock differential / claim (proves the work ran)
#   KIMI_K3_MAX_OVER_LLAMA 3.0    max claim as a multiple of the llama.cpp reference
#   KIMI_K3_JITTER_S       2.0    assumed load jitter; only used to check the guards overlap
#
# Accuracy bars (step 3). Both are RECORDED into the payload and checked by eval-label.yml,
# so a run scored under a moved bar is refused rather than silently re-derived:
#   KIMI_K3_KL_BAR         0.05   hard REJECT ceiling for mean KLD
#   KIMI_K3_KL_PREFER      0.02   soft flag (accuracy_warn) above this

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
# The model load and every other fixed cost cancel in the difference, leaving the marginal
# cost of TOKENS_HI - TOKENS_LO decode steps. A binary cannot print its way out of elapsed
# time.
#
# TWO NUMBERS, TWO JOBS, AND CONFLATING THEM WAS A MISTAKE.
#
#   The SELF-REPORT is the measurement. It is timed inside the process around decode only,
#   so it is precise, and precision is what a tier needs: a noisy number moves PRs across
#   band boundaries for no reason.
#
#   The DIFFERENTIAL is the guard. It is external and therefore unforgeable, but it carries
#   the run-to-run variance of the model load, so it is a BOUND, not a reading.
#
# Using the differential as the authoritative number failed on trusted main: at 8 and 24
# tokens the marginal signal was ~3.5 s against a ~29 s load, so jitter landed in it -- 4.55
# tok/s self-reported against 3.45 measured, a 32% error, and the run refused itself. The
# signal has to dominate the jitter, hence a fixed and much larger token margin rather than a
# multiple of TOKENS. (That jitter was estimated at 1-2 s here. Measured later on the h200x8
# box it is 7.0 s peak-to-peak -- see MARGIN_TOKENS below, which is sized off the measurement
# rather than the estimate.)
#
# THE DIFFERENTIAL BOUNDS NOTHING UNLESS THE SECOND RUN ACTUALLY DID THE EXTRA TOKENS.
#
# n_tokens is argv[4] of a binary the PR wrote. Nothing here compels it to be honoured, and
# a bench that ignores it does the SAME work in both runs: the difference collapses to load
# jitter, ext_tps goes to (nearly) infinity, and every claim clears the bound. The upper
# bound alone is therefore self-neutering -- the cheapest attack on it is to do less work,
# not more.
#
# So the marginal time must ALSO be at least as long as the claim itself implies. That side
# is safe in the way the upper bound is not: load jitter INFLATES d_ns (the differential
# reads slower than truth), so it can only ever make this check pass more easily, never fail
# spuriously. Together the two sides pin d_ns into a window, and a fabricated claim of X
# tok/s now has to burn MARGIN_TOKENS/X seconds of real wall clock to be believed.
#
# What this still cannot prove is that the elapsed time was spent DECODING -- sleep() buys
# the same seconds. That residual is bounded by the accuracy gate, which now runs off-box
# (step 2) -- but only at short context, because no 128k reference exists to check the
# seeked run against.
NDEV="$(printf '%s' "$DEVICES" | tr ',' '\n' | grep -c .)"
TOKENS_LO="$TOKENS"
# THE MARGIN HAS TO TRACK THE ENGINE, OR THE GUARD EATS ITS OWN WORK.
#
# 128 was chosen when main ran at 4.53 tok/s, where 128 marginal tokens is ~28 s of decode
# against 1-2 s of load jitter -- a 5% bound. That ratio is not a constant: the decode signal
# is MARGIN_TOKENS x ms/token, so every speedup this loop pays for SHRINKS it, while the load
# jitter it has to clear does not move. Four rounds of merged wins later:
#
#     main  4.53 tok/s -> 128 marginal tokens = 28.3 s
#     main  9.04 tok/s ->                       14.2 s
#     main 14.95 tok/s ->                        8.6 s
#     main 18.88 tok/s ->                        6.8 s     <- signal
#
# and measured on the h200x8 box at 131072 ctx, the 554 GiB load takes 33.25-40.27 s across
# five back-to-back runs: 7.0 s of spread. The noise had overtaken the signal. #77 was refused
# by the work_tol side for exactly this -- 128 tokens that should cost 6.36 s were measured at
# 2.96 s, a 3.4 s shortfall that fits inside the load spread -- with a self-report of 49.70
# ms/token that the other four sweep points corroborate to within 0.2%. A guard that refuses
# an honest PR because the thing it guards got faster is a countdown, not a check.
#
# 512 restores the margin: 27.1 s of decode, and work_tol=2.0 tolerates a 13.6 s shortfall
# against 7.0 s of observed jitter. It costs ~20 s more wall clock per bench pair, which is
# nothing against a ~40 minute round, and it re-widens as ms/token falls -- so this number
# owes a review the next time the frontier doubles. Env-tunable for that reason.
MARGIN_TOKENS="${KIMI_K3_MARGIN_TOKENS:-512}"
TOKENS_HI="$(( TOKENS_LO + MARGIN_TOKENS ))"

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

# The two runs are back to back, and the first one is holding ~1.1 TiB across 8 devices when
# it exits. Freeing that is not instant: two consecutive 136-token runs on an OTHERWISE IDLE
# box (nvidia-smi: 0 MiB used on all 8) died in cudaMalloc partway through layer 92 -- the
# last layer, so it very nearly fit -- while the same command run a minute later succeeded
# twice. The driver had not finished reclaiming the previous process's memory.
#
# That is survivable but wasteful: the bench returns 1 on init failure (kimi_k3_tp_bench.cpp
# returns 1 at "init failed"), so run_bench aborts the whole eval and the PR gets no number
# at all through no fault of its own. Wait for the devices to actually come back instead.
settle_gpus() {  # wait until every device is ~free, or give up and let the run try anyway
    command -v nvidia-smi >/dev/null 2>&1 || { sleep 5; return 0; }
    local i busy
    for i in $(seq 1 "${KIMI_K3_SETTLE_TRIES:-30}"); do
        # `|| true` is load-bearing: this script runs under `set -euo pipefail`, so without
        # it a single non-zero nvidia-smi -- a driver hiccup, an ECC scrub, a transient --
        # fails the pipeline, fails the assignment, and kills the whole eval. A wait added to
        # stop a PR losing its eval would then be a new way for a PR to lose its eval.
        # Failing here degrades to "assume free" (wc -l prints 0 on empty input), which is
        # exactly what the no-nvidia-smi path above does.
        busy="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
                | awk -v lim="${KIMI_K3_SETTLE_MIB:-2048}" '$1 > lim' | wc -l || true)"
        [[ "$busy" == "0" ]] && { [[ "$i" == "1" ]] || echo ">> GPUs settled after ${i}s" >&2; return 0; }
        sleep 1
    done
    echo ">> warning: GPUs still busy after ${KIMI_K3_SETTLE_TRIES:-30}s — running anyway" >&2
}

echo ">> timing sparkinfer decode at ctx $SCORED_CTX (2 runs, ${TOKENS_LO} and ${TOKENS_HI} tokens) ..."
SPEED_OUT="$(mktemp -t k3speed_XXXXXX)"
trap 'rm -f "$SPEED_OUT"' EXIT
settle_gpus
NS_LO="$(run_bench "$TOKENS_LO" 3>"$SPEED_OUT")" || { echo "kimi_k3_eval: bench failed" >&2; exit 1; }
settle_gpus
NS_HI="$(run_bench "$TOKENS_HI" 3>>"$SPEED_OUT")" || { echo "kimi_k3_eval: bench failed" >&2; exit 1; }

# The self-report is the reading; the differential is the bound it has to survive. A missing
# self-report is fatal on its own -- there is no number to score, and a binary that prints
# nothing has not been measured by anything.
SELF_TPS="$(sed -n 's/.*ms\/token[^(]*(\([0-9.]*\) tok\/s).*/\1/p' "$SPEED_OUT" | head -1)"
SELF_MS="$(sed -n 's/.*ms\/token *\([0-9.]*\).*/\1/p' "$SPEED_OUT" | head -1)"
if [[ -z "$SELF_TPS" ]]; then
    tail -20 "$SPEED_OUT" >&2
    echo "kimi_k3_eval: the bench reported no ms/token line — nothing to score" >&2
    exit 1
fi

# A GUARD THAT CANNOT ABORT IS COMMENTARY. This used to be
#   read -r TPS MSTOK <<<"$(python3 ...)" || exit 1
# and the `|| exit 1` binds to `read`, not to the substitution — bash discards the
# substitution's exit status, and read returns 0 on the empty string a refusing guard
# leaves behind. So the refusal printed its message and the script walked on with TPS=""
# through the empty ">> sparkinfer:  tok/s" echo into label.py, which died on float('')
# with a traceback that buried the actual reason. #64 hit exactly this on the guard's
# first live firing: refused for a collapsed differential, then "eval failed —
# ValueError" instead of the refusal being the last word. Capture first, so the exit
# status is REAL; then parse.
SPEED_VERDICT="$(python3 - "$NS_LO" "$NS_HI" "$TOKENS_LO" "$TOKENS_HI" \
                                 "$SELF_TPS" "$SELF_MS" "${KIMI_K3_SPEED_TOL:-1.5}" \
                                 "${KIMI_K3_WORK_TOL:-2.0}" "$LLAMA_REF" \
                                 "${KIMI_K3_MAX_OVER_LLAMA:-3.0}" "${KIMI_K3_JITTER_S:-2.0}" <<'PY'
import sys
ns_lo, ns_hi, n_lo, n_hi = (int(x) for x in sys.argv[1:5])
self_tps, self_ms, tol = float(sys.argv[5]), float(sys.argv[6] or 0), float(sys.argv[7])
work_tol = float(sys.argv[8])
llama_ref, max_over_llama, jitter_s = (float(x or 0) for x in sys.argv[9:12])
d_ns, d_n = ns_hi - ns_lo, n_hi - n_lo
if d_n <= 0 or d_ns <= 0:
    sys.stderr.write(f"kimi_k3_eval: non-positive time delta ({d_ns} ns over {d_n} tokens) — "
                     "the longer run was not slower, so nothing was measured\n")
    raise SystemExit(1)
ext_ms = d_ns / d_n / 1e6
ext_tps = 1000.0 / ext_ms
sys.stderr.write(f">> wall-clock bound: {ext_tps:.2f} tok/s ({ext_ms:.2f} ms/token) "
                 f"over {d_n} marginal tokens\n")
# The bound is one-sided ON PURPOSE. Load jitter inflates the differential's ms/token, so
# the external number reads SLOWER than truth; a self-report slower than the bound is
# therefore never suspicious. Only a claim FASTER than elapsed time allows is evidence of
# fabrication, and tol carries the jitter.
if self_tps > ext_tps * tol:
    sys.stderr.write(
        f"kimi_k3_eval: the bench claims {self_tps} tok/s but wall clock allows at most "
        f"{ext_tps * tol:.2f} ({ext_tps:.2f} x {tol}).\n"
        "  refusing to score: the binary is built from the PR's runtime/, and a self-report\n"
        "  faster than elapsed time is the signature of a fabricated number.\n")
    raise SystemExit(1)
# THE OTHER SIDE: the marginal tokens have to have COST something. A bench that ignores its
# n_tokens argument does identical work in both runs, d_ns is then load jitter alone, and the
# check above passes for any claim at all. Jitter can only inflate d_ns, so this side has
# nothing to absorb but the same slack -- work_tol is generous because it does not need to be
# tight, it needs to separate "128 tokens of decode" from "no tokens of decode".
if self_ms > 0 and ext_tps > self_tps * work_tol:
    implied_s = d_n * self_ms / 1000.0
    sys.stderr.write(
        f"kimi_k3_eval: {d_n} extra tokens took {d_ns / 1e9:.2f} s of wall clock, but at the "
        f"claimed {self_ms:.2f} ms/token they must take about {implied_s:.2f} s "
        f"({implied_s / work_tol:.2f} s at the {work_tol}x floor).\n"
        "  refusing to score: the longer run did not do the extra work, so the differential\n"
        "  bounds nothing and the self-report is unchecked.\n")
    raise SystemExit(1)

# A TIMING GUARD CAN ONLY BOUND A CLAIM IN PROPORTION TO THE TIME THAT CLAIM IMPLIES.
#
# The check above costs an attacker MARGIN_TOKENS/claim seconds of real wall clock. That is
# ruinous for a claim near the truth (128 tokens at main's ~1 tok/s is ~128 s) and free for
# an absurd one: at 500 tok/s the 128 marginal tokens "should" take 0.26 s, which the load
# jitter supplies for nothing. The crossover is MARGIN_TOKENS / (jitter * work_tol) -- above
# it, elapsed time stops being evidence, and no amount of tuning the tolerance changes that.
#
# So the band above the crossover has to be closed by PLAUSIBILITY instead of by timing.
# llama.cpp on the same weights, box and context is the anchor already pinned in
# reference.lock, and label.py saturates at XL well before a few multiples of it -- a claim
# far past the reference earns nothing extra and is not a number this harness should be
# auto-scoring. A real breakthrough gets re-measured by hand and the knob raised on purpose.
crossover = d_n / (jitter_s * work_tol) if jitter_s > 0 and work_tol > 0 else float("inf")
if llama_ref > 0:
    ceiling = llama_ref * max_over_llama
    if self_tps > ceiling:
        sys.stderr.write(
            f"kimi_k3_eval: the bench claims {self_tps} tok/s, over {max_over_llama}x the "
            f"llama.cpp reference ({llama_ref}) at this context — ceiling {ceiling:.2f}.\n"
            "  refusing to score: past this point the wall-clock differential is smaller than\n"
            "  load jitter, so nothing here can tell a breakthrough from a fabrication.\n"
            "  Re-measure by hand and raise KIMI_K3_MAX_OVER_LLAMA deliberately if it is real.\n")
        raise SystemExit(1)
    if ceiling > crossover:
        sys.stderr.write(
            f">> WARN: guard coverage gap — claims between {crossover:.1f} and {ceiling:.1f} "
            f"tok/s can ride {jitter_s} s of load jitter.\n"
            f">>       Raise KIMI_K3_MARGIN_TOKENS (now {d_n}) or lower "
            "KIMI_K3_MAX_OVER_LLAMA to close it.\n")
else:
    sys.stderr.write(
        f">> WARN: no llama.cpp reference for this node/quant/context, so the plausibility\n"
        f">>       ceiling is off and any claim above ~{crossover:.1f} tok/s is unguarded.\n")

# Scored on the self-report: precise, and the tier must not wobble with load jitter.
print(f"{self_tps:.2f} {self_ms:.2f}")
PY
)" || exit 1
read -r TPS MSTOK <<<"$SPEED_VERDICT"
# Belt and braces: the guard prints exactly "TPS MSTOK" on success, so an empty parse here
# means the wiring above regressed, not that the run was slow. Refuse rather than limp on.
if [[ -z "$TPS" || -z "$MSTOK" ]]; then
    echo "kimi_k3_eval: speed guard exited 0 but produced no verdict — wiring bug, not a measurement" >&2
    exit 1
fi
echo ">> sparkinfer: $TPS tok/s ($MSTOK ms/token)  [self-timed, within the wall-clock bound]"

# ---- 2. correctness -------------------------------------------------------
# THE ANSWER KEY MUST NOT BE ON THE SAME MACHINE AS THE BINARY BEING GRADED.
#
# This used to check bench/refdata out onto the box and compare there. The reference sat at
# a fixed relative path in the very working tree the PR's binary ran in, with cwd at the
# repo root -- so a bench could open bench/refdata/hello.spkl and write it back out as its
# own --logits dump. That is top1 1.0 and KL 0.0 for a binary that computed nothing, and no
# rule rejected a suspiciously perfect result: label.py only bounds top1 >= 0.90 and
# KL <= 0.05 (honest main measures 0.004, so an exact 0.0 was a tell nothing looked for).
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

# K3 PINS ITS OWN ACCURACY BARS. label.py is shared with the inherited Qwen track, whose
# honest runs measure KL 0.0175-0.03, so its DEFAULT must stay loose (0.20) or every Qwen
# run would be REJECTed. K3 measures 0.004046 on main, so 0.20 would let a PR degrade
# parity forty-fold and pass clean.
#
# The knob exists for local experiments, and that is exactly why the values are RECORDED
# into provenance below. eval-label.yml re-derives the verdict with its own pinned 0.05,
# so a box that ran with a different bar produces a label the workflow cannot reproduce --
# which used to surface as "reported label != re-derived: payload edited or harness version
# mismatch", naming the two things that were not wrong. Publishing the bars lets the
# workflow say which knob was moved instead of guessing.
KL_BAR="${KIMI_K3_KL_BAR:-0.05}"
KL_PREFER="${KIMI_K3_KL_PREFER:-0.02}"

PROV="$(python3 - "$NODE" "$DEVICES" "$LAYERS" "$MODEL" "$MSTOK" "$FINGERPRINT" \
        "${EXT_TOP1:+controller}" "$SCORED_CTX" "$KL_BAR" "$KL_PREFER" <<'PY'
import json, sys, os
node, devs, layers, model, mstok, fp, acc_src, ctx, kl_bar, kl_prefer = sys.argv[1:11]
print(json.dumps({
    "node": node, "devices": devs, "layers": int(layers),
    "quant": os.path.basename(os.path.dirname(model)),
    "ms_per_token": float(mstok) if mstok else None,
    "engine": "sparkinfer/kimi_k3_tp_bench",
    "llama_commit": os.environ.get("KIMI_K3_LLAMACPP_COMMIT", ""),
    # How the two numbers that decide the tier were obtained. Both used to come from the
    # binary under test; a verifier reading the log could not tell, because nothing recorded
    # it.
    #
    # SAY WHAT ACTUALLY HAPPENED. This used to claim "wall_clock_differential" while the
    # scored value is the binary's SELF-REPORT (admitted only inside the wall-clock window
    # -- see step 1). A provenance field that asserts the trustworthy path regardless of
    # which path ran is worse than no field: it teaches a log reader to stop checking.
    # eval-label.yml now refuses any value it does not recognise, which is what makes this
    # a record instead of a caption.
    "speed_source": "self_report_wall_clock_bounded",
    "accuracy_source": acc_src or "on_box",
    "model_fingerprint": fp,
    "scored_ctx": int(ctx),
    # The accuracy policy this verdict was computed under. Not a scoring key -- label.py's
    # allowlist would reject it if it were -- but the trusted re-derivation compares it to
    # its own pin, so a moved knob is named rather than inferred.
    "kl_bar": float(kl_bar),
    "kl_prefer": float(kl_prefer),
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

# The bars pinned above. eval-label.yml pins the SAME pair when it re-derives; a bot and a
# workflow disagreeing about REJECT is the same class of bug as disagreeing about the
# frontier.
RESULT="$(SPARKINFER_DIFFICULTY_REF="$LLAMA_REF" SPARKINFER_SCORED_CONTEXT="$SCORED_CTX" \
    SPARKINFER_KL_BAR="$KL_BAR" \
    SPARKINFER_KL_PREFER="$KL_PREFER" \
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
