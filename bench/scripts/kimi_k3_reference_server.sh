#!/usr/bin/env bash
# Start the Kimi K3 llama.cpp reference server — the accuracy side of the baseline.
#
#   bench/scripts/kimi_k3_reference_server.sh [--ctx N] [--port N] [--foreground]
#   bench/scripts/kimi_k3_reference_server.sh --stop
#
# accuracy_compare.py scores a candidate engine against this server's per-position
# distributions (/completion with n_probs), on the SAME GGUF. That is the only honest
# correctness gate available for K3: there is no second independent implementation, and
# the 1-bit quant's own top-1 vs full precision is 78.9%, so absolute quality numbers are
# meaningless — only agreement with the reference on the identical weights is.
#
# Two K3-specific requirements, both load-bearing:
#   --no-context-shift  hybrid recurrent arch; llama.cpp cannot shift context or restore
#                       slots for it, and a long eval dies mid-run without this
#   --no-jinja          accuracy_compare.py posts raw token ids to /completion; applying
#                       the chat template would prepend tokens the candidate never saw
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$HERE/_common.sh"
# shellcheck source=_kimi_k3.sh
source "$HERE/_kimi_k3.sh"

CTX="${KIMI_K3_SERVER_CTX:-8704}"
PORT="${KIMI_K3_SERVER_PORT:-8081}"
FOREGROUND=0; STOP=0
PIDFILE="${KIMI_K3_SERVER_PIDFILE:-/tmp/kimi_k3_llama_server.pid}"
LOGFILE="${KIMI_K3_SERVER_LOG:-/tmp/kimi_k3_llama_server.log}"

while [ $# -gt 0 ]; do case "$1" in
  --ctx)        shift; CTX="$1" ;;
  --port)       shift; PORT="$1" ;;
  --foreground) FOREGROUND=1 ;;
  --stop)       STOP=1 ;;
  -h|--help)    sed -n '2,19p' "$0"; exit 0 ;;
  *)            echo "!! unknown arg: $1" >&2; exit 2 ;;
esac; shift; done

if [ "$STOP" = 1 ]; then
  if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo ">> stopped reference server"
  else
    echo ">> no pidfile at $PIDFILE — nothing to stop"
  fi
  exit 0
fi

ARCH="$(detect_arch)"
GGUF="$(kimi_k3_gguf)"
[ -f "$GGUF" ] || { echo "!! missing weights: $GGUF  (run kimi_k3_baseline.sh --download)" >&2; exit 1; }
kimi_k3_check_fits
export LLAMACPP_EXTRA_TARGETS="${LLAMACPP_EXTRA_TARGETS:-llama-tokenize}"
ensure_llamacpp "$ARCH"
SRV_BIN="$LLAMACPP_DIR/build/bin/llama-server"

# shellcheck disable=SC2206,SC2046
CMD=("$SRV_BIN" -m "$GGUF" -c "$CTX" --port "$PORT"
     $(kimi_k3_llama_flags) $(kimi_k3_server_flags))

echo ">> ${CMD[*]}"
if [ "$FOREGROUND" = 1 ]; then
  exec "${CMD[@]}"
fi

"${CMD[@]}" >"$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo ">> reference server pid $(cat "$PIDFILE"), log $LOGFILE"

# A 553 GiB load off local NVMe is minutes, not seconds — 20 min ceiling, not the 4 min
# the Qwen-scale gate uses. Fail loudly rather than let the caller time out on /completion.
DEADLINE="${KIMI_K3_SERVER_TIMEOUT:-1200}"
for _ in $(seq 1 $((DEADLINE / 5))); do
  if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
    echo ">> reference server healthy on port $PORT (ctx=$CTX)"
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "!! reference server died during load — last 40 log lines:" >&2
    tail -40 "$LOGFILE" >&2
    rm -f "$PIDFILE"
    exit 1
  fi
  sleep 5
done
echo "!! reference server did not become healthy within ${DEADLINE}s — last 40 log lines:" >&2
tail -40 "$LOGFILE" >&2
exit 1
