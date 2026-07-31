#!/usr/bin/env python3
"""Enforce that every pinned Kimi K3 baseline traces back to a recorded measurement.

  python3 bench/scripts/check_reference_lock.py [--results-dir bench/results]

The whole premise of this repo is that a native runtime gets scored against a reference
somebody else can reproduce. That collapses the moment a baseline can be typed in by hand:
downstream, `KIMI_K3_H200X8_LLAMA_4K=311.2` looks identical whether it came from a sweep on
a real 8x H200 or from a hopeful guess. reference.lock is maintainer-only, so review is the
first line of defence — this is the second, and unlike review it cannot be forgotten.

Rules:
  1. A baseline of 0 means NOT MEASURED. Always allowed.
  2. A non-zero baseline must appear in at least one bench/results/*.json emitted by
     kimi_k3_baseline.sh, for the same node and context, within FLOAT_TOL.
  3. A results file must not claim a node whose profile does not exist.

Exit 0 = clean, 1 = a pinned value has no measurement behind it.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "bench" / "scripts" / "reference.lock"

# Nodes with a profile in _kimi_k3.sh. Keep in sync with kimi_k3_node_arch().
KNOWN_NODES = {"h200x8", "b200x8", "b300x4"}

# reference.lock suffix -> the context key kimi_k3_baseline.sh writes into the JSON.
CTX_OF_SUFFIX = {
    "128": 128, "512": 512, "4K": 4096, "32K": 32768, "1M": 1048576,
    "4K_PP": 4096, "32K_PP": 32768, "1M_PP": 1048576,
}
FLOAT_TOL = 0.01     # the lock stores 2dp of what the sweep measured

# KIMI_K3_<NODE>[_<QUANT>]_LLAMA_<CTX>[_PP].
#
# The optional QUANT segment exists because the node alone does not identify a
# reference: UD-IQ1_S is 249 GiB smaller than the UD-Q2_K_XL target and decodes
# several times faster on the same box. Pinning an IQ1_S number into an unqualified
# slot would silently score every future PR against the wrong denominator — the exact
# class of undetectable-downstream error this checker exists to prevent. So a
# quant-qualified slot must match a results file for THAT quant, not just that node.
VAR_RE = re.compile(
    r'^(KIMI_K3_(?P<node>[A-Z0-9]+)(?:_(?P<quant>[A-Z0-9]+))?_LLAMA_(?P<suffix>[0-9]+K?M?(?:_PP)?))='
    r'"\$\{\1:-(?P<value>[0-9.]+)\}"\s*$'
)


def _norm_quant(q):
    """UD-IQ1_S -> IQ1S, so the lock's identifier charset matches the JSON's."""
    return re.sub(r'[^A-Z0-9]', '', (q or '').upper()).replace('UD', '', 1) or None


def parse_lock(path: Path):
    """-> [(var, node, suffix, is_prefill, value)] for every KIMI_K3_*_LLAMA_* line."""
    out = []
    for line in path.read_text().splitlines():
        m = VAR_RE.match(line.strip())
        if not m:
            continue
        suffix = m.group("suffix")
        out.append((
            m.group(1),
            m.group("node").lower(),
            suffix,
            suffix.endswith("_PP"),
            float(m.group("value")),
            m.group("quant"),
        ))
    return out


def load_results(results_dir: Path):
    """-> {(node, quant, ctx, is_prefill): [values]} from every baseline JSON on disk.

    Keyed by quant as well as node; an entry is also recorded under quant=None so an
    unqualified lock slot still resolves the way it always did."""
    measured: dict[tuple, list[float]] = {}
    bad_nodes = []
    for f in sorted(results_dir.glob("kimi_k3_*_baseline_*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"  !! unreadable results file {f.name}: {e}")
            continue
        node = (d.get("node_profile") or "").lower()
        quant = _norm_quant(d.get("quant"))
        if node and node not in KNOWN_NODES:
            bad_nodes.append((f.name, node))
        for ctx_s, row in (d.get("contexts") or {}).items():
            try:
                ctx = int(ctx_s)
            except ValueError:
                continue
            for is_pp, key in ((False, "decode_tps"), (True, "prefill_pp")):
                v = row.get(key)
                if isinstance(v, (int, float)) and v > 0:
                    measured.setdefault((node, quant, ctx, is_pp), []).append(float(v))
                    measured.setdefault((node, None, ctx, is_pp), []).append(float(v))
    return measured, bad_nodes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "bench" / "results"))
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    entries = parse_lock(LOCK)
    if not entries:
        print("!! no KIMI_K3_*_LLAMA_* entries found in reference.lock — parser or lock changed")
        return 1

    measured, bad_nodes = load_results(results_dir) if results_dir.is_dir() else ({}, [])

    pinned = [e for e in entries if e[4] > 0]
    print(f"reference.lock: {len(entries)} K3 baseline slots, {len(pinned)} pinned, "
          f"{len(entries) - len(pinned)} unmeasured (0)")
    print(f"results dir   : {results_dir}"
          f"{'' if results_dir.is_dir() else '  (absent — fine while nothing is pinned)'}")

    errors = []
    for name, node in bad_nodes:
        errors.append(f"{name}: claims unknown node profile {node!r} "
                      f"(known: {sorted(KNOWN_NODES)})")

    for var, node, suffix, is_pp, value, quant in pinned:
        if node not in KNOWN_NODES:
            errors.append(f"{var}: unknown node {node!r} — add a profile or drop the slot")
            continue
        ctx = CTX_OF_SUFFIX.get(suffix)
        if ctx is None:
            errors.append(f"{var}: suffix {suffix!r} maps to no context — extend CTX_OF_SUFFIX")
            continue
        found = measured.get((node, quant, ctx, is_pp), [])
        if not found:
            errors.append(
                f"{var} = {value} is pinned but NO results file records "
                f"{'prefill' if is_pp else 'decode'} at ctx={ctx} on node={node}. "
                f"Run kimi_k3_baseline.sh --node {node} and commit its JSON, or reset to 0."
            )
            continue
        if not any(abs(v - value) <= max(FLOAT_TOL, abs(value) * 1e-4) for v in found):
            errors.append(
                f"{var} = {value} does not match any measurement at ctx={ctx} on "
                f"node={node} (recorded: {sorted(found)})"
            )

    if errors:
        print("\nFAIL — pinned baselines without a measurement behind them:\n")
        for e in errors:
            print(f"  * {e}")
        print("\nA hand-filled baseline is indistinguishable from a measured one downstream.")
        return 1

    print("OK — every pinned baseline traces to a recorded measurement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
