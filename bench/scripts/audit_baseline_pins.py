#!/usr/bin/env python3
"""Detect drift in the two external things this repo's baseline depends on.

  python3 bench/scripts/audit_baseline_pins.py

The Kimi K3 reference is a PR HEAD on somebody else's fork, and the weights live in
somebody else's Hugging Face repo. Neither is immutable:

  1. unslothai/llama.cpp PR #48 can be force-pushed. `ensure_llamacpp` already refuses to
     run when the ref no longer resolves to the pinned sha — but only when somebody runs a
     sweep. This surfaces it on a schedule instead of at the moment you need the node.
  2. unsloth can re-upload a quant with a different shard count. `kimi_k3_n_shards` pins
     those counts so an interrupted download fails loudly; a legitimate re-upload turns the
     same pin into a spurious failure. Better to learn that here.

Exit 0 = pins match reality, 1 = drift (or the check could not be completed).
Network required. `--offline` prints what would be checked and exits 0.
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "bench" / "scripts" / "reference.lock"
KIMI_SH = ROOT / "bench" / "scripts" / "_kimi_k3.sh"
TIMEOUT = 30


def lock_var(name: str, text: str) -> str | None:
    m = re.search(rf'^{name}="\$\{{{name}:-([^}}]*)\}}"', text, re.M)
    return m.group(1) if m else None


def pinned_shards(text: str) -> dict[str, int]:
    """Parse the case arms of kimi_k3_n_shards(): `UD-Q2_K_XL) n=19 ;;`."""
    body = re.search(r"kimi_k3_n_shards\(\)\s*\{(.*?)\n\}", text, re.S)
    if not body:
        return {}
    out = {}
    for quant, n in re.findall(r"(UD-[A-Z0-9_]+)\)\s*n=(\d+)", body.group(1)):
        out[quant] = int(n)
    return out


def fetch_json(url: str, token: str | None = None):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "sparkinfer-k3-pin-audit",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def check_fork_pin(text: str, token: str | None) -> list[str]:
    repo_url = lock_var("KIMI_K3_LLAMACPP_REPO", text) or ""
    ref = lock_var("KIMI_K3_LLAMACPP_REF", text) or ""
    want = lock_var("KIMI_K3_LLAMACPP_COMMIT", text) or ""
    print(f"\n[1] baseline engine pin\n    repo   {repo_url}\n    ref    {ref}\n    commit {want}")
    if not (repo_url and ref and want):
        return ["could not parse KIMI_K3_LLAMACPP_{REPO,REF,COMMIT} from reference.lock"]

    slug = repo_url.rstrip("/").removeprefix("https://github.com/")
    m = re.match(r"refs/pull/(\d+)/head$", ref)
    api = (f"https://api.github.com/repos/{slug}/pulls/{m.group(1)}" if m
           else f"https://api.github.com/repos/{slug}/commits/{ref}")
    try:
        d = fetch_json(api, token)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return [f"could not reach {api}: {e}"]

    if m:
        got = (d.get("head") or {}).get("sha", "")
        state = d.get("state")
        merged = d.get("merged")
        print(f"    PR #{m.group(1)} state={state} merged={merged}")
        print(f"    head   {got}")
        notes = []
        if got != want:
            notes.append(
                f"PR head moved: {ref} is now {got}, pinned {want}. The baseline changed "
                f"under us — review the new commits, then re-pin DELIBERATELY (and re-measure, "
                f"because any pinned number was produced by the old code)."
            )
        if merged:
            print("    note: PR is merged — consider re-pinning to the upstream commit")
        return notes

    got = d.get("sha", "")
    print(f"    head   {got}")
    return [] if got == want else [f"ref {ref} is now {got}, pinned {want}"]


def check_shard_counts(text_sh: str) -> list[str]:
    pins = pinned_shards(text_sh)
    print(f"\n[2] GGUF shard counts ({len(pins)} quants pinned in _kimi_k3.sh)")
    if not pins:
        return ["could not parse shard counts from kimi_k3_n_shards()"]
    try:
        d = fetch_json("https://huggingface.co/api/models/unsloth/Kimi-K3-GGUF")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return [f"could not reach the Hugging Face API: {e}"]

    live: dict[str, int] = {}
    for s in d.get("siblings", []):
        m = re.search(r"(UD-[A-Z0-9_]+)/Kimi-K3-\1-00001-of-(\d{5})\.gguf$", s["rfilename"])
        if m:
            live[m.group(1)] = int(m.group(2))

    notes = []
    for quant in sorted(set(pins) | set(live)):
        want, got = pins.get(quant), live.get(quant)
        mark = "ok " if want == got else "!! "
        print(f"    {mark}{quant:12} pinned={want if want is not None else '-':<4} "
              f"published={got if got is not None else '-'}")
        if want is None:
            notes.append(f"{quant} is published ({got} shards) but has no pin in kimi_k3_n_shards()")
        elif got is None:
            notes.append(f"{quant} is pinned at {want} shards but is no longer published")
        elif want != got:
            notes.append(f"{quant} shard count changed: pinned {want}, published {got} — "
                         f"a re-upload invalidates any manifest and any measured baseline")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="describe the checks, make no requests")
    ap.add_argument("--token", default=None, help="GitHub token (raises the API rate limit)")
    args = ap.parse_args()

    text = LOCK.read_text()
    text_sh = KIMI_SH.read_text()

    if args.offline:
        print("offline: would check")
        print(f"  - {lock_var('KIMI_K3_LLAMACPP_REF', text)} still resolves to "
              f"{lock_var('KIMI_K3_LLAMACPP_COMMIT', text)}")
        print(f"  - shard counts for {sorted(pinned_shards(text_sh))}")
        return 0

    problems = check_fork_pin(text, args.token) + check_shard_counts(text_sh)

    if problems:
        print("\nDRIFT DETECTED\n")
        for p in problems:
            print(f"  * {p}")
        print("\nNothing is broken yet — but a pinned number measured against the old state is "
              "no longer reproducible from the current one.")
        return 1
    print("\nOK — both external pins match reality")
    return 0


if __name__ == "__main__":
    sys.exit(main())
