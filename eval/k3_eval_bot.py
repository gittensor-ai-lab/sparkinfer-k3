#!/usr/bin/env python3
"""Kimi K3 PR evaluation bot.

Replaces the inherited Qwen evaluator (pr_eval_bot.py), which had zero K3 support and
would have auto-closed every K3 pull request. This one does exactly one job:

    find an eligible PR -> measure it on the 8x H200 node -> post the result

and deliberately nothing else. It does not label, merge, close, or block. Labelling is
.github/workflows/eval-label.yml, which re-derives the tier from reference.lock rather
than trusting anything posted here; capping, stale-closing and copycat detection are
already GitHub Actions workflows. A bot that only reports is a bot whose worst failure is
a wrong number in a comment a maintainer can see, rather than a merged PR or a banned
contributor.

WHY IT ONLY REPORTS. The scoring path pays SN74 emissions. Every autonomous mutation this
bot could perform is a way to pay the wrong person or block the right one, and none of
them need to be autonomous to make evaluation automatic. So the split is: measuring is
mechanical and slow, and is automated here; deciding is cheap and is left to a human.

TRANSPORT. Self-contained on purpose. It talks to a pinned box over ssh via EVAL_SSH_HOST
/ EVAL_SSH_PORT and does not import eval/ssh_box.py or eval/vast_eval.py, so removing the
Qwen track cannot break it.

    EVAL_SSH_HOST=1.2.3.4 EVAL_SSH_PORT=22 \\
    python3 eval/k3_eval_bot.py --repo gittensor-ai-lab/sparkinfer-k3 --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import sys

REPO_DEFAULT = "gittensor-ai-lab/sparkinfer-k3"
BOX_REPO_DIR = os.environ.get("K3_BOX_REPO_DIR", "/workspace/k3-botrun")
BOX_MODELS_DIR = os.environ.get("K3_BOX_MODELS_DIR", "/workspace/models_k3")
NODE = os.environ.get("KIMI_K3_NODE", "h200x8")
DEVICES = os.environ.get("K3_DEVICES", "0,1,2,3,4,5,6,7")
BUILD_JOBS = os.environ.get("K3_BUILD_JOBS", "48")

# The PR template's hardware attestation. Matched loosely on purpose -- the template uses a
# multiplication sign (8x H200) that is easy to retype as an ASCII x, and failing a real
# submission over a homoglyph is worse than accepting a near-miss. What must be exact is
# the ticked box, because that is the author's attestation that they ran it.
TICKED = re.compile(r"-\s*\[\s*[xX]\s*\]")
H200 = re.compile(r"h200", re.I)


def gh(args, timeout=120):
    return subprocess.run(["gh"] + args, capture_output=True, text=True, timeout=timeout)


def sh(cmd, timeout=7200):
    """Run a command on the pinned eval box."""
    host = os.environ.get("EVAL_SSH_HOST", "").strip()
    port = os.environ.get("EVAL_SSH_PORT", "22").strip()
    if not host:
        raise SystemExit("k3_eval_bot: set EVAL_SSH_HOST to the 8x H200 node")
    try:
        return subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
             "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=40",
             "-p", str(port), f"root@{host}", cmd],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess([], 1, stdout="", stderr=f"ssh timeout after {timeout}s")


def list_prs(repo):
    r = gh(["pr", "list", "-R", repo, "--state", "open", "--limit", "50", "--json",
            "number,title,isDraft,headRefOid,body,author,labels,isCrossRepository"])
    if r.returncode != 0:
        raise SystemExit(f"k3_eval_bot: gh pr list failed: {r.stderr.strip()}")
    return json.loads(r.stdout or "[]")


def eligibility(pr):
    """Return (ok, reason). Mirrors node-attestation.yml rather than inventing a rule."""
    if pr.get("isDraft"):
        return False, "draft — unfinished work is not evaluated"
    labels = {l.get("name", "") for l in pr.get("labels") or []}
    if "hold" in labels:
        return False, "hold label set"
    body = pr.get("body") or ""
    ticked = any(TICKED.search(ln) and H200.search(ln) for ln in body.splitlines())
    if not ticked:
        return False, "the 8x H200 box is not ticked — the author has not attested a node run"
    return True, "eligible"


def evaluate(pr, repo, seal=False):
    """Check the PR head out on the box, build it, and run the K3 harness there.

    Returns the parsed RESULT_JSON, or raises RuntimeError with the box's own output. The
    build and the eval both happen on the node; nothing is measured locally.
    """
    sha = pr["headRefOid"]
    num = pr["number"]
    steps = " && ".join([
        f"cd {BOX_REPO_DIR}",
        "git fetch -q origin +refs/pull/*/head:refs/remotes/origin/pr/* --force",
        f"git fetch -q origin {sha} || true",
        f"git checkout -q --detach {sha}",
        "export PATH=/usr/local/cuda/bin:$PATH",
        "cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 >/dev/null",
        f"cmake --build build -j{BUILD_JOBS} --target kimi_k3_tp_bench >/dev/null",
    ])
    r = sh(f"{steps} && echo K3_BUILD_OK", timeout=5400)
    if "K3_BUILD_OK" not in (r.stdout or ""):
        raise RuntimeError(f"build failed for #{num} @ {sha[:8]}\n{(r.stderr or r.stdout)[-1500:]}")

    seal_flag = " --seal" if seal else ""
    evalcmd = (
        f"cd {BOX_REPO_DIR} && "
        f"KIMI_K3_MODELS_DIR={BOX_MODELS_DIR} SPARKINFER_BUILD={BOX_REPO_DIR}/build "
        f"bash bench/scripts/kimi_k3_eval.sh --node {NODE} --devices {DEVICES}{seal_flag}"
    )
    r = sh(evalcmd, timeout=7200)
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    line = next((l for l in out.splitlines() if l.strip().startswith("RESULT_JSON")), None)
    if not line:
        raise RuntimeError(f"no RESULT_JSON from the harness for #{num}\n{out[-1500:]}")
    res = json.loads(line.split("RESULT_JSON", 1)[1].strip())

    # Bind the measurement to the commit we asked for. eval-label.yml refuses a payload
    # whose commit is not the PR head, so a mismatch here would only fail later and more
    # confusingly -- and a harness that measured something other than what we checked out
    # is a bug worth stopping on, not papering over.
    got = str(res.get("commit", "")).lower()
    if not sha.lower().startswith(got) or len(got) < 7:
        raise RuntimeError(
            f"harness reported commit {got!r} but #{num}'s head is {sha[:12]!r} — refusing "
            f"to post a result that does not belong to this PR")
    return res


def post(repo, num, res, dry_run):
    body = (
        "/eval RESULT_JSON " + json.dumps(res, separators=(",", ":")) + "\n\n"
        "<sub>Measured on the pinned 8x H200 node by `eval/k3_eval_bot.py`. The tier is "
        "re-derived from `bench/scripts/reference.lock` on `main` by `eval-label.yml`; the "
        "numbers above are inputs to that, not the verdict.</sub>"
    )
    if dry_run:
        print(f"--- dry-run: would comment on #{num} ---\n{body}\n")
        return True
    r = gh(["pr", "comment", str(num), "-R", repo, "--body", body])
    if r.returncode != 0:
        print(f"!! failed to comment on #{num}: {r.stderr.strip()}", file=sys.stderr)
        return False
    print(f">> posted eval to #{num}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Measure eligible K3 PRs on the H200 node and post the result.")
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--only-pr", type=int, default=0, help="evaluate just this PR number")
    ap.add_argument("--dry-run", action="store_true", help="print the comment instead of posting")
    ap.add_argument("--seal", action="store_true", help="pass --seal to the harness (needs POLARIS_API_KEY on the box)")
    ap.add_argument("--list", action="store_true", help="show eligibility for every open PR and exit")
    args = ap.parse_args()

    # This bot is K3-specific in the same way the old one was Qwen-specific. Pointing it at
    # another repo would check out that repo's head on a box holding 554 GiB of K3 weights
    # and score whatever came out against K3's frontier.
    if not re.search(r"-k3(/|$)", args.repo):
        sys.stderr.write(f"k3_eval_bot: refusing to run against {args.repo!r} — this bot only "
                         f"evaluates Kimi K3 repositories.\n")
        return 2

    prs = list_prs(args.repo)
    if args.only_pr:
        prs = [p for p in prs if p["number"] == args.only_pr]
        if not prs:
            sys.stderr.write(f"k3_eval_bot: #{args.only_pr} is not an open PR on {args.repo}\n")
            return 1

    if args.list:
        for pr in prs:
            ok, why = eligibility(pr)
            print(f"  #{pr['number']:<5} {'ELIGIBLE' if ok else 'skip':<9} {why:<62} {pr['title'][:44]}")
        return 0

    evaluated = 0
    for pr in prs:
        num = pr["number"]
        ok, why = eligibility(pr)
        if not ok:
            print(f"#{num}: skip — {why}")
            continue
        print(f"#{num}: evaluating {pr['headRefOid'][:8]} on {NODE} …")
        try:
            res = evaluate(pr, args.repo, seal=args.seal)
        except RuntimeError as exc:
            print(f"#{num}: eval failed — {exc}", file=sys.stderr)
            continue
        print(f"#{num}: tps={res.get('tps')} top1={res.get('top1')} kl={res.get('kl')} "
              f"label={res.get('label')}")
        post(args.repo, num, res, args.dry_run)
        evaluated += 1

    print(f"done — {evaluated} evaluated, no merges and no labels (by design)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
