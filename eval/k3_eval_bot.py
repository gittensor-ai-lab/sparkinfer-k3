#!/usr/bin/env python3
"""Kimi K3 PR evaluation bot.

Replaces the inherited Qwen evaluator (pr_eval_bot.py), which had zero K3 support and
would have auto-closed every K3 pull request. This one does exactly one job:

    measure main -> find eligible PRs -> measure each on the 8x H200 node -> post the result

THE FRONTIER IS MEASURED EVERY ROUND, NOT PINNED. main is benchmarked on the same box,
minutes before the PRs it is compared against, and that number is what each PR is scored
over. A pinned constant was tried and it failed the way constants do: #25 merged and took
main from 3.55 to 9.54 tok/s, nothing updated reference.lock, and the next PR's 8% gain was
priced against a frontier main had already beaten by 2.7x -- an XL for an S. Measuring also
cancels the box: thermal state and neighbours on a rented machine move tok/s a few percent,
and main and the PR share all of it.

It does not decide a tier. Tiers come from .github/workflows/eval-label.yml, which
re-derives them with label.py from reference.lock on the protected branch rather than
trusting anything posted here -- so a PR cannot understate the frontier to inflate itself.
That is also why this bot WRITES the measured frontier back into reference.lock (one slot,
this node and quant, raise-only): the pinned value is the real tier basis, so leaving it
stale means the posted number and the applied tier disagree.

MERGING. --merge-first only labels. --auto-merge QUEUES GitHub native auto-merge, which
still waits for the required approving review and every required check. --merge-admin does
pass the admin flag and does bypass the review requirement; it is an explicit operator
choice, guarded by NEVER_MERGE_LABELS and NEVER_MERGE_PATHS.

WHY THE SPLIT. The scoring path pays SN74 emissions, so every autonomous mutation is a way
to pay the wrong person or block the right one. Measuring is mechanical and slow, so it is
automated. Deciding is cheap, so it stays human. Queuing a merge is the seam between them:
the bot can say "this one is next", and a human still says "yes".

TRANSPORT. Self-contained on purpose. It talks to a pinned box over ssh via EVAL_SSH_HOST
/ EVAL_SSH_PORT and does not import eval/ssh_box.py or eval/vast_eval.py, so removing the
Qwen track cannot break it.

    EVAL_SSH_HOST=1.2.3.4 EVAL_SSH_PORT=22 \\
    python3 eval/k3_eval_bot.py --repo gittensor-ai-lab/sparkinfer-k3 --dry-run
"""

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import time

REPO_DEFAULT = "gittensor-ai-lab/sparkinfer-k3"
BOX_REPO_DIR = os.environ.get("K3_BOX_REPO_DIR", "/workspace/k3-botrun")
BOX_MODELS_DIR = os.environ.get("K3_BOX_MODELS_DIR", "/workspace/models_k3")
NODE = os.environ.get("KIMI_K3_NODE", "h200x8")
DEVICES = os.environ.get("K3_DEVICES", "0,1,2,3,4,5,6,7")
BUILD_JOBS = os.environ.get("K3_BUILD_JOBS", "48")
POLARIS_ENV = os.environ.get("K3_POLARIS_ENV", "/root/.polaris_env")
LOG_REPO = os.environ.get("K3_LOG_REPO", "gittensor-ai-lab/sparkinfer-k3-log")
BOX_RECEIPTS = os.environ.get("K3_BOX_RECEIPTS", f"{BOX_REPO_DIR}/bench/results/receipts")

# Guards for --merge-admin. That flag bypasses the approving-review requirement on main, so
# these are the only thing left between a measurement and an unreviewed merge of a change
# that carries a payout tier. Every one of them is a reason a human would have wanted to
# look, and none can be satisfied by the PR simply being fast.
NEVER_MERGE_LABELS = {"hold", "copycat", "copycat-warn", "flagged:gaming", "penalty",
                      "needs-benchmark", "needs-node-run", "llm-judge", "needs-rebase"}
# Paths that decide payouts. sensitive-paths-guard already fails a non-maintainer PR that
# touches these, but a bot that merges without anyone reading should not depend on another
# check having run correctly.
NEVER_MERGE_PATHS = ("eval/", ".github/", ".gittensor/", "bench/scripts/", "bench/results/",
                     "bench/refdata/", "dashboard/", "CODEOWNERS")
REBASE_LABEL = "needs-rebase"

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


def _box_build(sha, what):
    """Check `sha` out on the box and build the K3 bench there.

    `what` names the thing being built ("#20", "main") so a failure says which one died.
    Shared by the PR path and the frontier measurement on purpose: a frontier measured with
    a different build than the PR it scores is not a comparison, it is two numbers.
    """
    steps = " && ".join([
        f"cd {BOX_REPO_DIR}",
        # Start from a clean box every time. Two ways the previous PR poisons this one:
        #
        #   An orphaned bench still holding GPU memory. Killing the controller does not kill
        #   what it started over ssh, so an interrupted run leaves 8-125 GiB resident per
        #   device and the next load dies in cudaMalloc partway through layer 92 -- which
        #   reads like an OOM bug in the PR under test rather than leftover state.
        #   NOT pkill -f. The pattern would appear in this very command line, so pkill
        #   matches the remote shell running it and the cleanup kills itself -- the && chain
        #   dies before the build and the PR is reported as "build failed" with no output
        #   at all. Killing exactly what holds GPU memory cannot match a shell.
        #
        #   A dirty tree. Restoring the protected harness stages bench/scripts, so the next
        #   git checkout --detach aborts with "local changes would be overwritten" and the
        #   PR is reported as a build failure it had nothing to do with.
        "for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); "
        "do kill -9 $p 2>/dev/null || true; done",
        "git reset -q --hard",
        "git clean -qfd",
        "git fetch -q origin +refs/pull/*/head:refs/remotes/origin/pr/* --force",
        f"git fetch -q origin {sha} || true",
        f"git checkout -q --detach {sha}",
        # GRADE WITH THE PROTECTED HARNESS, NEVER THE PR'S COPY. Two reasons, and the
        # second is why this is not merely a convenience:
        #
        #   Version skew. The harness moves independently of the code under test, so an
        #   older branch carries an older harness. #25's head predates --seal entirely, so
        #   sealing failed with "unknown arg --seal" on a PR that was otherwise fine. The
        #   same skew is what made its reference.lock resolve frontier=0 and label a real
        #   speedup BASELINE.
        #
        #   Trust. CONTRIBUTING states the evaluator "grades with the harness pinned to the
        #   protected branch, so editing it in a PR never affects that PR's own score."
        #   That was not true of this bot. sensitive-paths-guard stops a non-maintainer
        #   editing these paths, but a guard that fails the PR and a grader that ignores
        #   the PR's copy are different defences, and only the second holds if the guard is
        #   ever misconfigured. refdata/ is included because it IS the accuracy answer key.
        "git fetch -q origin main",
        "git checkout -q origin/main -- bench/scripts bench/refdata",
        "export PATH=/usr/local/cuda/bin:$PATH",
        # -DSPARKINFER_TP=ON is not optional here. Without it the runtime has no NCCL
        # collective, and kimi_k3_tp_bench refuses to run sharded rather than silently
        # producing a wrong number -- correct behaviour, but it means a build configured
        # without this flag fails at eval time with no hint that configure was the cause.
        "cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 "
        "-DSPARKINFER_TP=ON >/dev/null",
        f"cmake --build build -j{BUILD_JOBS} --target kimi_k3_tp_bench >/dev/null",
    ])
    r = sh(f"{steps} && echo K3_BUILD_OK", timeout=5400)
    if "K3_BUILD_OK" not in (r.stdout or ""):
        # An empty failure is worse than a wrong one: it says nothing about where the chain
        # stopped. Report the exit status too, since a setup step that kills the shell
        # produces no output at all and that is exactly the case worth recognising.
        tail = ((r.stderr or "") + (r.stdout or "")).strip()
        tail = "\n".join(l for l in tail.splitlines() if "setlocale" not in l)[-1500:]
        raise RuntimeError(
            f"build failed for {what} @ {sha[:8]} (ssh exit {r.returncode})"
            + (f"\n{tail}" if tail else " — no output at all, so a setup step aborted the "
                                        "chain before cmake ran"))


def _box_eval(what, frontier=None, seal=False):
    """Run the K3 harness on the box against the build already checked out there.

    `frontier` is passed through to the harness as --frontier. Passing it EXPLICITLY is the
    point: left to itself the harness reads the frontier out of reference.lock, and a pinned
    constant only stays true until the next perf PR merges. It did not -- #25 took main from
    3.55 to 9.54 tok/s and the lock kept saying 3.55, so #20's 8% gain was priced against a
    frontier main had already beaten by 2.7x and came out XL instead of S.

    Returns (RESULT_JSON dict, combined output, ssh exit status).
    """
    seal_flag = " --seal" if seal else ""
    front_flag = "" if frontier is None else f" --frontier {frontier}"
    # POLARIS_API_KEY lives in a 0600 file on the box, not in this command line and not in
    # the repo: anything passed as an ssh argument shows up in ps on a shared machine.
    evalcmd = (
        f"cd {BOX_REPO_DIR} && "
        f"[ -r {POLARIS_ENV} ] && . {POLARIS_ENV}; "
        # CMake emits executables under build/runtime/, not build/. The harness resolves
        # $SPARKINFER_BUILD/kimi_k3_tp_bench, so pointing this at build/ makes it exit 2 with
        # "not built" immediately after a build that in fact succeeded.
        f"KIMI_K3_MODELS_DIR={BOX_MODELS_DIR} SPARKINFER_BUILD={BOX_REPO_DIR}/build/runtime "
        f"bash bench/scripts/kimi_k3_eval.sh --node {NODE} --devices {DEVICES}"
        f"{front_flag}{seal_flag}"
    )
    r = sh(evalcmd, timeout=7200)
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    line = next((l for l in out.splitlines() if l.strip().startswith("RESULT_JSON")), None)
    if not line:
        detail = (r.stderr or "").strip().splitlines()
        why = next((l for l in reversed(detail) if l.strip() and "setlocale" not in l), "")
        raise RuntimeError(
            f"no RESULT_JSON from the harness for {what}"
            + (f" — {why}" if why else "") + f"\n{out[-1200:]}")
    return json.loads(line.split("RESULT_JSON", 1)[1].strip()), out, r.returncode


def measure_frontier(repo):
    """Benchmark main ON THE SAME BOX, THIS ROUND, and return (sha, tok/s).

    This is the frontier every PR in the round is scored against. Measuring it beats pinning
    it for two independent reasons:

      It cannot go stale. A pinned number is only correct until the next perf PR merges, and
      nothing in the merge path was updating it -- which is exactly how #20 got an XL.

      It cancels the box. Thermal state, driver version and neighbours on a rented machine
      all move tok/s a few percent. main and the PR measured minutes apart on the same node
      share all of that; a constant measured on some other day shares none of it.

    The cost is one extra build+eval per round, amortised across every PR in it.
    """
    r = gh(["api", f"repos/{repo}/commits/main", "--jq", ".sha"])
    sha = (r.stdout or "").strip()
    if r.returncode != 0 or len(sha) < 7:
        raise RuntimeError(f"could not resolve main on {repo}: {r.stderr.strip()[:200]}")
    print(f"main: measuring the frontier at {sha[:8]} on {NODE} …")
    _box_build(sha, "main")
    # frontier=0 makes label.py return BASELINE, which is what we want: this run is not a
    # submission and must not be sealed or scored. Only its tps is used.
    res, _, _ = _box_eval("main", frontier=0, seal=False)
    tps = float(res.get("tps") or 0)
    if tps <= 0:
        raise RuntimeError(f"main measured {tps} tok/s — refusing to score a round against it")
    got = str(res.get("commit", "")).lower()
    if not sha.lower().startswith(got) or len(got) < 7:
        raise RuntimeError(f"harness measured commit {got!r} but main is {sha[:12]!r}")
    print(f"main: frontier = {tps} tok/s @ {sha[:8]}")
    return sha, tps, str(res.get("quant") or "UD-IQ1_S")


LOCK_PATH = "bench/scripts/reference.lock"
# Agreement band. Below this the measurement and the pin are the same number with noise on
# it, and rewriting the lock would be churn -- which is NOT free here: `main` has
# required_status_checks.strict, so every commit to it puts every open PR BEHIND and costs
# each author a rebase. A frontier commit has to be worth that.
FRONTIER_TOL = 0.02
# Below this fraction of the pin, "main got slower" stops being noise and becomes a fact
# that needs a human: either something regressed on main or the box is degraded. Both are
# reasons to stop, not to quietly re-baseline the thing that decides payouts.
FRONTIER_ALARM = 0.90


# The scored context is 131,072, so the frontier lives in the _128K slot. Keep this in step
# with kimi_k3_eval.sh's CTX_SUFFIX and eval-label.yml's slot(): all three read the same
# pin, and a mismatch means the bot writes one slot while CI scores from another.
CTX_SUFFIX = "128K"


def _frontier_slot(node, quant):
    pfx = "KIMI_K3_" + re.sub(r"[^A-Za-z0-9]", "", node).upper()
    q = re.sub(r"[^A-Za-z0-9]", "", re.sub(r"^UD-", "", quant or "")).upper()
    return f"{pfx}_{q}_SPARKINFER_{CTX_SUFFIX}"


def reconcile_lock(repo, node, quant, measured, main_sha, dry_run):
    """Make reference.lock's frontier agree with what main actually does.

    WHY THIS EXISTS. eval-label.yml deliberately does not trust the bot: it re-derives the
    tier from reference.lock on the protected branch and discards the payload's
    frontier_tps, so a PR cannot understate the frontier to inflate its own tier. That is
    the right design, and it means a measured frontier in this bot changes nothing on its
    own -- the pinned number IS the tier basis. #25 merged, main went 3.55 -> 9.54, the pin
    stayed 3.55, and CI stamped #20 eval:xl for a gain that is really S.

    WHAT IT COSTS. Writing here gives the bot commit access to the file that sets payout
    tiers, on a path CODEOWNERS and sensitive-paths-guard exist to protect. Three things
    keep that narrow: only the one SPARKINFER slot for this node+quant is ever rewritten,
    the value is always something just measured on the box (never computed or carried over),
    and the frontier is only ever raised automatically. Lowering it makes every subsequent
    PR's gain look bigger, so a throttled box could mint tiers for everyone behind it --
    that direction stops the round instead.

    Returns the frontier the round should score against, or None to abort.
    """
    slot = _frontier_slot(node, quant)
    r = gh(["api", f"repos/{repo}/contents/{LOCK_PATH}?ref=main"])
    if r.returncode != 0:
        print(f"!! could not read {LOCK_PATH} from main: {r.stderr.strip()[:200]}",
              file=sys.stderr)
        return None
    meta = json.loads(r.stdout or "{}")
    text = base64.b64decode(meta.get("content", "")).decode()
    pat = re.compile(rf'^({re.escape(slot)})="\$\{{\1:-([0-9.]+)\}}"\s*$', re.M)
    m = pat.search(text)
    if not m:
        print(f"!! {slot} is not in {LOCK_PATH} — refusing to guess where the frontier lives",
              file=sys.stderr)
        return None
    pinned = float(m.group(2))

    if pinned > 0 and measured < pinned * FRONTIER_ALARM:
        print(f"!! main measured {measured} tok/s but {slot} pins {pinned} — main is "
              f"{100 * (1 - measured / pinned):.1f}% slower than its own frontier.\n"
              f"   That is a regression on main or a degraded box, and lowering the frontier "
              f"would inflate every tier scored after it. Stopping.", file=sys.stderr)
        return None
    if pinned > 0 and abs(measured - pinned) <= max(0.01, FRONTIER_TOL * pinned):
        print(f">> frontier: {slot} pins {pinned}, main measures {measured} — agrees within "
              f"{FRONTIER_TOL:.0%}, leaving the lock alone")
        return pinned
    if pinned > 0 and measured < pinned:
        print(f">> frontier: main measures {measured}, below the pinned {pinned} but within "
              f"the alarm band — scoring against the pin (the conservative direction)")
        return pinned

    # Rewrite BY LINE, not by splicing the regex match out of the whole text. Splicing looked
    # fine and quietly ate the blank line after the slot, because the trailing-whitespace part
    # of the match ran into it. This file is sourced by bash and parsed by two other tools;
    # the only safe edit is one that provably touches one line and copies every other
    # verbatim.
    new = f'{slot}="${{{slot}:-{measured}}}"'
    stamp = (f"# frontier: measured by eval/k3_eval_bot.py on main @ {main_sha[:12]} "
             f"({node}, {quant}). Do not hand-edit — the bot rewrites this each round.")
    line_re = re.compile(rf'^{re.escape(slot)}="\$\{{{re.escape(slot)}:-[0-9.]+\}}"\s*$')
    out, seen = [], 0
    for ln in text.split("\n"):
        if ln.startswith("# frontier: measured by eval/k3_eval_bot.py"):
            continue                      # drop the old stamp; a fresh one goes back below
        if line_re.match(ln):
            out.extend((stamp, new))
            seen += 1
            continue
        out.append(ln)
    if seen != 1:
        print(f"!! {slot} matched {seen} lines in {LOCK_PATH} — refusing to rewrite",
              file=sys.stderr)
        return None
    updated = "\n".join(out)

    print(f">> frontier: {slot} {pinned} -> {measured} (main @ {main_sha[:8]})")
    if dry_run:
        print(f"--- dry-run: would commit {LOCK_PATH} to main")
        return measured
    q = gh(["api", "-X", "PUT", f"repos/{repo}/contents/{LOCK_PATH}",
            "-f", f"message=eval: frontier {pinned} -> {measured} tok/s "
                  f"(main @ {main_sha[:8]}, {node}/{quant})",
            "-f", f"content={base64.b64encode(updated.encode()).decode()}",
            "-f", f"sha={meta['sha']}", "-f", "branch=main"], timeout=180)
    if q.returncode != 0:
        print(f"!! could not commit the frontier to main: {q.stderr.strip()[:300]}\n"
              f"   Refusing to score against a frontier CI will not agree with.",
              file=sys.stderr)
        return None
    print(f">> {LOCK_PATH} updated on main — eval-label.yml will re-derive against {measured}")
    print(f"   note: main moved, so every open PR is now BEHIND (branch protection is "
          f"strict) and needs a rebase before it can merge")
    return measured


def update_pr_branch(repo, pr, dry_run, tries=12, delay=5):
    """Bring a PR branch up to date with main, and return its NEW head sha.

    THE BOT CAUSED THIS. Committing the measured frontier moves main, and with
    required_status_checks.strict every open PR is BEHIND the moment that lands. Labelling
    them needs-rebase and waiting is asking contributors to hand-fix bookkeeping the bot
    did to them -- and it is the reason nothing merged: needs-rebase is in
    NEVER_MERGE_LABELS, so the frontier commit blocked the very PR it was measured for.

    The update is a merge of main into the branch, which changes the head sha. That is why
    this runs BEFORE the PR is measured: the payload's commit has to be the head that
    eval-label.yml sees, or it refuses the result as measured on another commit. Measure the
    thing that will actually merge.

    Fork PRs need maintainer_can_modify (GitHub's "allow edits by maintainers"); without it
    the API refuses and the author genuinely does have to rebase.
    """
    num = pr["number"]
    if dry_run:
        print(f"--- dry-run: would update #{num} onto main")
        return pr["headRefOid"]
    before = pr["headRefOid"]
    r = gh(["api", "-X", "PUT", f"repos/{repo}/pulls/{num}/update-branch",
            "-f", "expected_head_sha=" + before], timeout=120)
    if r.returncode != 0:
        err = (r.stderr or "").strip()[:200]
        print(f"!! #{num}: could not update the branch onto main — {err}\n"
              f"   (a fork PR needs 'allow edits by maintainers'); leaving it to the author",
              file=sys.stderr)
        return None
    # The new head appears asynchronously. Poll for it rather than assuming: measuring the
    # old sha would produce a result eval-label.yml rejects, after a 40-minute node run.
    for _ in range(tries):
        time.sleep(delay)
        q = gh(["pr", "view", str(num), "-R", repo, "--json", "headRefOid,mergeStateStatus"])
        try:
            info = json.loads(q.stdout or "{}")
        except json.JSONDecodeError:
            continue
        head, state = info.get("headRefOid", ""), info.get("mergeStateStatus")
        if head and head != before and state != "BEHIND":
            print(f">> #{num}: branch updated onto main, {before[:8]} -> {head[:8]}")
            return head
    print(f"!! #{num}: branch update did not settle in {tries * delay}s — not measuring a "
          f"head that may still move", file=sys.stderr)
    return None


def evaluate(pr, repo, frontier, seal=False):
    """Check the PR head out on the box, build it, and run the K3 harness there.

    Returns the parsed RESULT_JSON, or raises RuntimeError with the box's own output. The
    build and the eval both happen on the node; nothing is measured locally.
    """
    sha = pr["headRefOid"]
    num = pr["number"]
    _box_build(sha, f"#{num}")
    res, out, rc = _box_eval(f"#{num}", frontier=frontier, seal=seal)
    # --seal publishes to sparkinfer-k3-log and prints the receipt id. Carry it in the
    # payload: eval-label.yml looks it up there when REQUIRE_EVAL_RECEIPT is on, and
    # without it a sealed run is indistinguishable from an unsealed one.
    if seal:
        # kimi_k3_eval.sh prints RESULT_JSON and THEN seals, exiting 1 if the publish fails.
        # Scraping stdout and ignoring the exit status is how a run got reported as sealed
        # while the log stayed empty: the harness said "FAILED to seal/publish" and nothing
        # listened.
        rid = re.search(r"receipt[_ ]?id[\"'\s:=]+([0-9a-f]{8,})", out, re.I)
        if rid:
            # SIGNING AND PUBLISHING ARE SEPARATE, and only signing has to happen on the
            # box. kimi_k3_attest.py --publish git-pushes to the log, which needs write
            # credentials wherever it runs; the box has none and should not be given any,
            # since a rented shared machine that can rewrite the ledger it is judged
            # against is a worse trade than one that can only sign.
            #
            # So a failed box-side publish is expected, not fatal: the receipt is minted
            # and signed on the box either way, and publish_receipt() copies it here and
            # commits it with credentials that never left this machine. Treating the
            # harness's exit code as "not sealed" threw away a perfectly good receipt and
            # made the controller path unreachable -- every run reported 0 sealed while a
            # signed receipt sat on the box.
            res["receipt_id"] = rid.group(1)
            if rc != 0:
                print(f"#{num}: box-side publish failed (expected — no git credentials "
                      f"there); publishing {rid.group(1)[:12]} from here instead")
        else:
            print(f"#{num}: warning — --seal was requested but no receipt id appeared in "
                  f"the harness output; the run is NOT in the log", file=sys.stderr)

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


# Fields eval-label.yml derives itself from reference.lock on the PROTECTED branch. The bot
# must not post them. It runs the harness from the PR's own checkout, so these come out of
# whatever reference.lock that branch happens to carry -- which for an older branch is the
# pre-#24 lookup that resolves the frontier to 0 and labels everything BASELINE. Posting a
# label computed that way makes eval-label hard-error "reported label != re-derived" on a
# perfectly good run, and posting a frontier at all invites the exact substitution #35
# closed. Strip them and let the trusted side compute the verdict.
DERIVED_BY_CI = ("label", "speed_label", "frontier_tps", "note", "pct_over_frontier",
                 "pct_of_llama", "delta_tps", "tier_basis", "pass",
                 # label.py grew these after the list was written, and a
                 # hand-maintained denylist silently misses whatever is added
                 # next. Asserted against label.py's real output by a test.
                 "pct_of_ceiling", "effective_pct")


def post(repo, num, res, dry_run):
    res = {k: v for k, v in res.items() if k not in DERIVED_BY_CI}
    # RESULT_JSON must stay on the FIRST line: eval-label.yml gates on
    # startsWith(comment.body, '/eval') and then sed-scrapes the object off one line. The
    # readable form goes underneath, and the raw payload is repeated in a collapsed block
    # so a human can audit exactly what the workflow parsed.
    rid = res.get("receipt_id", "")
    rows = [
        ("decode tok/s", res.get("tps")),
        ("ms / token", res.get("ms_per_token")),
        ("top-1 vs llama.cpp", res.get("top1")),
        ("mean KLD vs llama.cpp", res.get("kl")),
        ("commit", f"`{res.get('commit','')}`"),
        ("node / devices", f"{res.get('node','')} · {res.get('layers','')} layers"),
        ("quant", res.get("quant")),
        ("receipt", f"`{rid}`" if rid else "_unsealed_"),
    ]
    table = "\n".join(f"| {k} | {v} |" for k, v in rows if v not in (None, ""))
    body = (
        "/eval RESULT_JSON " + json.dumps(res, separators=(",", ":")) + "\n\n"
        "### Node measurement\n\n"
        "| metric | value |\n|---|--:|\n" + table + "\n\n"
        "Measured on the pinned 8x H200 node by `eval/k3_eval_bot.py`. **These are inputs, "
        "not the verdict** — the tier is re-derived from `bench/scripts/reference.lock` on "
        "`main` by `eval-label.yml`, so nothing this comment claims can set a payout.\n\n"
        "<details><summary>raw payload as parsed</summary>\n\n```json\n"
        + json.dumps(res, indent=2) + "\n```\n</details>"
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


def publish_receipt(repo_log, num, res, box_out, dry_run):
    """Push the sealed run into the log FROM HERE, not from the box.

    kimi_k3_attest.py --publish clones the log and git-pushes, which needs write credentials
    wherever it runs. The eval box is rented and shared, so giving it a token that can write
    the ledger it is being judged against is the wrong trade -- a Polaris key that can only
    sign is a much smaller thing to leave lying around than one that can rewrite history.

    So the box seals (mints and signs) and the controller publishes, using credentials that
    never leave this machine. The Contents API also avoids cloning an ever-growing log just
    to append one directory.
    """
    rid = res.get("receipt_id")
    if not rid:
        return False
    # Match on the receipt id INSIDE the file. The filenames are timestamp-based
    # (k3_<node>_<stamp>.receipt.json) and carry no id, so a glob on the id never matches
    # and any fallback to "newest file" quietly publishes whichever run finished last --
    # which, with two PRs evaluated back to back, is a coin flip on the second one.
    got = sh(f"grep -l '\"{rid}\"' {box_out}/*.receipt.json 2>/dev/null | head -1 | "
             f"xargs -r cat", timeout=120)
    body = (got.stdout or "").strip()
    if not body.startswith("{"):
        print(f"#{num}: sealed but the receipt could not be read back from the box — "
              f"not publishing a run I cannot show", file=sys.stderr)
        return False
    if dry_run:
        print(f"--- dry-run: would publish runs/{rid} to {repo_log}")
        return True
    for name, payload in (("receipt.json", body),
                          ("result.json", json.dumps(res, indent=2, sort_keys=True))):
        enc = base64.b64encode(payload.encode()).decode()
        q = gh(["api", "-X", "PUT", f"repos/{repo_log}/contents/runs/{rid}/{name}",
                "-f", f"message=eval {rid}: tps={res.get('tps')} (PR #{num})",
                "-f", f"content={enc}"])
        if q.returncode != 0:
            # Append-only: an existing path is a refusal, not a failure to fix by forcing.
            print(f"#{num}: could not publish {name}: {q.stderr.strip()[:160]}",
                  file=sys.stderr)
            return False
    print(f">> #{num}: published runs/{rid} to {repo_log}")
    return True


# Which merge states each path may proceed through.
#
# The distinction that matters is PERMISSION gates versus CONTENT gates.
#
#   BLOCKED is a permission gate: the required approving review has not happened yet (or a
#   required check is still pending). Bypassing precisely that is what --admin IS for. The
#   strict list refused it, and since `main` always requires a review, BLOCKED is where every
#   PR here permanently sits -- so --merge-admin could never merge anything at all. It failed
#   on #20 twice for this reason, reported as "blocked by a required check or review", which
#   reads like a real objection rather than the flag refusing its own purpose.
#
#   BEHIND is a content gate: the branch does not contain main, so the tier measured on it
#   was scored against a frontier that has since moved. Admin rights do not make that number
#   true. The bot fixes this by updating the branch and RE-MEASURING, never by merging
#   through it.
#
#   DIRTY is conflicts -- there is nothing to bypass, the merge would be a guess.
#   UNKNOWN is "GitHub has not finished computing mergeability": not evidence, and it is what
#   #20 reported seconds after #25 merged while actually being DIRTY.
MERGE_STATES = {
    "strict": ("CLEAN", "UNSTABLE"),
    "admin": ("CLEAN", "UNSTABLE", "BLOCKED"),
}


class _Tee:
    """Mirror everything the round prints into a buffer so it can be published.

    A receipt proves a number was signed. It says nothing about what the round DID to get
    there -- whether the frontier moved and to what, which build failed first, why a merge
    was refused. That context existed only in the operator's terminal, which is exactly
    where someone checking the number afterwards cannot look.
    """

    def __init__(self, stream, buf):
        self._s, self._b = stream, buf

    def write(self, s):
        self._s.write(s)
        self._b.write(s)
        return len(s)

    def flush(self):
        self._s.flush()

    def isatty(self):
        return self._s.isatty()


def publish_log(repo_log, path, text, dry_run, label=""):
    """Publish a round log to the immutable log repo at `path`.

    Append-only, like the receipt itself: an existing path is a refusal, not something to
    force.

    Two shapes of path, and the second exists because the first cannot cover every round:

      runs/<receipt_id>/eval.log   a round that SEALED. The measurement, the verdict and
                                   the session that produced them arrive together, so a
                                   receipt is self-describing.
      rounds/<main_sha>/eval.log   a round that sealed NOTHING. Indexing only by receipt id
                                   meant exactly the rounds worth reading vanished: the one
                                   that died on GPU contention mid-load left no trace,
                                   because a failed round mints no receipt. Same for a
                                   correctness fix, which scores `none`, is never sealed,
                                   and so was permanently unauditable.
    """
    if not path or not text.strip():
        return False
    what = label or path
    if dry_run:
        print(f"--- dry-run: would publish {path} to {repo_log}")
        return True
    q = gh(["api", "-X", "PUT", f"repos/{repo_log}/contents/{path}",
            "-f", f"message=eval {what}: round log",
            "-f", f"content={base64.b64encode(text.encode()).decode()}"], timeout=180)
    if q.returncode != 0:
        print(f"!! could not publish {path}: {q.stderr.strip()[:160]}", file=sys.stderr)
        return False
    print(f">> published {path} to {repo_log}")
    return True


def publish_round_log(repo_log, main_sha, results, text, dry_run):
    """EVERY round leaves a record. Sealed runs get the log beside their receipt; a round
    that sealed nothing still gets one under rounds/<main_sha>/.

    The rule this enforces: a receipt proves a number was signed, and says nothing about
    what the round did to produce it. That context used to exist only in the operator's
    terminal, which is precisely where someone auditing a payout-bearing number cannot look.
    A round that FAILED is the most informative of all and was the one guaranteed to be lost.

    Returns the number of paths written.
    """
    n = 0
    for num, r in results:
        rid = r.get("receipt_id")
        if rid and publish_log(repo_log, f"runs/{rid}/eval.log", text, dry_run, rid):
            n += 1
    if n:
        return n
    # Nothing sealed. Fall back to a round-level path so the log still lands. Keyed by the
    # main commit the round measured against, which is the only identifier a failed round
    # reliably has -- and a stable one, so a re-run against the same main is a refusal rather
    # than a silent overwrite of the first attempt's evidence.
    if not main_sha:
        print("!! round sealed nothing and main is unknown — no log path to publish under",
              file=sys.stderr)
        return 0
    return 1 if publish_log(repo_log, f"rounds/{main_sha[:12]}/eval.log", text, dry_run,
                            f"round @ {main_sha[:8]}") else 0


def merge_blockers(repo, num, pr, waiting_is_blocking=True, mode="strict"):
    """Every reason not to merge this without a human reading it. Empty list means clear.

    Two kinds of reason, and they are not the same kind of thing:

      SUBSTANTIVE -- a label, a maintainer-owned path, a missing tier. These say a human has
      to look at this PR. No amount of waiting resolves them.

      WAITING -- the merge state: behind, conflicted, or blocked on the required review.
      These resolve by themselves when someone approves or the author rebases.

    --merge-admin must respect both, because it merges NOW. Queued auto-merge only respects
    the substantive ones (waiting_is_blocking=False): waiting for the review is precisely
    what it is for, and refusing to queue because the review has not happened yet would make
    the flag do nothing. The substantive guards still apply -- queueing auto-merge on a PR
    labelled `hold` or `copycat` would hand it a merge the moment someone clicked approve.
    """
    bad = []
    # Read labels FRESH rather than from the snapshot taken when the round started. The round
    # mutates them -- merge-first goes on, needs-rebase comes off once a branch is updated --
    # so deciding from the opening snapshot refuses a merge over a label that no longer
    # exists. That is precisely what stopped #20: the snapshot still said needs-rebase.
    st = gh(["pr", "view", str(num), "-R", repo, "--json", "mergeStateStatus,labels"])
    try:
        live = json.loads(st.stdout or "{}")
    except json.JSONDecodeError:
        live = {}
    labels = {l.get("name", "") for l in (live.get("labels") or pr.get("labels") or [])}
    hit = labels & NEVER_MERGE_LABELS
    if hit:
        bad.append(f"labels {sorted(hit)}")
    if not any(l.startswith("eval:") for l in labels):
        bad.append("no eval:* tier — it has not passed the gate")
    r = gh(["pr", "diff", str(num), "-R", repo, "--name-only"])
    files = [f for f in (r.stdout or "").split() if f]
    if r.returncode != 0 or not files:
        bad.append("could not read the diff")
    touched = sorted({f for f in files if f.startswith(NEVER_MERGE_PATHS)})
    if touched:
        bad.append(f"touches maintainer-owned paths: {touched[:4]}")
    if not waiting_is_blocking:
        return bad
    # Allowlist the states that are safe, rather than denylisting the bad ones.
    #
    # BEHIND is the important one and is easy to read as harmless: the branch merely trails
    # main. But it trails main because something merged, which is exactly when the frontier
    # moved, so the tier on it was computed against a baseline that no longer exists. That
    # PR needs re-measuring, not merging.
    #
    # UNKNOWN is not "fine", it is "GitHub has not finished computing mergeability" --
    # asynchronous, and it is what #20 reported seconds after #25 merged while actually
    # being DIRTY. Treating unknown as clear is how a conflicted branch gets merged.
    state = live.get("mergeStateStatus") or "UNKNOWN"
    if state not in MERGE_STATES.get(mode, MERGE_STATES["strict"]):
        bad.append({
            "BEHIND": "behind main — its tier predates the current frontier",
            "DIRTY": "merge conflicts",
            "BLOCKED": "blocked by a required check or review",
            "UNKNOWN": "mergeability not yet computed by GitHub",
        }.get(state, f"merge state {state}"))
    return bad


def merge_winner(repo, num, pr, dry_run):
    """Merge the round's winner with admin rights, or say exactly why it was not.

    Admin merging bypasses the approving-review requirement on `main`. That is an explicit
    operator choice, not a default -- it trades human review for throughput on a change
    carrying a payout tier -- so a blocked merge is reported loudly rather than skipped
    quietly. A silent no-op here would be indistinguishable from a successful merge in the
    run log, and that is exactly how an unreviewed merge goes unnoticed.
    """
    blockers = merge_blockers(repo, num, pr, mode="admin")
    if blockers:
        print(f"!! #{num}: NOT merged — {'; '.join(blockers)}", file=sys.stderr)
        return False
    if dry_run:
        print(f"--- dry-run: would admin-merge #{num}")
        return True
    r = gh(["pr", "merge", str(num), "-R", repo, "--squash", "--admin"], timeout=180)
    if r.returncode != 0:
        print(f"!! #{num}: admin merge failed: {r.stderr.strip()[:200]}", file=sys.stderr)
        return False
    print(f">> #{num}: merged (admin) — round's largest verified gain")
    return True


def sync_rebase_labels(repo, prs, merged_num, dry_run):
    """After a merge, every other open PR is scored against a frontier that just moved.

    Their tier is stale by definition rather than by suspicion: the denominator changed.
    So they get the label and an explanation. It clears automatically once the branch is no
    longer behind -- a label only a maintainer can remove turns a mechanical state into a
    queue somebody has to babysit, and the miner has already done the work by rebasing.
    """
    for pr in prs:
        num = pr["number"]
        if num == merged_num:
            continue
        st = gh(["pr", "view", str(num), "-R", repo, "--json", "mergeStateStatus,state,labels"])
        try:
            info = json.loads(st.stdout or "{}")
        except json.JSONDecodeError:
            continue
        if info.get("state") != "OPEN":
            continue
        labels = {l.get("name", "") for l in info.get("labels") or []}
        # The label means ONE thing: "your branch trails main, so the tier on it was measured
        # against a frontier that has moved". Only BEHIND and DIRTY say that about the branch.
        #
        # BLOCKED does not, and treating it as stale deadlocked #20: BLOCKED is what GitHub
        # reports for "waiting on the required approving review", which is the branch's normal
        # resting state here. So the label could never clear, and because needs-rebase is in
        # NEVER_MERGE_LABELS it also blocked the merge -- the label survived on the strength of
        # the review it was helping to prevent, while the author had already rebased.
        #
        # UNKNOWN means GitHub has not finished computing mergeability. That is not evidence
        # either way, so it neither sets nor clears: acting on it would label people over an
        # API race.
        state = info.get("mergeStateStatus")
        stale = state in ("BEHIND", "DIRTY")
        current = state in ("CLEAN", "UNSTABLE", "BLOCKED")
        behind = stale
        has = REBASE_LABEL in labels
        # The label means "your measured tier is stale because the frontier moved". A PR
        # with no eval:* tier has no stale tier -- telling its author to rebase because of
        # a scoring change that never applied to them is noise, and noise on a label people
        # are asked to act on is how labels stop being read at all. Clearing still applies,
        # so a tier removed later does not strand the label.
        if behind and not any(l.startswith("eval:") for l in labels):
            continue
        if behind and not has:
            if dry_run:
                print(f"--- dry-run: would label #{num} {REBASE_LABEL}")
                continue
            gh(["label", "create", REBASE_LABEL, "-R", repo, "--color", "FBCA04",
                "--description", "frontier moved — rebase and it is re-evaluated"])
            q = gh(["api", "-X", "POST", f"repos/{repo}/issues/{num}/labels",
                    "-f", f"labels[]={REBASE_LABEL}"])
            if q.returncode != 0:
                print(f"!! #{num}: could not label {REBASE_LABEL}: {q.stderr.strip()[:120]}",
                      file=sys.stderr)
                continue
            # merged_num is -1 when this is a standalone sweep rather than the tail of a
            # merge, and interpolating the sentinel put a literal "#-1 merged" in front of
            # a contributor. A sentinel that reaches a human is a bug regardless of what
            # the code does with it.
            cause = (f"#{merged_num} merged, so the frontier" if merged_num > 0
                     else "The frontier")
            gh(["pr", "comment", str(num), "-R", repo, "--body",
                f"### Rebase needed\n\n{cause} this PR is "
                f"scored against has moved. The tier currently on it was measured against "
                f"the old baseline and no longer means anything: you are paid for the gain "
                f"**on top of what merged**, not the gain over where `main` used to be. If "
                f"the merged work already covers this change, the honest re-measurement is "
                f"a small number, and that is the mechanism working rather than a "
                f"penalty.\n\nRebase onto `main` and push. The `{REBASE_LABEL}` label "
                f"clears by itself once the branch is current, and the next round "
                f"re-measures against the new frontier."])
            print(f">> #{num}: {REBASE_LABEL} (frontier moved)")
        elif has and current:
            if dry_run:
                print(f"--- dry-run: would clear {REBASE_LABEL} from #{num}")
                continue
            gh(["api", "-X", "DELETE", f"repos/{repo}/issues/{num}/labels/{REBASE_LABEL}"])
            print(f">> #{num}: {REBASE_LABEL} cleared — branch is current again")


def mark_merge_first(repo, results, dry_run, queue_auto_merge=False, prs=()):
    """Label the round's biggest verified gain `merge-first`, and clear it from the rest.

    The winner is by measured tok/s against the SAME frontier, which is the only comparison
    that means anything: two PRs evaluated in one round share a baseline, so the faster one
    genuinely is the larger gain. Once it merges the frontier moves, and everything else has
    to be re-evaluated against the new one -- which is the point of merging one at a time.

    NOTE ON AUTO-MERGE. This never merges directly and never passes --admin. `main` requires
    an approving review, and a bot approving the PR it merges is not a review. What
    --auto-merge does is QUEUE GitHub's native auto-merge, so the merge fires by itself once
    every requirement is satisfied -- including that human approval. The autonomy is in the
    waiting, not in the bypassing.
    """
    ranked = sorted(results, key=lambda r: r[1].get("tps") or 0, reverse=True)
    if not ranked:
        return
    winner = ranked[0][0]
    for num, res in ranked:
        has = num == winner
        act = "--add-label" if has else "--remove-label"
        if dry_run:
            print(f"--- dry-run: would {act} merge-first on #{num} "
                  f"(tps={res.get('tps')})")
            continue
        if has:
            gh(["label", "create", "merge-first", "-R", repo, "--color", "0E8A16",
                "--description", "round's largest verified gain — merge this one first"])
        # NOT `gh pr edit --add-label`: it queries projectCards, which GitHub has
        # deprecated, so it exits 1 even when the repo has no projects. The bot printed
        # ">> merge-first: #25" while the label never landed, because nothing checked the
        # return code. The issues REST endpoint touches no GraphQL.
        if has:
            q = gh(["api", "-X", "POST", f"repos/{repo}/issues/{num}/labels",
                    "-f", "labels[]=merge-first"])
        else:
            q = gh(["api", "-X", "DELETE", f"repos/{repo}/issues/{num}/labels/merge-first"])
        if q.returncode != 0 and has:
            print(f"!! #{num}: failed to set merge-first: {q.stderr.strip()[:200]}",
                  file=sys.stderr)
    print(f">> merge-first: #{winner} ({ranked[0][1].get('tps')} tok/s)")
    if not queue_auto_merge:
        return
    # Queueing is not merging, but it is arming: once the required review lands, GitHub
    # merges with nobody looking again. So the substantive guards apply here too -- the
    # merge-state ones do not, because waiting for them IS the feature.
    blockers = merge_blockers(repo, winner, {p["number"]: p for p in prs}.get(winner, {}),
                              waiting_is_blocking=False)
    if blockers:
        print(f"!! #{winner}: NOT queued for auto-merge — {'; '.join(blockers)}",
              file=sys.stderr)
        return
    if dry_run:
        print(f"--- dry-run: would queue auto-merge on #{winner}")
        return
    r = gh(["pr", "merge", str(winner), "-R", repo, "--squash", "--auto"])
    if r.returncode == 0:
        print(f">> #{winner}: auto-merge queued — fires when review + checks pass")
    else:
        print(f"!! #{winner}: could not queue auto-merge: {r.stderr.strip()}",
              file=sys.stderr)


def main():
    # Mirror the round into a buffer from the first line, so what gets published is the whole
    # session and not the tail of it. Both streams share one buffer: the interesting parts of
    # a bad round are on stderr, and a log that drops them reads like a clean run.
    log_buf = io.StringIO()
    _real_out, _real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(_real_out, log_buf), _Tee(_real_err, log_buf)

    ap = argparse.ArgumentParser(description="Measure eligible K3 PRs on the H200 node and post the result.")
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--only-pr", type=int, default=0, help="evaluate just this PR number")
    ap.add_argument("--dry-run", action="store_true", help="print the comment instead of posting")
    ap.add_argument("--no-seal", action="store_true",
                    help="skip attestation. Sealing is the default: an unsealed run leaves "
                         "no record in sparkinfer-k3-log, so nothing outside this terminal "
                         "can corroborate the number that set someone's tier.")
    ap.add_argument("--merge-first", action="store_true",
                    help="label the round's largest gain merge-first")
    ap.add_argument("--merge-admin", action="store_true",
                    help="MERGE the merge-first winner with admin rights, bypassing the "
                         "approving-review requirement on main, then label the other open "
                         "PRs needs-rebase. This lands an emissions-bearing change with no "
                         "human reading it; the guards in NEVER_MERGE_LABELS and "
                         "NEVER_MERGE_PATHS are what remain.")
    ap.add_argument("--auto-merge", action="store_true",
                    help="queue GitHub native auto-merge on the merge-first winner. Waits "
                         "for the required review and checks; never bypasses them.")
    ap.add_argument("--rebase-sweep", action="store_true",
                    help="only reconcile needs-rebase labels against each PR's merge state "
                         "and exit — no node run. Cheap enough to poll, which is what makes "
                         "the label clear itself once a miner has rebased.")
    ap.add_argument("--frontier", type=float, default=None,
                    help="skip the main benchmark and score against this tok/s. For re-runs "
                         "within a round; the lock is still reconciled against it.")
    ap.add_argument("--no-lock-update", action="store_true",
                    help="measure main but never write reference.lock. The posted number "
                         "will then disagree with the tier eval-label.yml re-derives from "
                         "the pinned frontier, so this is for inspection, not for a round "
                         "that is meant to set tiers.")
    ap.add_argument("--publish-log", metavar="RECEIPT_ID",
                    help="publish a round log against an EXISTING receipt and exit. For "
                         "backfilling a run that was sealed before the log was captured, or "
                         "whose PR has since merged and can no longer be re-evaluated.")
    ap.add_argument("--log-file", help="log to publish with --publish-log ('-' for stdin)")
    ap.add_argument("--list", action="store_true", help="show eligibility for every open PR and exit")
    args = ap.parse_args()

    # This bot is K3-specific in the same way the old one was Qwen-specific. Pointing it at
    # another repo would check out that repo's head on a box holding 554 GiB of K3 weights
    # and score whatever came out against K3's frontier.
    if not re.search(r"-k3(/|$)", args.repo):
        sys.stderr.write(f"k3_eval_bot: refusing to run against {args.repo!r} — this bot only "
                         f"evaluates Kimi K3 repositories.\n")
        return 2

    if args.publish_log:
        # Backfill. Kept separate from a round on purpose: it publishes a log somebody hands
        # it rather than one it produced, so it must never be confused with the automatic
        # path. It refuses unless the receipt already exists -- the log is an annotation on a
        # sealed run, and one filed against an id with nothing behind it is worse than
        # absent, since a reader has no measurement to check it against.
        if not args.log_file:
            sys.stderr.write("k3_eval_bot: --publish-log needs --log-file\n")
            return 2
        rid = args.publish_log
        q = gh(["api", f"repos/{LOG_REPO}/contents/runs/{rid}", "--silent"])
        if q.returncode != 0:
            sys.stderr.write(f"k3_eval_bot: no receipt {rid} in {LOG_REPO} — refusing to file "
                             f"a log against a run that was never sealed\n")
            return 1
        try:
            text = sys.stdin.read() if args.log_file == "-" else open(args.log_file).read()
        except OSError as exc:
            sys.stderr.write(f"k3_eval_bot: cannot read {args.log_file}: {exc}\n")
            return 1
        return 0 if publish_log(LOG_REPO, f"runs/{rid}/eval.log", text,
                                args.dry_run, rid) else 1

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

    if args.rebase_sweep:
        # Deliberately cheap and separable from evaluation. The label is supposed to clear
        # itself the moment a miner rebases, and that only works if checking costs seconds
        # -- requiring a 40-minute node eval to notice someone already did what they were
        # asked would make the self-clearing property useless in practice, and leave people
        # sitting under a stale label they cannot remove.
        sync_rebase_labels(args.repo, prs, merged_num=-1, dry_run=args.dry_run)
        return 0

    eligible = [p for p in prs if eligibility(p)[0]]
    for pr in prs:
        ok, why = eligibility(pr)
        if not ok:
            print(f"#{pr['number']}: skip — {why}")
    if not eligible:
        print("nothing eligible — not booking the node for a frontier measurement")
        return 0

    # Every exit from here on has booked the node, so every one of them owes a published log.
    # The early returns below are the FAILED rounds -- a frontier that would not measure, a
    # lock that would not reconcile -- and those are the logs most worth reading, so they must
    # not be the ones that silently never land.
    def _bail(code, sha=""):
        sys.stdout, sys.stderr = _real_out, _real_err
        publish_round_log(LOG_REPO, sha, [], log_buf.getvalue(), args.dry_run)
        return code

    # THE FRONTIER IS MEASURED, NOT PINNED. Do this before anything is posted: eval-label.yml
    # reads reference.lock at comment time, so a comment posted before the lock is reconciled
    # gets a tier derived from the old frontier -- which is the whole failure being fixed.
    try:
        if args.frontier is not None:
            main_sha, main_tps = "", float(args.frontier)
            quant = "UD-IQ1_S"
            print(f">> frontier: {main_tps} tok/s (--frontier, main not re-measured)")
            r = gh(["api", f"repos/{args.repo}/commits/main", "--jq", ".sha"])
            main_sha = (r.stdout or "").strip()
        else:
            main_sha, main_tps, quant = measure_frontier(args.repo)
    except RuntimeError as exc:
        print(f"frontier measurement failed — {exc}", file=sys.stderr)
        # main_sha may be blank here (the resolve itself can fail); publish_round_log says
        # so loudly rather than inventing a path.
        return _bail(1, locals().get("main_sha", ""))

    if args.no_lock_update:
        print(">> --no-lock-update: reference.lock untouched; the applied tier will come "
              "from the pinned frontier, not this one", file=sys.stderr)
        frontier = main_tps
    else:
        frontier = reconcile_lock(args.repo, NODE, quant, main_tps, main_sha, args.dry_run)
        if frontier is None:
            return _bail(3, main_sha)

    # main may have just moved under these branches -- and if reconcile_lock committed, it
    # moved BECAUSE of this round. Bring them onto it before measuring rather than labelling
    # them needs-rebase and waiting: the bot caused the staleness, the head sha changes when
    # it is fixed, and the number has to be taken on the commit that will actually merge.
    for pr in list(eligible):
        num = pr["number"]
        st = gh(["pr", "view", str(num), "-R", args.repo, "--json", "mergeStateStatus"])
        try:
            state = json.loads(st.stdout or "{}").get("mergeStateStatus")
        except json.JSONDecodeError:
            state = None
        if state != "BEHIND":
            continue
        head = update_pr_branch(args.repo, pr, args.dry_run)
        if head is None:
            print(f"#{num}: skip — behind main and the branch could not be brought current, "
                  f"so any number measured here would be scored against the wrong frontier")
            eligible.remove(pr)
            continue
        pr["headRefOid"] = head

    if not eligible:
        print("nothing left to evaluate after reconciling branches")
        return _bail(0, main_sha)

    evaluated = 0
    results = []
    for pr in eligible:
        num = pr["number"]
        print(f"#{num}: evaluating {pr['headRefOid'][:8]} on {NODE} "
              f"against frontier {frontier} …")
        try:
            res = evaluate(pr, args.repo, frontier, seal=not args.no_seal)
        except RuntimeError as exc:
            print(f"#{num}: eval failed — {exc}", file=sys.stderr)
            continue
        print(f"#{num}: tps={res.get('tps')} top1={res.get('top1')} kl={res.get('kl')} "
              f"ms/token={res.get('ms_per_token')} — tier is eval-label.yml's to derive")
        if res.get("receipt_id"):
            publish_receipt(LOG_REPO, num, res, BOX_RECEIPTS, args.dry_run)
        post(args.repo, num, res, args.dry_run)
        results.append((num, res))
        evaluated += 1

    # Reconcile the stale-tier labels BEFORE any merge decision, not after. needs-rebase is
    # in NEVER_MERGE_LABELS, so one left over from before the branches were brought current
    # blocks the merge it was never meant to block. That is exactly how #20 ended up labelled
    # merge-first and then refused in the same round.
    if args.merge_first or args.auto_merge or args.merge_admin:
        sync_rebase_labels(args.repo, prs, merged_num=-1, dry_run=args.dry_run)

    if args.merge_first or args.auto_merge or args.merge_admin:
        mark_merge_first(args.repo, results, args.dry_run,
                         queue_auto_merge=args.auto_merge, prs=prs)

    if args.merge_admin and results:
        winner = max(results, key=lambda r: r[1].get("tps") or 0)[0]
        by_num = {p["number"]: p for p in prs}
        if merge_winner(args.repo, winner, by_num.get(winner, {}), args.dry_run):
            # Only after something actually merged does the frontier move, so the rebase
            # sweep is conditional on the merge -- labelling everything needs-rebase after a
            # merge that was blocked would tell every contributor to redo work for nothing.
            sync_rebase_labels(args.repo, prs, winner, args.dry_run)
    elif args.merge_admin:
        print("no results to merge", file=sys.stderr)

    # A receipt id proves a receipt was minted, not that anyone can find it. The log is the
    # only thing that makes a number checkable by someone who was not at this terminal, so
    # ask the log rather than trusting the id.
    sealed = 0
    for num, r in results:
        rid = r.get("receipt_id")
        if not rid:
            continue
        q = gh(["api", f"repos/{LOG_REPO}/contents/runs/{rid}", "--silent"])
        if q.returncode == 0:
            sealed += 1
        else:
            r.pop("receipt_id", None)
            print(f"#{num}: receipt {rid[:12]} is NOT in {LOG_REPO} — the run is unattested",
                  file=sys.stderr)
    print(f"done — {evaluated} evaluated, {sealed} sealed to the log; "
          f"tiers are eval-label.yml's")

    # Publish the round's own output alongside each receipt. Last, so the log contains the
    # whole round: the frontier measurement, the branch updates, the merge decision and the
    # reason for it. Restore the real streams first -- the publish output describes the
    # publish and belongs on the terminal, not recursively inside the artefact.
    sys.stdout, sys.stderr = _real_out, _real_err
    if not publish_round_log(LOG_REPO, main_sha, results, log_buf.getvalue(), args.dry_run):
        print("!! the round log was NOT published — this round is unauditable outside this "
              "terminal", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
