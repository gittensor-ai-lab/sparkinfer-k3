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
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse

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
                      "needs-benchmark", "needs-node-run", "llm-judge", "needs-rebase",
                      # Applied by the round itself when a PR's KL is >= KL_RATCHET_REJECT x
                      # main's, measured on the same box minutes apart. A speed win bought
                      # with parity is a trade a human should make on purpose.
                      "accuracy-regression"}
# Paths that decide payouts. sensitive-paths-guard already fails a non-maintainer PR that
# touches these, but a bot that merges without anyone reading should not depend on another
# check having run correctly.
NEVER_MERGE_PATHS = ("eval/", ".github/", ".gittensor/", "bench/scripts/", "bench/results/",
                     "bench/refdata/", "dashboard/", "CODEOWNERS")
REBASE_LABEL = "needs-rebase"
CONFLICT_LABEL = "merge-conflict"

# THE TIERS THAT MEAN A PR ACTUALLY WON SOMETHING.
#
# merge_blockers used to accept any label starting with "eval:" as proof the PR had passed the
# gate. `eval:none` starts with "eval:". So the one label whose entire meaning is "no
# significant gain" satisfied the check that exists to require a gain.
#
# #81 merged through that hole and cost main 31.5%. It had been measured honestly at +4.17%
# against a 20.14 main in an earlier round; #74 then landed underneath it, the two changes
# conflicted, and the re-measurement on current main came back 14.54 against a 21.24 frontier.
# Every part of the loop that was supposed to catch this DID: needs-rebase forced the
# re-measurement, and eval-label.yml correctly applied eval:none. The merge decision then
# ignored both.
SCORING_TIERS = {"eval:xs", "eval:s", "eval:m", "eval:l", "eval:xl"}
NO_GAIN_TIER = "eval:none"

# The PR template's hardware attestation. Matched loosely on purpose -- the template uses a
# multiplication sign (8x H200) that is easy to retype as an ASCII x, and failing a real
# submission over a homoglyph is worse than accepting a near-miss. What must be exact is
# the ticked box, because that is the author's attestation that they ran it.
TICKED = re.compile(r"-\s*\[\s*[xX]\s*\]")
H200 = re.compile(r"h200", re.I)
NODE_RUN_HEADING = re.compile(r"^##\s+node run\s*$", re.I)
ANY_HEADING = re.compile(r"^##\s+")


def node_run_section(body):
    """Only the text between '## Node run' and the next '##' heading.

    Same bug, same fix as node-attestation.yml (#78): a body-wide TICKED+H200 scan matched
    the PR-template Checklist boilerplate '- [x] `...--node h200x8 --dry-run` resolves' --
    ticked on nearly every PR and containing 'h200' -- so a PR read as attested while the
    real Node run box sat unticked. Here the consequence was bounded (the bot still
    measures for real) but live: #71 was evaluated in two rounds and #74 queued, neither
    ever having attested a node run. The attestation is a claim made in ONE place; read
    only that place. No section => not attested, which fails closed.
    """
    lines = (body or "").split("\n")
    start = next((i for i, ln in enumerate(lines) if NODE_RUN_HEADING.match(ln)), -1)
    if start == -1:
        return ""
    end = next((i for i in range(start + 1, len(lines)) if ANY_HEADING.match(lines[i])),
               len(lines))
    return "\n".join(lines[start + 1:end])


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
            "number,title,isDraft,headRefOid,body,author,labels,isCrossRepository,mergeable"])
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

    # A conflicted branch cannot be brought current, so it cannot be measured against a
    # frontier that is about to move under it, and it cannot be merged at the end. #64 sat in
    # a round as CONFLICTING: update_pr_branch could not advance it -- the round log shows
    # #77, #74 and #71 updated and #64 simply absent -- so anything it measured would have
    # described a tree that does not exist on main. That is ~40 GPU-minutes spent on a result
    # nobody can act on.
    #
    # Only an explicit CONFLICTING skips. GitHub computes mergeability lazily and answers
    # UNKNOWN while recomputing, which is the normal state for every open PR in the seconds
    # after main moves -- treating that as a conflict would skip the entire field. UNKNOWN
    # falls through to the merge path, where wait_mergeable_state() already blocks for a real
    # answer.
    if pr.get("mergeable") == "CONFLICTING":
        return False, "conflicts with main — rebase before it can be evaluated or merged"

    # CI's own verdict on the attestation, re-derived by node-attestation.yml on every edit
    # and synchronize, so it does not go stale. It is already in NEVER_MERGE_LABELS: a PR
    # carrying it cannot merge at the end of the round however fast it measures, so booking
    # the node for it is spending GPU on a foregone conclusion.
    #
    # It is also a cross-check on the scan below. Both read the same checkbox out of the same
    # body, so a disagreement means one of them is broken -- which is exactly the state this
    # file was in before the fix below: the label said #74 and #71 had no node run, the bot
    # said they did, and the bot was wrong. Two independent readings that must agree beat one
    # reading trusted absolutely.
    if "needs-node-run" in labels:
        return False, ("needs-node-run — node-attestation.yml found no ticked node box "
                       "in '## Node run'")

    section = node_run_section(pr.get("body") or "")
    ticked = any(TICKED.search(ln) and H200.search(ln) for ln in section.splitlines())
    if not ticked:
        return False, "the 8x H200 box is not ticked — the author has not attested a node run"
    return True, "eligible"


# A GUARD THAT CAN BE RETRIED IS NOT A GUARD.
#
# These are the harness saying "I will not score this", which is an ANSWER, not a glitch.
# Running the same measurement again until it passes is not a retry, it is sampling until the
# result is convenient -- and every one of these guards exists because a number could
# otherwise be fabricated. If one of them fires, the round takes the refusal.
VERDICT_MARKERS = (
    "refusing to score",
    "wall clock allows",
    "did not do the extra work",
    "harness measured commit",
    "non-positive time delta",
    "reported no ms/token line",
)
# These are the BOX misbehaving: the model never loaded, the collective never initialised, the
# connection died. They say nothing about the code under test, so the round should try again
# rather than throw away every PR in it.
TRANSIENT_RE = re.compile(
    r"ncclCommInitAll|unhandled cuda error|\bNCCL\b|cudaMalloc failed|init failed at tp=|"
    r"weight load failed|out of memory|\bCUDA error\b|bench failed|"
    r"ssh timeout|Connection (?:timed out|closed|refused|reset)|Broken pipe|"
    r"kex_exchange_identification|banner exchange",
    re.I)


def is_transient(exc):
    """True if `exc` is the box failing, not the harness reaching a verdict.

    Order matters: a verdict marker anywhere in the message wins, even if a transient-looking
    word also appears in the 1200-char tail the error carries. Refusing to retry is the safe
    direction -- it costs a round, where retrying a verdict costs the guard.
    """
    msg = str(exc)
    if any(m in msg for m in VERDICT_MARKERS):
        return False
    return bool(TRANSIENT_RE.search(msg))


def with_box_retry(what, fn, tries=3, delay=20):
    """Run fn(), retrying only when the BOX failed rather than the code under test.

    A frontier failure used to discard the entire round: measure_frontier raises, main() bails,
    and not one PR is evaluated however healthy it was. Two rounds in the ledger died exactly
    there -- rounds/66955446f62f and rounds/11eb5e4d02b7, the latter on

        [tp] FATAL: ncclCommInitAll(n=8): unhandled cuda error
        init failed at tp=8, 93 layers

    -- and every eligible PR in both got nothing. A transient collective-init error is not a
    statement about anyone's code, and it should cost a couple of minutes, not a round.

    Each attempt re-enters _box_build, which kills orphaned compute processes before it builds,
    so a retry is a genuinely clean attempt rather than the same poisoned box twice.

    Verdicts are never retried -- see is_transient.
    """
    for i in range(1, tries + 1):
        try:
            return fn()
        except RuntimeError as exc:
            if i >= tries or not is_transient(exc):
                raise
            print(f"!! {what}: attempt {i}/{tries} failed on the box — {str(exc).splitlines()[0][:160]}",
                  file=sys.stderr)
            print(f"   retrying in {delay}s; the next attempt rebuilds and clears stray "
                  "compute processes first", file=sys.stderr)
            time.sleep(delay)


def sync_conflict_labels(repo, prs, dry_run):
    """Label the PRs the round skipped for conflicting with main, and clear the rest.

    The skip was already correct but invisible: it printed a line in a log only the operator
    reads, so a contributor saw their PR sit out round after round with nothing on the PR
    saying why. #55, #87 and #90 all sat conflicted through a full round like that.

    SELF-CLEARING, like needs-rebase. Mergeability is re-resolved every round before this
    runs, so a PR that has been rebased loses the label on the next round without anyone
    asking. A label only a maintainer can remove turns a mechanical state into a queue
    somebody has to babysit.

    Deliberately NOT in NEVER_MERGE_LABELS. GitHub already refuses to merge a conflicted
    branch, so the label would add nothing there -- and if it ever went stale it would block
    a PR that is now perfectly mergeable. The state is the authority; the label only reports it.
    """
    for pr in prs:
        num = pr["number"]
        has = CONFLICT_LABEL in {l.get("name", "") for l in pr.get("labels") or []}
        conflicted = pr.get("mergeable") == "CONFLICTING"
        if conflicted == has:
            continue
        if dry_run:
            print(f"--- dry-run: would {'add' if conflicted else 'remove'} "
                  f"{CONFLICT_LABEL} on #{num}")
            continue
        if conflicted:
            gh(["label", "create", CONFLICT_LABEL, "-R", repo, "--color", "B60205",
                "--description", "conflicts with main — rebase and the round picks it up"])
            # NOT `gh pr edit --add-label`: it queries projectCards, which GitHub has
            # deprecated, so it exits 1 even where the repo has no projects.
            q = gh(["api", "-X", "POST", f"repos/{repo}/issues/{num}/labels",
                    "-f", f"labels[]={CONFLICT_LABEL}"])
            if q.returncode != 0:
                print(f"!! #{num}: could not label {CONFLICT_LABEL}: {q.stderr.strip()[:120]}",
                      file=sys.stderr)
                continue
            print(f">> #{num}: {CONFLICT_LABEL} (conflicts with main — not evaluated)")
        else:
            gh(["api", "-X", "DELETE", f"repos/{repo}/issues/{num}/labels/{CONFLICT_LABEL}"])
            print(f">> #{num}: {CONFLICT_LABEL} cleared — no longer conflicting")


def resolve_mergeability(repo, prs, tries=8, delay=5):
    """Settle pr['mergeable'] for the PRs where it decides whether to book the node.

    THE BOT ASKS THIS QUESTION AT THE ONE MOMENT THE ANSWER IS GUARANTEED TO BE MISSING.

    GitHub computes mergeability lazily and answers UNKNOWN while recomputing, which is the
    normal state for every open PR in the seconds after the base branch moves -- and this bot
    moves the base branch itself, committing the frontier to reference.lock at the start of
    every round. So the CONFLICTING skip was least reliable exactly when it was needed.
    Verified live: seconds after a merge, #64 read ELIGIBLE while being CONFLICTING; a minute
    later the same call read CONFLICTING and skipped it correctly.

    Polled only for PRs that are otherwise eligible, so it costs a handful of API calls rather
    than the ~40 GPU-minutes the answer is protecting. A PR already skipped for an unticked
    box or a needs-node-run label is not asked about -- mergeability cannot change that.

    FAILS OPEN on timeout, loudly. A wrong skip during a wide UNKNOWN (a GitHub incident)
    would stop every round and the payouts with it; a wrong include costs one wasted eval that
    merge_blockers and wait_mergeable_state still refuse at the end. The asymmetry decides it.
    """
    for pr in prs:
        if pr.get("mergeable") in ("MERGEABLE", "CONFLICTING"):
            continue
        if not eligibility(pr)[0]:
            continue
        num = pr["number"]
        for i in range(tries):
            info = _pr_state(repo, num)
            # No data at all is "cannot read this PR", not "still recomputing" -- polling it
            # for a minute helps nobody. One retry covers a transient blip, as in
            # wait_mergeable_state.
            if not info:
                if i >= 1:
                    print(f"!! #{num}: cannot read merge state from {repo}", file=sys.stderr)
                    break
                time.sleep(min(delay, 2))
                continue
            state = info.get("mergeable") or "UNKNOWN"
            if state != "UNKNOWN":
                pr["mergeable"] = state
                if i:
                    print(f"   #{num}: mergeability settled to {state} after ~{i * delay}s")
                break
            time.sleep(delay)
        else:
            print(f"!! #{num}: mergeability still UNKNOWN after {tries * delay}s — evaluating "
                  "it anyway; a conflicted branch is still refused at the merge",
                  file=sys.stderr)


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
        # bench/scripts ONLY -- bench/refdata is deliberately NOT restored any more.
        #
        # The answer key used to be checked out into the very tree this binary runs in, at a
        # fixed relative path, with cwd at the repo root. A bench could open
        # bench/refdata/hello.spkl and write it back out as its own --logits dump: top1 1.0,
        # KL 0.0, for a binary that computed nothing. Accuracy is measured by the controller
        # now (measure_accuracy), against a reference this box never receives.
        # bench/refdata comes back, but WITHOUT its answer key.
        #
        # The .ids are INPUTS: which tokens the executor is fed. They are public in the
        # repo, so withholding them protects nothing, and the 32k-token parity prompt is
        # ~200 KB -- far past MAX_ARG_STRLEN, so shipping it per round over ssh is not
        # possible anyway. They come from origin/main precisely so a PR cannot swap in a
        # shorter or easier prompt and be graded on it.
        #
        # The .spkl are the ANSWERS, and they still never reach this box. That is the
        # attack this defends: a bench could open bench/refdata/*.spkl and write it back
        # out as its own --logits dump -- top1 1.0, KL 0.0, for a binary that computed
        # nothing. Accuracy is graded by the controller (measure_accuracy) against the
        # controller's own copy.
        # NOTE THE PATHSPEC: '*.ids' ONLY. The answer key is never written to this disk at
        # all, not even for the instant before a delete.
        "git checkout -q origin/main -- bench/scripts 'bench/refdata/*.ids'",
        # …and any .spkl the PR's OWN checkout carried is removed, since bench/refdata is
        # committed and a branch can contain whatever it likes.
        "rm -f bench/refdata/*.spkl",
        "export PATH=/usr/local/cuda/bin:$PATH",
        # EVERY BUILD IS FROM SCRATCH.
        #
        # build/ is gitignored, so `git clean -qfd` leaves it (that needs -x) and every PR
        # was compiled on top of the previous PR's objects, round after round, across a dozen
        # checkouts. On 2026-08-03 that path produced numbers no fresh build can reproduce:
        # #81 measured 14.54 in a round and 22.17 on six clean builds, and main measured
        # 14.14 through the same build directory against 21.25 clean. The regression that
        # reading caused was real -- #81 was reverted in #84 on the strength of it.
        #
        # A single incremental step is not obviously at fault: main->#81 in isolation
        # reproduces 22.18. What the bot does is not a single step, it is a dozen, and the
        # difference is not worth reasoning about when the cure is this cheap. A clean build
        # is ~20 s here, about 2 minutes across a full round, against a ~40 minute round --
        # far less than one wrong measurement costs.
        "rm -rf build",
        # PIN THE COMPILER. /usr/local/cuda is a symlink the box owner can move, and it DID
        # move today: the note in memory recorded 12.8, the link now resolves to 13.0. Both
        # measure the same here (21.24 vs 21.25), so this did not cause the above -- but a
        # loop that pays on measurements should not let a symlink choose its compiler, and
        # the version belongs in the log so the next discrepancy is diagnosable rather than
        # archaeological.
        'NVCC="$(command -v nvcc)"',
        '[ -x "$NVCC" ] || { echo "no nvcc on PATH" >&2; exit 1; }',
        'echo ">> toolchain: $($NVCC --version | tail -1)"',
        # -DSPARKINFER_TP=ON is not optional here. Without it the runtime has no NCCL
        # collective, and kimi_k3_tp_bench refuses to run sharded rather than silently
        # producing a wrong number -- correct behaviour, but it means a build configured
        # without this flag fails at eval time with no hint that configure was the cause.
        #
        # Configure output is NO LONGER discarded. It used to go to /dev/null, so a failed
        # configure was invisible and `cmake --build` walked on against whatever cache
        # happened to exist -- which is how a stale build directory can decide a payout
        # without anything in the log admitting it.
        'cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 '
        '-DSPARKINFER_TP=ON -DCMAKE_CUDA_COMPILER="$NVCC" '
        '|| { echo "cmake configure FAILED" >&2; exit 1; }',
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


REFDATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "bench", "refdata")

# THE DEPTHS THE PARITY SUITE PROBES, BEYOND THE HISTORICAL 4-TOKEN ONE.
#
# WHY MORE THAN ONE PROBE. The gate used to be a single next-token distribution after a
# 4-token prompt -- one row of one .spkl, n=1. Everything the engine is actually scored on
# happens at 131,072 tokens, and everything that only appears once the KV cache is
# populated (the F16 latent cache, per-rank CUDA graphs, the banded LM head, 2-D MoE
# sharding) was gated by a measurement that barely touches it. A change can be bit-exact
# at 4 tokens and wrong at 100k and the old gate would pass it clean.
#
# These are nested prefixes of ONE document (bench/refdata/longctx.txt), so they test
# genuine long-range dependency rather than ten unrelated short prompts, and one pass over
# the longest prefix produces all of them.
#
# WHY IT STOPS AT 8192, AND WHAT IT COSTS.
#
# The deep pass feeds the prefix through the DECODE path one token at a time -- the
# runtime has no batched prefill -- so the cost is max(depth)/decode_tok_s per measured
# build, paid once for main plus once per PR in a round. Measured on the 8x H200 node,
# main @ be3c818, decode time for one continuous pass:
#
#     to  4096   136 s   (2.3 min)   <-- the default
#     to  8192   236 s   (3.9 min)
#     to 16384   429 s   (7.2 min)
#     to 32768   812 s  (13.5 min)
#
# END TO END this function costs MORE than the decode figure, because it pays TWO model
# loads: the 4-token probe and the deep pass are separate invocations (tp_bench takes one
# --ids set, and they are different prompts). Measured at the 8192 default: 452 s per
# measured build, of which ~219 s is model loading. A round pays that once for main plus
# once per PR.
#
# Decode rate is flat at ~42 tok/s across those segments: per-token cost is dominated by
# streaming the active weights, and MLA keeps the context-dependent term small. So the
# cost is very close to linear in max(depth), and 32768 triples a round's parity budget
# to buy two more depths.
#
# References for 8192, 16384 and 32768 ARE captured and committed. Opt in with
# K3_PARITY_DEPTHS=...,8192,16384,32768 when a change plausibly breaks only very deep.
#
# 4096 is the default because ingestion is the round's dominant cost and buys no extra
# coverage per second: a 6-build round pays 35 min at 4096 against 45 min at 8192. That
# ceiling is a deliberate trade against the 131,072 the engine is SCORED at -- see #119.
# It moves back up for free the moment batched prefill exists.
#
# WHY NOT 131072. Two reasons, and only one of them is the reference. The capture side
# grows with depth, but OUR side is the binding one: ingestion runs at decode speed
# (measured 42.80 tok/s, 1.03x our scored decode rate), so a 131,072-token probe costs
# ~52 minutes PER BUILD. This narrows the untested gap from "everything past 4 tokens" to
# "everything past 4k" (32k when opted in) -- it does not close it, and README/CONTRIBUTING
# say so rather than implying full-context parity is proven.
PARITY_DEPTHS = [int(x) for x in os.environ.get(
    "K3_PARITY_DEPTHS",
    "128,256,512,1024,2048,4096").split(",") if x.strip()]
PARITY_CORPUS = os.environ.get("K3_PARITY_CORPUS", "longctx")
# The deep pass is minutes of decode, not seconds, and it runs behind the same ssh call
# as the 4-token probe. 3600 was sized for the latter alone.
PARITY_TIMEOUT = int(os.environ.get("K3_PARITY_TIMEOUT", "10800"))
# Wait for a busy node before loading 553 GiB onto it. A GPU under this many MiB counts as
# free -- a few hundred MiB of driver/context is normal and never clears.
PARITY_SETTLE_MIB = int(os.environ.get("K3_PARITY_SETTLE_MIB", "2048"))
PARITY_SETTLE_TRIES = int(os.environ.get("K3_PARITY_SETTLE_TRIES", "60"))
PARITY_SETTLE_SLEEP = int(os.environ.get("K3_PARITY_SETTLE_SLEEP", "20"))


def measure_accuracy(what):
    """Grade the box's logits HERE, against a reference the box never sees.

    The correctness gate is the only thing standing between a fast-but-wrong kernel and a
    payout, and it used to be decided on the box: bench/refdata was checked out into the
    working tree the PR's binary ran in, so the binary could read the answer key and echo it
    back as its own output. Perfect top1, zero KL, no work done -- and nothing rejected a
    suspiciously exact result, because label.py only bounds top1 >= 0.90 and KL <= 0.20.

    So: the ids go out, the logits come back, and compare_logits.py runs on this machine
    against this machine's copy of the reference. The box cannot match a file it does not
    have.

    Returns (top1, kl, depths):

        top1    the WORST top-1 agreement across the suite
        kl      the WORST (highest) mean KLD across the suite
        depths  {probe_name: {"top1": float, "kl": float}} for every probe

    The two scalars are worst-case on purpose. A suite is only as strong as the depth it
    fails at, and averaging would let a 32k regression hide behind nine good shallow
    probes -- which is the exact failure mode this replaced.

    Raises RuntimeError if any probe fails to come back: a missing depth would silently
    become "no comparison at that depth", which is what this exists to remove.
    """
    probes = _parity_probes()

    outdir = "/tmp/k3par_$$"
    ndev = len(DEVICES.split(","))
    gguf = f"$(ls {BOX_MODELS_DIR}/*/*-00001-of-*.gguf | head -1)"
    binary = f"{BOX_REPO_DIR}/build/runtime/kimi_k3_tp_bench"
    steps = [f"cd {BOX_REPO_DIR}", f"mkdir -p {outdir}"]

    # WAIT FOR THE NODE. This is a RENTED box and it is not always ours: on 2026-08-04 a
    # round walked into another tenant's benchmark sweep, both jobs tried to allocate ~70
    # GiB/GPU, and both died on cudaMalloc. The harness gained settle_gpus() for its own
    # run, but that is in _box_eval -- by which point THIS function has already tried to
    # load 553 GiB. Wait here, where the first allocation actually happens.
    #
    # Advisory, not fatal: if the box never clears we still try, because refusing to score
    # because a neighbour is busy is its own failure mode. The point is to not collide with
    # something that is about to finish.
    steps.append(
        f'for i in $(seq 1 {PARITY_SETTLE_TRIES}); do '
        f'busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null '
        f"| awk -v lim={PARITY_SETTLE_MIB} '$1 > lim' | wc -l || true); "
        f'[ "${{busy:-0}}" = "0" ] && break; '
        f'echo "waiting for $busy busy GPU(s) …" >&2; sleep {PARITY_SETTLE_SLEEP}; done')

    # The 4-token probe: fed inline exactly as it always was, so its number stays
    # comparable with every round this gate has ever reported.
    #
    # STDERR GOES TO A FILE, NOT /dev/null. It used to be discarded, and when the probe
    # died of cudaMalloc the only thing the round could say was "probe ctx4 produced no
    # logits" -- which reads as a bug in the parity suite and sent the investigation to
    # entirely the wrong place. The log is small and rides back in the same tar.
    steps.append(f"env -u POLARIS_API_KEY {binary} {gguf} {ndev} 93 1 "
                 f"--ids {_hello_ids_csv()} --logits {outdir}/hello.spkl "
                 f"> {outdir}/hello.log 2>&1")

    # The deep suite: ONE pass over the longest prefix, dumping at each checkpoint. The
    # ids are read from the box's own origin/main checkout -- 32768 ids is ~200 KB, far
    # past MAX_ARG_STRLEN, so they cannot be passed as an argument.
    if PARITY_DEPTHS:
        deep_ids = f"{BOX_REPO_DIR}/bench/refdata/{PARITY_CORPUS}.ctx{max(PARITY_DEPTHS)}.ids"
        # Say so on stderr rather than letting the bench fail obscurely on a missing file:
        # the ids come from origin/main's checkout, so an absent one means the restore
        # pathspec in _box_build and the depths configured here have drifted apart.
        steps.append(f'test -s {deep_ids} || echo "MISSING PARITY IDS: {deep_ids}" >&2')
        steps.append(f"env -u POLARIS_API_KEY {binary} {gguf} {ndev} 93 1 "
                     f"--ids @{deep_ids} "
                     f"--checkpoints {','.join(str(d) for d in PARITY_DEPTHS)} "
                     f"--logits-prefix {outdir}/{PARITY_CORPUS} "
                     f"--ctx {max(PARITY_DEPTHS) + 16} > {outdir}/deep.log 2>&1")

    # tar+base64 back over the same ssh channel: no scp, no second auth, and the transfer
    # is part of the command whose exit status we already check.
    cmd = ("; ".join(steps) +
           f"; tar cf - -C {outdir} . | base64 -w0; rm -rf {outdir}")
    r = sh(cmd, timeout=PARITY_TIMEOUT)
    blob = (r.stdout or "").strip()
    if not blob:
        raise RuntimeError(f"no logits dump came back for {what} — cannot grade correctness"
                           + (f": {(r.stderr or '').strip()[:200]}" if r.stderr else ""))

    workdir = tempfile.mkdtemp(prefix="k3par_")
    try:
        tar_path = os.path.join(workdir, "probes.tar")
        with open(tar_path, "wb") as f:
            f.write(base64.b64decode(blob))
        with tarfile.open(tar_path) as tf:
            _extract_probes(tf, workdir)

        depths = {}
        for name, ref_spkl, remote_name in probes:
            ours = os.path.join(workdir, remote_name)
            if not os.path.isfile(ours) or os.path.getsize(ours) == 0:
                # Quote the PROBE'S OWN log, not just ssh's stderr. The probe writes to
                # <outdir>/{hello,deep}.log and it rides back in the same tar, so the
                # actual reason -- most often `cudaMalloc failed`, i.e. the node was busy
                # -- is right here instead of being inferred from an absent file.
                why = []
                for log in ("hello.log", "deep.log"):
                    p = os.path.join(workdir, log)
                    if os.path.isfile(p):
                        tail = open(p, errors="replace").read().strip()
                        tail = "\n".join(l for l in tail.splitlines()
                                         if "setlocale" not in l)[-400:]
                        if tail:
                            why.append(f"{log}: …{tail}")
                # DO NOT SAY "refusing to score" HERE.
                #
                # That exact phrase is VERDICT_MARKERS[0], and is_transient() treats a
                # verdict marker ANYWHERE in the message as decisive -- so wording it that
                # way told with_box_retry that a busy node was a permanent scoring verdict
                # and killed the round on the first attempt. An incomplete suite is the
                # box failing, not the harness reaching a conclusion about this PR: the
                # right response is to retry, and TRANSIENT_RE already matches the
                # `cudaMalloc failed` / `init failed at tp=` the probe log carries.
                #
                # The REFUSAL is unchanged -- 7 of 8 probes is still not parity and is
                # still never scored. Only the classification of WHY it happened changes.
                raise RuntimeError(
                    f"{what}: incomplete parity suite — probe {name} produced no logits. "
                    f"Came back: "
                    f"{sorted(f for f in os.listdir(workdir) if f.endswith('.spkl'))}"
                    + ("\n" + "\n".join(why) if why else "")
                    + (f"\nssh stderr: {(r.stderr or '').strip()[:200]}" if r.stderr else ""))
            depths[name] = _compare_logits_here(ref_spkl, ours, what, name)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    top1 = min(d["top1"] for d in depths.values())
    kl = max(d["kl"] for d in depths.values())
    width = max(len(n) for n in depths)
    for name, d in depths.items():
        print(f"  {name:>{width}}: top1={d['top1']:.6f} kl={d['kl']:.9f}")
    print(f"{what}: accuracy measured HERE over {len(depths)} depths — "
          f"worst top1={top1} worst kl={kl}")
    return top1, kl, depths


def _extract_probes(tf, dest):
    """Unpack the probe tar, refusing anything that is not a plain file under dest.

    THIS TAR IS BUILT ON THE BOX, and the box is the machine running unmodified PR code.
    A bench that dropped a symlink or a "../.." entry into its output directory could
    otherwise turn this extract into an arbitrary write on the CONTROLLER -- the one
    machine in this system that holds the answer key and the merge credentials. Nothing
    legitimate here is anything but a regular .spkl file.
    """
    root = os.path.realpath(dest)
    for m in tf.getmembers():
        if m.name in (".", "./"):
            continue
        if not m.isfile():
            raise RuntimeError(f"probe archive contains a non-regular entry {m.name!r} "
                               f"({'symlink' if m.issym() or m.islnk() else 'special'}) — "
                               f"refusing to extract anything the box may have planted")
        target = os.path.realpath(os.path.join(root, m.name))
        if target != root and not target.startswith(root + os.sep):
            raise RuntimeError(f"probe archive entry {m.name!r} escapes the extract "
                               f"directory — refusing to extract")
    # filter="data" is the belt to the above braces on 3.12+; older runtimes get the
    # validation loop alone, which already covers the same cases.
    try:
        tf.extractall(dest, filter="data")
    except TypeError:
        tf.extractall(dest)


def _hello_ids_csv():
    """The historical 4-token probe's ids, from the controller's own copy."""
    ids_file = os.path.join(REFDATA, "hello.ids")
    if not os.path.isfile(ids_file):
        raise RuntimeError(f"missing controller-side reference {ids_file} — refusing to "
                           f"score without a correctness gate")
    with open(ids_file) as f:
        # hello.ids holds prompt ids THEN the reference continuations; the prompt is the
        # first 4. Taking the whole file would score a different step.
        return ",".join(l.strip() for l in list(f)[:4] if l.strip())


def _parity_probes():
    """[(name, controller-side ref .spkl, filename the box will produce)] for the suite.

    Every reference must exist HERE before a single GPU-second is spent: discovering a
    missing answer key after the run would either waste the round or, worse, tempt a
    partial score.
    """
    probes = [("ctx4", os.path.join(REFDATA, "hello.spkl"), "hello.spkl")]
    for depth in PARITY_DEPTHS:
        probes.append((f"ctx{depth}",
                       os.path.join(REFDATA, f"{PARITY_CORPUS}.ctx{depth}.spkl"),
                       f"{PARITY_CORPUS}.ctx{depth}.spkl"))
    missing = [p for _, p, _ in probes if not os.path.isfile(p)]
    if missing:
        raise RuntimeError(
            "missing controller-side parity reference(s) — refusing to score without the "
            "full correctness gate: " + ", ".join(os.path.basename(m) for m in missing) +
            ". Capture them with bench/scripts/capture_parity_refs.sh")
    return probes


def _compare_logits_here(ref_spkl, ours, what, tag):
    """Run compare_logits.py on THIS machine and return {"top1":…, "kl":…}."""
    cmp_py = os.path.join(os.path.dirname(REFDATA), "scripts", "compare_logits.py")
    p = subprocess.run([sys.executable, cmp_py, ref_spkl, ours, "--json"],
                       capture_output=True, text=True, timeout=600)
    # ITS EXIT CODE IS A VERDICT, NOT AN ERROR, AND IT IS ALWAYS 1 FOR K3.
    #
    # compare_logits returns 0 only for mean_kld < 1e-5 -- a SAME-IMPLEMENTATION bar,
    # "two implementations of one arithmetic". K3's accepted parity is 4.05e-03, about
    # 400x that, from a known cause: K3 keeps f32 activations where ggml quantizes them
    # before a quantized mat-vec. CONTRIBUTING documents it. So the tool says FAIL on a
    # perfectly good run, every time.
    #
    # The shell this replaced ran it with `|| true` and read the JSON; porting it to
    # Python without that turned every round into "compare_logits failed" with an empty
    # stderr, which is how the first hardened round died. The gate that matters is
    # label.py's (top1 >= 0.90, kl <= 0.20), applied downstream.
    #
    # So parse the output and let the numbers speak. Only a MISSING or unparseable
    # payload is a real failure -- that means the tool did not run, which is different
    # from it disagreeing.
    out = (p.stdout or "").strip()
    if not out:
        raise RuntimeError(f"compare_logits produced no output for {what} {tag} (rc="
                           f"{p.returncode}): {(p.stderr or '').strip()[:300]}")
    try:
        d = json.loads(out)
        return {"top1": float(d["top1_agreement"]), "kl": float(d["mean_kld"])}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"compare_logits output for {what} {tag} is not the expected "
                           f"JSON ({exc}): {out[:200]}") from exc


def _box_eval(what, frontier=None, seal=False, top1=None, kl=None):
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
    # Harness knobs set on the CONTROLLER, forwarded to the box. bench/scripts is restored
    # from origin/main before every build, so the harness the box runs is whatever is on the
    # protected branch -- there is otherwise no way to run a round against a tuned constant
    # without merging it first. That is the right default for anything a PR could influence,
    # and the wrong one for an operator fixing a harness bug the round is currently hitting.
    #
    # PRINTED, not silent. These change how a number was produced, so a round log that does
    # not name them describes a measurement nobody can reproduce. The value goes in the log
    # that ships to sparkinfer-k3-log, next to the result it produced.
    passthru = {k: v for k, v in sorted(os.environ.items())
                if k.startswith("KIMI_K3_") and k not in ("KIMI_K3_MODELS_DIR",)}
    if passthru:
        print("   harness overrides from the controller: "
              + " ".join(f"{k}={v}" for k, v in passthru.items()))
    env_prefix = "".join(f"{k}={shlex.quote(v)} " for k, v in passthru.items())
    # Accuracy graded on THIS machine, handed to the harness. Without these the harness
    # falls back to comparing on the box against a reference the binary can read.
    acc_flag = "" if top1 is None or kl is None else f" --top1 {top1} --kl {kl}"
    # POLARIS_API_KEY lives in a 0600 file on the box, not in this command line and not in
    # the repo: anything passed as an ssh argument shows up in ps on a shared machine.
    evalcmd = (
        f"cd {BOX_REPO_DIR} && "
        f"[ -r {POLARIS_ENV} ] && . {POLARIS_ENV}; "
        # CMake emits executables under build/runtime/, not build/. The harness resolves
        # $SPARKINFER_BUILD/kimi_k3_tp_bench, so pointing this at build/ makes it exit 2 with
        # "not built" immediately after a build that in fact succeeded.
        f"KIMI_K3_MODELS_DIR={BOX_MODELS_DIR} SPARKINFER_BUILD={BOX_REPO_DIR}/build/runtime "
        f"{env_prefix}"
        f"bash bench/scripts/kimi_k3_eval.sh --node {NODE} --devices {DEVICES}"
        f"{front_flag}{acc_flag}{seal_flag}"
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
    top1, kl, depths = measure_accuracy("main")
    res, _, _ = _box_eval("main", frontier=0, seal=False, top1=top1, kl=kl)
    tps = float(res.get("tps") or 0)
    if tps <= 0:
        raise RuntimeError(f"main measured {tps} tok/s — refusing to score a round against it")
    got = str(res.get("commit", "")).lower()
    if not sha.lower().startswith(got) or len(got) < 7:
        raise RuntimeError(f"harness measured commit {got!r} but main is {sha[:12]!r}")
    print(f"main: frontier = {tps} tok/s @ {sha[:8]}")
    # main's PER-DEPTH KL is the round's PARITY BASELINE, measured on the same box minutes
    # before every PR in the round. The absolute bars cannot see drift; only this can.
    # Returned per depth, not collapsed: a PR that is fine at 4 tokens and 2x worse at 32k
    # is exactly the case a single number cannot express. See kl_ratchet().
    return sha, tps, str(res.get("quant") or "UD-IQ1_S"), depths


# THE ACCURACY GATE WAS A FLOOR, NOT A RATCHET.
#
# label.py asks "is KL under the bar?" and never "is it worse than main?". At K3's bars that
# left a lot of room: main measures 0.0038674, KL_PREFER is 0.02 and KL_BAR is 0.05, so a PR
# can degrade parity 5x and still pass CLEAN -- not even annotated. #74 did exactly that,
# measuring 0.0059102 (1.53x main) and merging with nothing said, which moved main's baseline
# permanently. The next 1.53x then lands at 0.0089 and also passes clean. Roughly five such
# merges fit under the warn line, each individually "fine", and parity drifts with no round
# ever reporting a problem.
#
# A ratchet compares against what main measured THIS ROUND, on THIS BOX, minutes earlier --
# the only comparison that can see drift at all.
#
# NOT set to 1.0. Changing float reduction order changes the numbers legitimately: #74 swaps
# the TP collective to peer-oneshot, and #77/#81 were bit-identical only because they did not
# touch reduction order. A ratchet at parity would reject honest work. These factors are wide
# enough for that and far tighter than the 13x the absolute bar allows.
KL_RATCHET_WARN = float(os.environ.get("K3_KL_RATCHET_WARN", "1.25"))
KL_RATCHET_REJECT = float(os.environ.get("K3_KL_RATCHET_REJECT", "2.0"))
KL_REGRESSION_LABEL = "accuracy-regression"
# MATERIALITY FLOOR: a ratio is only a regression if the number it moved is big enough to
# matter. The depth sweep grades parity from ctx4 down to ctx4096, where main sits at
# 1.5e-06 -- five orders of magnitude below label.py's KL_BAR of 0.05. At those magnitudes a
# reduction-order change moves the ratio by multiples while moving the ABSOLUTE divergence by
# nothing anyone can act on, and the ratchet exists to stop parity drifting toward the bar,
# not to police arithmetic that is already 500x under it.
#
# #104 is the case: its only >=2x depth was ctx512 at 0.000097 -- 0.19% of the bar -- while it
# was BETTER than main at ctx4, the depth that carries real weight. It was blocked anyway.
# #115 is the contrast and must stay blocked: ctx2048 at 0.0019 is 3.9% of the bar and its
# ctx4 parity degraded 60% (0.0067 -> 0.0107, 21% of the bar).
#
# 5e-4 is 1% of KL_BAR and sits mid-plateau: every floor from 1e-4 to 1e-3 clears #104 and
# keeps #115, so this is not tuned to two cases. Depths below it are still REPORTED, marked
# with *, and a depth that grows material stops being exempt on its own.
KL_RATCHET_FLOOR = float(os.environ.get("K3_KL_RATCHET_FLOOR", "5e-4"))


def _depth_sort_key(name):
    """ctx4 < ctx128 < … < ctx32768, so notes read in depth order rather than lexically."""
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else 0


def _as_depth_kls(v):
    """Normalise a parity result to {depth_name: kl_float}.

    Accepts the suite's {name: {"top1":…, "kl":…}}, a plain {name: float}, or a bare float
    (treated as the single historical 4-token probe) so a caller holding only the old
    scalar still gets a meaningful comparison rather than a crash.
    """
    if isinstance(v, dict):
        out = {}
        for k, d in v.items():
            try:
                out[str(k)] = float(d["kl"] if isinstance(d, dict) else d)
            except (TypeError, ValueError, KeyError):
                continue
        return out
    try:
        return {"ctx4": float(v)}
    except (TypeError, ValueError):
        return {}


def kl_ratchet(pr_kl, main_kl):
    """Return (verdict, worst_ratio, note) comparing a PR's parity against main's.

    Compares DEPTH BY DEPTH and reports the worst. A PR that is bit-identical at 4 tokens
    and 2x worse at 32k is precisely the regression a single scalar cannot express, and
    precisely the one that matters: the engine is scored at 131,072, not at 4.

    verdict is "ok", "warn" or "reject". A missing or zero baseline yields "ok" with a note
    -- an unmeasurable baseline must not become a silent reject, and the absolute bars
    still apply underneath. This only ever ADDS a constraint; it never lets through
    anything the floor would have caught.
    """
    pr = _as_depth_kls(pr_kl)
    mn = _as_depth_kls(main_kl)
    shared = [k for k in pr if k in mn and mn[k] > 0 and pr[k] >= 0]
    if not shared:
        return "ok", 0.0, "parity vs main: unavailable (no baseline measured this round)"

    ratios = {k: pr[k] / mn[k] for k in shared}
    order = sorted(shared, key=_depth_sort_key)
    # Only depths whose absolute divergence is material can produce a verdict. The rest are
    # reported with a trailing * so the full picture is still visible in the round log.
    graded = [k for k in shared if pr[k] >= KL_RATCHET_FLOOR]
    detail = " ".join(f"{k}={ratios[k]:.2f}x" + ("" if k in graded else "*") for k in order)
    if not graded:
        top = max(shared, key=lambda k: ratios[k])
        return "ok", ratios[top], (
            f"parity vs main: every depth below the {KL_RATCHET_FLOOR:g} materiality floor "
            f"(worst {ratios[top]:.2f}x at {top}, {pr[top]:.9f}) [{detail}]")
    worst = max(graded, key=lambda k: ratios[k])
    ratio = ratios[worst]
    note = (f"KLD {pr[worst]:.9f} vs main {mn[worst]:.9f} — {ratio:.2f}x at {worst} "
            f"(worst of {len(graded)} material depths of {len(shared)}) [{detail}]")
    # A depth main measured but the PR did not is lost coverage, not a pass. measure_accuracy
    # already hard-fails on a partial suite; this catches the case where the two sides were
    # measured with different K3_PARITY_DEPTHS, which would otherwise silently narrow the gate.
    dropped = sorted(set(mn) - set(pr), key=_depth_sort_key)
    if dropped:
        note += f"  [WARNING: not measured on the PR: {', '.join(dropped)}]"

    if ratio >= KL_RATCHET_REJECT:
        return "reject", ratio, f"{note} (>= {KL_RATCHET_REJECT}x — accuracy regression)"
    if ratio >= KL_RATCHET_WARN:
        return "warn", ratio, f"{note} (>= {KL_RATCHET_WARN}x — worse than main)"
    return "ok", ratio, note


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
    top1, kl, depths = measure_accuracy(f"#{num}")
    res, out, rc = _box_eval(f"#{num}", frontier=frontier, seal=seal, top1=top1, kl=kl)
    # Carry the FULL-PRECISION per-depth parity beside the harness payload.
    #
    # res["kl"] is not this number: it went to the box as --kl, and label.py writes it into
    # RESULT_JSON as round(kl, 4). The ratchet used to read it back from there and compare
    # it against main's un-rounded value -- 4 significant decimals against 9. At KL ~0.0067
    # that quantises the ratio by about 1.5%, one-sided, right where the 1.25x line sits,
    # and it printed as "0.0067000" so the lost precision was invisible in the log.
    res["kl_depths"] = depths
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


def post(repo, num, res, dry_run, parity=None):
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
    # Parity against main measured THIS ROUND, printed whatever the verdict. The absolute
    # bars cannot show drift, so an unannotated "passes clean" is exactly how #74 moved main's
    # baseline 1.53x with no round reporting anything. A number a human can see beats a
    # threshold nobody is told about.
    if parity:
        verdict, ratio, note = parity
        rows.append(("parity vs main", note))
    table = "\n".join(f"| {k} | {v} |" for k, v in rows if v not in (None, ""))
    banner = ""
    if parity and parity[0] == "reject":
        banner = (f"\n> **Accuracy regression.** {parity[2]}\n>\n"
                  f"> Labelled `{KL_REGRESSION_LABEL}`, which blocks the automatic merge. A "
                  "speed win bought with parity is a trade a maintainer should make on "
                  "purpose; remove the label to allow it.\n")
    elif parity and parity[0] == "warn":
        banner = (f"\n> **Note:** {parity[2]}. Under the absolute bar, so not blocking — "
                  "recorded so the drift is visible.\n")
    body = (
        "/eval RESULT_JSON " + json.dumps(res, separators=(",", ":")) + "\n\n"
        "### Node measurement\n\n"
        "| metric | value |\n|---|--:|\n" + table + "\n" + banner + "\n"
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
    index_run(repo_log, rid, res, json.loads(body), num, dry_run)
    print(f">> #{num}: published runs/{rid} to {repo_log}")
    return True


def index_run(repo_log, rid, res, receipt, num, dry_run):
    """Append the run to ledger.jsonl and index.json.

    THE INDEX IS THE PAGE. The log's README calls index.json "newest-first summary, for the
    page" and ledger.jsonl "append-only, one line per run" -- but only kimi_k3_attest.py ever
    wrote them, and that is the BOX-side publisher, which cannot run: the box has no git
    write credentials, deliberately, because a machine being judged should not be able to
    rewrite the ledger judging it. When publishing moved here the index maintenance was left
    behind, so 24 of 25 runs never reached either file and both had been frozen since
    2026-07-31 on a single 3.55 BASELINE row.

    Same schema as the sealer, so the two paths cannot disagree about what a row looks like.
    Non-fatal on failure: the run directory is the record of truth and is already published;
    a missing index row is a display bug, and losing the receipt over one would be worse.
    """
    att = receipt.get("attestation", {}) if isinstance(receipt, dict) else {}
    entry = {
        "run_id": rid,
        "timestamp_utc": att.get("timestamp_utc", ""),
        "label": res.get("label"),
        "tps": res.get("tps"),
        "top1": res.get("top1"),
        "kl": res.get("kl"),
        "commit": att.get("code", {}).get("commit", "") or res.get("commit", ""),
        "attestation_type": receipt.get("attestation_type", "") if isinstance(receipt, dict) else "",
        "pr": num,
    }
    if dry_run:
        print(f"--- dry-run: would index {rid} ({entry['label']} tps={entry['tps']})")
        return True
    ok = True
    for path, mutate in (("ledger.jsonl", lambda cur: cur + json.dumps(entry, sort_keys=True) + "\n"),
                         ("index.json", lambda cur: json.dumps(
                             [entry] + (json.loads(cur) if cur.strip() else []), indent=2))):
        r = gh(["api", f"repos/{repo_log}/contents/{path}"])
        sha, cur = "", ""
        if r.returncode == 0:
            try:
                meta = json.loads(r.stdout or "{}")
                sha = meta.get("sha", "")
                cur = base64.b64decode(meta.get("content", "")).decode()
            except (ValueError, KeyError):
                pass
        args = ["api", "-X", "PUT", f"repos/{repo_log}/contents/{path}",
                "-f", f"message=index {rid}: {entry['label']} tps={entry['tps']} (PR #{num})",
                "-f", "content=" + base64.b64encode(mutate(cur).encode()).decode()]
        if sha:
            args += ["-f", f"sha={sha}"]
        if gh(args).returncode != 0:
            print(f"!! #{num}: could not update {path} — the run is published but the page "
                  "will not show it", file=sys.stderr)
            ok = False
    return ok


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


def _pr_state(repo, num):
    r = gh(["pr", "view", str(num), "-R", repo, "--json",
            "mergeStateStatus,mergeable,labels,headRefOid,state"])
    try:
        return json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return {}


def wait_mergeable_state(repo, num, tries=20, delay=6):
    """Poll until GitHub has actually COMPUTED mergeability, rather than reading UNKNOWN.

    mergeStateStatus is UNKNOWN while GitHub recomputes in the background, and it returns to
    UNKNOWN every time the base branch moves -- which this bot does to itself when it commits
    the frontier. Reading it too early is not a stale answer, it is NO answer, and the code
    treated it as one: #57 reported non-BEHIND before the recompute finished, so the branch
    update was skipped, and minutes later the merge was refused for being BEHIND. One race
    caused both halves of that.

    Returns the settled info dict; on timeout returns the last reading and says so, because a
    guess dressed as a fact is worse than a slow round.
    """
    for i in range(tries):
        info = _pr_state(repo, num)
        # No data at all is not "still recomputing" -- it is "cannot read this PR" (bad repo,
        # no auth, deleted). Polling that for two minutes helps nobody and made the guard
        # tests sleep; one retry covers a transient blip, then give up.
        if not info:
            if i >= 1:
                print(f"!! #{num}: cannot read merge state from {repo}", file=sys.stderr)
                return {}
            time.sleep(min(delay, 2))   # a failed read is not a recompute; retry briefly
            continue
        if (info.get("mergeStateStatus") or "UNKNOWN") != "UNKNOWN":
            return info
        time.sleep(delay)
    print(f"!! #{num}: mergeability still UNKNOWN after {tries * delay}s — proceeding on an "
          f"uncomputed state, which the merge guards treat as not-clear", file=sys.stderr)
    return _pr_state(repo, num)


def clear_stale_tier(repo, num, dry_run):
    """Remove any eval:* label BEFORE posting this round's result. Returns True only when
    the PR is VERIFIED to carry no eval:* label afterwards.

    Without this, wait_for_tier is satisfied by the PREVIOUS round's tier. #59 carried
    eval:none from a round scored against a 4.53 frontier; the wait found an eval:* label
    instantly and returned, and eval-label.yml only later replaced it with eval:xs. Nothing
    broke because #59 was not the winner -- but a merge decision made on a tier measured
    against a frontier that has since moved is precisely the mispricing this whole loop
    exists to prevent.

    Clearing first makes it unambiguous: any eval:* present afterwards belongs to this
    round. It also fails in the safe direction -- if eval-label.yml never runs, the PR ends
    with NO tier and merge_blockers refuses, rather than merging on a stale one.

    THE RETURN VALUE IS THE WHOLE GUARANTEE, AND IT HAS TO BE EARNED. gh() does not raise:
    it hands back a CompletedProcess and an unchecked DELETE loop treats a 403 exactly like
    a success. That printed "cleared stale tier" over a label that was still there, and
    wait_for_tier then returned it instantly -- #59 again, this time with a log line
    asserting it had been prevented. Two silent-failure paths, both closed here: an
    unreadable state is not "no labels", and an issued DELETE is not a removed label.

    Re-read rather than trust the exit codes. A 404 means someone else removed it, which is
    a fine outcome and a failed call; only what is on the PR afterwards decides anything.
    """
    info = _pr_state(repo, num)
    if not info:
        print(f"!! #{num}: cannot read labels from {repo} — refusing to assume any tier "
              f"found later belongs to this round", file=sys.stderr)
        return False
    stale = sorted(l for l in {l.get("name", "") for l in (info.get("labels") or [])}
                   if l.startswith("eval:"))
    if not stale:
        return True
    if dry_run:
        print(f"--- dry-run: would clear stale {', '.join(stale)} from #{num}")
        return True

    errs = []
    for l in stale:
        r = gh(["api", "-X", "DELETE",
                f"repos/{repo}/issues/{num}/labels/{urllib.parse.quote(l, safe='')}"])
        if r.returncode != 0:
            tail = (r.stderr or "").strip().splitlines()
            errs.append(f"{l}: {tail[-1] if tail else 'no stderr'}")

    after = _pr_state(repo, num)
    if not after:
        print(f"!! #{num}: cannot confirm the stale tier was cleared — treating it as not "
              f"cleared", file=sys.stderr)
        return False
    left = sorted(l for l in {l.get("name", "") for l in (after.get("labels") or [])}
                  if l.startswith("eval:"))
    if left:
        print(f"!! #{num}: could not clear stale tier ({', '.join(left)}) — "
              f"{'; '.join(errs) if errs else 'the label is still on the PR'}", file=sys.stderr)
        return False
    print(f">> #{num}: cleared stale tier ({', '.join(stale)}) — it was measured against a "
          f"frontier that has since moved")
    return True


def clear_moot_rebase_label(repo, num, evaluated_sha, dry_run):
    """Remove needs-rebase when the re-measurement it demanded has just happened.

    The label means ONE thing: "the tier on this PR was measured against a frontier that
    has moved". A fresh evaluation at the CURRENT head against the CURRENT frontier is
    precisely the re-measurement it asks for, so a label that survives it is no longer
    information — and because needs-rebase is in NEVER_MERGE_LABELS, it is an active
    hazard: #59 carried one left over from #63's merge, was re-evaluated at the same head
    against the ratcheted frontier and won the round, and the stale label alone would have
    made merge_blockers refuse the bot's own winner. A human checked and removed it by
    hand; this is that check, encoded.

    Moot requires BOTH:
      - the live head equals the sha this round evaluated — a push since the eval means
        the tier belongs to a commit that is no longer what would merge, and the label
        (whatever set it) is doing its job;
      - the branch is not DIRTY — conflicts are real staleness no re-measurement cures.

    Returns True when the PR is verified to carry no needs-rebase afterwards. Same
    discipline as clear_stale_tier: an unreadable state is not "no label", and an issued
    DELETE is not a removed label — re-read and believe only the PR.
    """
    live = _pr_state(repo, num)
    if not live:
        print(f"!! #{num}: cannot read PR state — leaving any {REBASE_LABEL} in place",
              file=sys.stderr)
        return False
    labels = {l.get("name", "") for l in (live.get("labels") or [])}
    if REBASE_LABEL not in labels:
        return True
    head = (live.get("headRefOid") or "").lower()
    want = (evaluated_sha or "").lower()
    if not want or not head or not (head.startswith(want) or want.startswith(head)):
        print(f"!! #{num}: {REBASE_LABEL} stands — head {head[:8]} is not the evaluated "
              f"{want[:8] or '?'}, so the fresh tier does not belong to this head",
              file=sys.stderr)
        return False
    if live.get("mergeStateStatus") == "DIRTY":
        print(f"!! #{num}: {REBASE_LABEL} stands — merge conflicts are real staleness",
              file=sys.stderr)
        return False
    if dry_run:
        print(f"--- dry-run: would clear {REBASE_LABEL} from #{num} (re-measured at {head[:8]})")
        return True
    gh(["api", "-X", "DELETE", f"repos/{repo}/issues/{num}/labels/{REBASE_LABEL}"])
    after = _pr_state(repo, num)
    left = {l.get("name", "") for l in ((after or {}).get("labels") or [])}
    if not after or REBASE_LABEL in left:
        print(f"!! #{num}: could not clear {REBASE_LABEL} — it is still on the PR",
              file=sys.stderr)
        return False
    print(f">> #{num}: {REBASE_LABEL} cleared — this round re-measured it at {head[:8]} "
          f"against the current frontier")
    return True


def wait_for_tier(repo, num, tries=30, delay=10):
    """Poll until eval-label.yml has applied the eval:* label for the comment just posted.

    THE TIER IS NOT THE BOT'S TO WRITE, AND THAT IS THE POINT. The bot posts /eval; the
    workflow re-derives the tier from reference.lock on the protected branch and applies
    exactly one label -- asynchronously, in CI. Asking for that label immediately after
    posting therefore always found nothing, so --merge-admin could never merge a PR it had
    just evaluated: it refused "no eval:* tier" on a PR whose tier landed seconds later.

    Waiting is the fix. Merging without one is NOT: the label is the trusted side's verdict,
    and the entire design is that the bot does not act on its own number.
    """
    for i in range(tries):
        info = _pr_state(repo, num)
        if not info:
            if i >= 1:
                print(f"!! #{num}: cannot read labels from {repo}", file=sys.stderr)
                return []
            time.sleep(min(delay, 2))   # a failed read is not a recompute; retry briefly
            continue
        labels = {l.get("name", "") for l in (info.get("labels") or [])}
        hit = sorted(l for l in labels if l.startswith("eval:"))
        if hit:
            print(f">> #{num}: tier applied by eval-label.yml — {', '.join(hit)}")
            return hit
        time.sleep(delay)
    print(f"!! #{num}: no eval:* tier after {tries * delay}s. eval-label.yml may have failed "
          f"or is still queued; refusing to merge on the bot's own number", file=sys.stderr)
    return []


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
    # Settled, not merely current: this runs right after the round moved main, so an
    # uncomputed UNKNOWN here is what refused #57 for being BEHIND after skipping its update.
    live = wait_mergeable_state(repo, num) if waiting_is_blocking else _pr_state(repo, num)
    labels = {l.get("name", "") for l in (live.get("labels") or pr.get("labels") or [])}
    hit = labels & NEVER_MERGE_LABELS
    if hit:
        bad.append(f"labels {sorted(hit)}")
    # Membership, not startswith. `eval:none` starts with "eval:" and means the OPPOSITE of
    # having passed -- it is eval-label.yml saying the gain did not clear the significance
    # gate. Accepting it merged #81 at -31.5%.
    if NO_GAIN_TIER in labels:
        bad.append(f"{NO_GAIN_TIER} — measured no significant gain over the current frontier")
    elif not (labels & SCORING_TIERS):
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
    # A needs-rebase this round already answered (fresh tier at this exact head) must not
    # refuse the round's own winner. The eval loop clears it once; this covers a label that
    # arrived between the eval and the merge — including from this round's own standalone
    # sweep. Genuinely stale labels (head moved, conflicts) survive the check and block
    # below, which is what they are for.
    clear_moot_rebase_label(repo, num, pr.get("headRefOid", ""), dry_run)
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


def sync_rebase_labels(repo, prs, merged_num, dry_run, force_stale=False):
    """After a merge, every other open PR is scored against a frontier that just moved.

    Their tier is stale by definition rather than by suspicion: the denominator changed.
    So they get the label and an explanation. It clears automatically once the branch is no
    longer behind -- a label only a maintainer can remove turns a mechanical state into a
    queue somebody has to babysit, and the miner has already done the work by rebasing.

    force_stale: pass True on the sweep that runs AS THE TAIL OF A MERGE. GitHub recomputes
    mergeability asynchronously, so seconds after main moves every other PR still reads
    UNKNOWN -- and the BEHIND/DIRTY test below, which is right for a standalone sweep,
    silently labels nothing at the one moment staleness is a certainty rather than a state
    to be queried. The merge IS the evidence: every other tier-carrying open PR trails main
    by at least that commit, whether or not GitHub has noticed yet. The no-tier rule still
    applies, and the merged PR itself is still skipped.
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
        stale = state in ("BEHIND", "DIRTY") or force_stale
        # force_stale also suppresses the clear branch: nothing is "current again" seconds
        # after main moved, whatever the not-yet-recomputed state claims.
        current = (not force_stale) and state in ("CLEAN", "UNSTABLE", "BLOCKED")
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
        # Display only. Deliberately NOT the flag name: labels go through the issues REST
        # endpoint below, and echoing a `gh pr edit` flag here made the dry-run describe a
        # code path that no longer exists.
        act = "add" if has else "remove"
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
    winner_pr = {p["number"]: p for p in prs}.get(winner, {})
    # Same moot-check as merge_winner: a needs-rebase this round already answered must not
    # keep the winner out of the auto-merge queue.
    clear_moot_rebase_label(repo, winner, winner_pr.get("headRefOid", ""), dry_run)
    blockers = merge_blockers(repo, winner, winner_pr, waiting_is_blocking=False)
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
    # OLDEST FIRST. `gh pr list` returns newest-first, so a round served the most recent
    # submission first and the longest-waiting one last. That ordering is backwards for a
    # queue that pays people: when several PRs implement the SAME idea -- #86, #87, #89 and
    # #90 are all "capture the decode token as a CUDA graph" -- only one can win, and the
    # rest re-measure against a main that already has it and score eval:none. Whoever is
    # served first should be whoever submitted first.
    #
    # Order does not decide the winner: mark_merge_first ranks by measured tok/s against the
    # shared frontier, not by position. What it decides is who gets measured at all if the
    # round dies partway, and who eats whatever box state the frontier measurement left
    # behind. Both of those should go to the earliest submission.
    prs.sort(key=lambda p: p["number"])
    if args.only_pr:
        prs = [p for p in prs if p["number"] == args.only_pr]
        if not prs:
            sys.stderr.write(f"k3_eval_bot: #{args.only_pr} is not an open PR on {args.repo}\n")
            return 1

    # Settle mergeability BEFORE anything reads it. gh pr list answers UNKNOWN while GitHub
    # recomputes, and the previous round's own frontier commit is what set it recomputing --
    # so the conflict skip has to wait for a real answer or it is decorative.
    resolve_mergeability(args.repo, prs)
    # After mergeability is settled, before the round books anything: the label reports a
    # state the round has just established, and it is the only place a contributor can see
    # why their PR was skipped. --list stays read-only, so it reports and does not label.
    if not args.list:
        sync_conflict_labels(args.repo, prs, args.dry_run)

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
    # Resolve main UP FRONT so every exit below has a log path.
    #
    # _bail used to read main_sha out of locals(), which is unbound if the frontier
    # measurement is what failed -- so the fallback printed "main is unknown, no log path"
    # and published nothing. That is the exact round the rule exists for: the first hardened
    # run died in measure_frontier and left no record, which is the failure mode
    # rounds/<main_sha>/ was added to cover. A path cannot be invented after the fact, so it
    # is established before anything can go wrong.
    _r = gh(["api", f"repos/{args.repo}/commits/main", "--jq", ".sha"])
    main_sha = (_r.stdout or "").strip()
    if not main_sha:
        print(f"!! could not resolve main on {args.repo} — a failed round will have nowhere "
              f"to publish its log", file=sys.stderr)

    def _bail(code, sha=""):
        sys.stdout, sys.stderr = _real_out, _real_err
        publish_round_log(LOG_REPO, sha or main_sha, [], log_buf.getvalue(), args.dry_run)
        return code

    # THE FRONTIER IS MEASURED, NOT PINNED. Do this before anything is posted: eval-label.yml
    # reads reference.lock at comment time, so a comment posted before the lock is reconciled
    # gets a tier derived from the old frontier -- which is the whole failure being fixed.
    try:
        if args.frontier is not None:
            main_tps = float(args.frontier)
            quant = "UD-IQ1_S"
            # --frontier skips the main measurement, so there is NO parity baseline for this
            # round. An empty suite makes kl_ratchet report "unavailable" rather than
            # inventing a comparison; the absolute KL bars still apply underneath, unchanged.
            main_parity = {}
            print(f">> frontier: {main_tps} tok/s (--frontier, main not re-measured; "
                  "no parity baseline, so the KL ratchet is inactive this round)")
            r = gh(["api", f"repos/{args.repo}/commits/main", "--jq", ".sha"])
            main_sha = (r.stdout or "").strip()
        else:
            # Retried, because this one failure discards every PR in the round. A persistent
            # failure still stops it -- falling back to the pinned lock value would score the
            # round against a number main has already beaten, which is the #20 mispricing.
            main_sha, main_tps, quant, main_parity = with_box_retry(
                "frontier", lambda: measure_frontier(args.repo))
    except RuntimeError as exc:
        print(f"frontier measurement failed after retries — {exc}", file=sys.stderr)
        # _bail falls back to the main_sha resolved before the try block, so this round is
        # published under rounds/<sha>/ even though it sealed nothing.
        return _bail(1)

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
        # The frontier commit above moved main, so GitHub is recomputing every open PR.
        # Asking before it finishes returns UNKNOWN and the update gets skipped -- then the
        # merge is refused for the BEHIND that the skipped update would have fixed.
        state = wait_mergeable_state(args.repo, num).get("mergeStateStatus")
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
    # PRs whose previous tier could not be cleared. They are still evaluated, posted and
    # sealed -- the measurement is good -- but merge_blockers only checks that SOME eval:*
    # label is present, so on those PRs a stale tier would satisfy it. Excluded from the
    # merge decision rather than merged on a number measured against an older frontier.
    unsafe_tier = set()
    for pr in eligible:
        num = pr["number"]
        print(f"#{num}: evaluating {pr['headRefOid'][:8]} on {NODE} "
              f"against frontier {frontier} …")
        try:
            # Same reasoning as the frontier, one PR down: a collective that failed to
            # initialise is not a verdict on this PR's kernels, and losing its round over one
            # is the same unfairness in miniature. A refusal from the harness is NOT retried.
            res = with_box_retry(
                f"#{num}",
                lambda: evaluate(pr, args.repo, frontier, seal=not args.no_seal))
        except RuntimeError as exc:
            print(f"#{num}: eval failed — {exc}", file=sys.stderr)
            continue
        print(f"#{num}: tps={res.get('tps')} top1={res.get('top1')} kl={res.get('kl')} "
              f"ms/token={res.get('ms_per_token')} — tier is eval-label.yml's to derive")
        # Parity against main measured minutes ago on this same box, depth by depth.
        # Reported every time, blocking only past KL_RATCHET_REJECT.
        #
        # Fed from res["kl_depths"] -- the controller's own full-precision measurement --
        # NOT res["kl"], which has been through label.py's round(kl, 4).
        parity = kl_ratchet(res.get("kl_depths"), main_parity)
        print(f"#{num}: {parity[2]}")
        if parity[0] == "reject":
            print(f"!! #{num}: accuracy regression — labelling {KL_REGRESSION_LABEL}, which "
                  "blocks the automatic merge", file=sys.stderr)
            if args.dry_run:
                print(f"--- dry-run: would label #{num} {KL_REGRESSION_LABEL}")
            else:
                gh(["label", "create", KL_REGRESSION_LABEL, "-R", args.repo,
                    "--color", "B60205",
                    "--description", "parity is worse than main's, measured the same round"])
                # NOT `gh pr edit --add-label`: it queries projectCards, which GitHub has
                # deprecated, so it exits 1 even where the repo has no projects. merge-first
                # and merge-conflict were both moved to the REST endpoint for exactly this
                # reason; THIS call was missed, so the label that blocks the merge never
                # landed. Observed on #104 and #115: both measured a parity regression, both
                # printed the warning, and both ended the round carrying only their eval:*
                # tier. Within a round unsafe_tier still excluded them, but the label IS the
                # durable enforcement -- a later round or a human sees a clean PR.
                lr = gh(["api", "-X", "POST", f"repos/{args.repo}/issues/{num}/labels",
                         "-f", f"labels[]={KL_REGRESSION_LABEL}"])
                if lr.returncode != 0:
                    # The label is what stops the merge, so failing to apply it must not
                    # leave the round merging the PR anyway.
                    print(f"!! #{num}: could not apply {KL_REGRESSION_LABEL} "
                          f"({lr.stderr.strip()[:120]}) — excluding it from the merge decision",
                          file=sys.stderr)
                    unsafe_tier.add(num)
        if res.get("receipt_id"):
            publish_receipt(LOG_REPO, num, res, BOX_RECEIPTS, args.dry_run)
        # This evaluation IS the re-measurement a leftover needs-rebase demanded, so clear
        # it now — in EVERY mode, not just merge modes. A plain round that leaves the moot
        # label standing hands the next merge-mode round (or a human) a winner that
        # merge_blockers refuses on a label whose complaint was already answered: #59.
        clear_moot_rebase_label(args.repo, num, pr.get("headRefOid", ""), args.dry_run)
        # Clear the previous round's tier BEFORE posting, so the label that appears after
        # can only be this round's. Otherwise wait_for_tier returns instantly on a stale one.
        tier_cleared = True
        if args.merge_admin or args.auto_merge:
            tier_cleared = clear_stale_tier(args.repo, num, args.dry_run)
        post(args.repo, num, res, args.dry_run, parity=parity)
        # eval-label.yml applies the tier asynchronously. Wait for it here, once, rather
        # than letting the merge decision below read a label that has not landed yet.
        if not args.dry_run and (args.merge_admin or args.auto_merge):
            if tier_cleared:
                wait_for_tier(args.repo, num)
            else:
                # Waiting is pointless when a stale label is still there -- it would return
                # on the first poll, on the old tier. Say so and take this PR out of the
                # merge decision instead of pretending the wait meant something.
                unsafe_tier.add(num)
                print(f"!! #{num}: evaluated and posted, but excluded from this round's merge "
                      f"decision — a tier from an earlier round is still on it",
                      file=sys.stderr)
        results.append((num, res))
        evaluated += 1

    # Reconcile the stale-tier labels BEFORE any merge decision, not after. needs-rebase is
    # in NEVER_MERGE_LABELS, so one left over from before the branches were brought current
    # blocks the merge it was never meant to block. That is exactly how #20 ended up labelled
    # merge-first and then refused in the same round.
    if args.merge_first or args.auto_merge or args.merge_admin:
        sync_rebase_labels(args.repo, prs, merged_num=-1, dry_run=args.dry_run)

    # The merge decision runs on the PRs whose tier is known to be this round's. Everything
    # else -- posting, sealing, the round log -- still sees every result.
    mergeable = [r for r in results if r[0] not in unsafe_tier]

    # A WINNER HAS TO HAVE WON. Ranking by absolute tok/s makes the least-bad result of a bad
    # round the "largest verified gain": with one result, a REGRESSION is trivially the
    # maximum. That is how #81 merged at 14.54 against a 21.24 frontier and cost main 31.5%.
    #
    # Compared against the frontier this round measured, which is the same baseline the tier
    # is derived from -- so this agrees with eval-label.yml by construction rather than by
    # coincidence. Reported per PR, because "nothing merged" and "nothing was faster than
    # main" are different rounds and the operator needs to know which one happened.
    losers = [(n, r) for n, r in mergeable if float(r.get("tps") or 0) <= frontier]
    for num, res in losers:
        print(f">> #{num}: {res.get('tps')} tok/s does not beat the {frontier} tok/s frontier "
              "— not a merge candidate")
    mergeable = [(n, r) for n, r in mergeable if float(r.get("tps") or 0) > frontier]

    if args.merge_first or args.auto_merge or args.merge_admin:
        mark_merge_first(args.repo, mergeable, args.dry_run,
                         queue_auto_merge=args.auto_merge, prs=prs)

    if args.merge_admin and mergeable:
        winner = max(mergeable, key=lambda r: r[1].get("tps") or 0)[0]
        by_num = {p["number"]: p for p in prs}
        if merge_winner(args.repo, winner, by_num.get(winner, {}), args.dry_run):
            # Only after something actually merged does the frontier move, so the rebase
            # sweep is conditional on the merge -- labelling everything needs-rebase after a
            # merge that was blocked would tell every contributor to redo work for nothing.
            #
            # force_stale: this sweep runs SECONDS after main moved, while GitHub still
            # reports every other PR as UNKNOWN. Waiting for BEHIND here labels nothing at
            # the one moment staleness is a certainty — the merge itself is the evidence.
            sync_rebase_labels(args.repo, prs, winner, args.dry_run, force_stale=True)
    elif args.merge_admin:
        # "no results" and "results, none of them safe to merge on" are different rounds and
        # the operator reading this needs to know which one happened.
        if losers:
            beaten = ", ".join(f"#{n} at {r.get('tps')}" for n, r in losers)
            print(f"nothing merged — no PR beat the {frontier} tok/s frontier this round "
                  f"({beaten})", file=sys.stderr)
        elif unsafe_tier:
            print(f"no results safe to merge — {len(unsafe_tier)} PR(s) still carry a tier "
                  f"from an earlier round: {', '.join(f'#{n}' for n in sorted(unsafe_tier))}",
                  file=sys.stderr)
        else:
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
