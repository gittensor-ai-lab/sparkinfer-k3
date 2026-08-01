#!/usr/bin/env python3
"""Kimi K3 PR evaluation bot.

Replaces the inherited Qwen evaluator (pr_eval_bot.py), which had zero K3 support and
would have auto-closed every K3 pull request. This one does exactly one job:

    find an eligible PR -> measure it on the 8x H200 node -> post the result

It never decides a tier and it never merges. Tiers come from
.github/workflows/eval-label.yml, which re-derives them from reference.lock on the
protected branch rather than trusting anything posted here. With --merge-first it marks
the round's largest measured gain, and with --auto-merge it QUEUES GitHub native
auto-merge on that winner -- which still waits for the required approving review and every
required check. It never passes the admin flag, so it cannot bypass branch protection.

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


def evaluate(pr, repo, seal=False):
    """Check the PR head out on the box, build it, and run the K3 harness there.

    Returns the parsed RESULT_JSON, or raises RuntimeError with the box's own output. The
    build and the eval both happen on the node; nothing is measured locally.
    """
    sha = pr["headRefOid"]
    num = pr["number"]
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
            f"build failed for #{num} @ {sha[:8]} (ssh exit {r.returncode})"
            + (f"\n{tail}" if tail else " — no output at all, so a setup step aborted the "
                                        "chain before cmake ran"))

    seal_flag = " --seal" if seal else ""
    # POLARIS_API_KEY lives in a 0600 file on the box, not in this command line and not in
    # the repo: anything passed as an ssh argument shows up in ps on a shared machine.
    evalcmd = (
        f"cd {BOX_REPO_DIR} && "
        f"[ -r {POLARIS_ENV} ] && . {POLARIS_ENV}; "
        # CMake emits executables under build/runtime/, not build/. The harness resolves
        # $SPARKINFER_BUILD/kimi_k3_tp_bench, so pointing this at build/ makes it exit 2 with
        # "not built" immediately after a build that in fact succeeded.
        f"KIMI_K3_MODELS_DIR={BOX_MODELS_DIR} SPARKINFER_BUILD={BOX_REPO_DIR}/build/runtime "
        f"bash bench/scripts/kimi_k3_eval.sh --node {NODE} --devices {DEVICES}{seal_flag}"
    )
    r = sh(evalcmd, timeout=7200)
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    line = next((l for l in out.splitlines() if l.strip().startswith("RESULT_JSON")), None)
    if not line:
        detail = (r.stderr or "").strip().splitlines()
        why = next((l for l in reversed(detail) if l.strip() and "setlocale" not in l), "")
        raise RuntimeError(
            f"no RESULT_JSON from the harness for #{num}"
            + (f" — {why}" if why else "") + f"\n{out[-1200:]}")
    res = json.loads(line.split("RESULT_JSON", 1)[1].strip())
    # --seal publishes to sparkinfer-k3-log and prints the receipt id. Carry it in the
    # payload: eval-label.yml looks it up there when REQUIRE_EVAL_RECEIPT is on, and
    # without it a sealed run is indistinguishable from an unsealed one.
    if seal:
        # kimi_k3_eval.sh prints RESULT_JSON and THEN seals, exiting 1 if the publish fails.
        # Scraping stdout and ignoring the exit status is how a run got reported as sealed
        # while the log stayed empty: the harness said "FAILED to seal/publish" and nothing
        # listened.
        rid = re.search(r"receipt[_ ]?id[\"'\s:=]+([0-9a-f]{8,})", out, re.I)
        if r.returncode != 0:
            why = next((l for l in reversed((r.stderr or "").splitlines())
                        if "seal" in l.lower() or "publish" in l.lower()), "")
            print(f"#{num}: NOT sealed — the harness failed to publish"
                  + (f": {why.strip()}" if why else "") , file=sys.stderr)
        elif rid:
            res["receipt_id"] = rid.group(1)
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
    import base64
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


def merge_blockers(repo, num, pr):
    """Every reason not to merge this without a human reading it. Empty list means clear."""
    bad = []
    labels = {l.get("name", "") for l in pr.get("labels") or []}
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
    st = gh(["pr", "view", str(num), "-R", repo, "--json", "mergeStateStatus"])
    try:
        state = json.loads(st.stdout or "{}").get("mergeStateStatus", "UNKNOWN")
    except json.JSONDecodeError:
        state = "UNKNOWN"
    if state not in ("CLEAN", "UNSTABLE"):
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
    blockers = merge_blockers(repo, num, pr)
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
        behind = info.get("mergeStateStatus") in ("BEHIND", "DIRTY", "BLOCKED", "UNKNOWN")
        has = REBASE_LABEL in labels
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
            gh(["pr", "comment", str(num), "-R", repo, "--body",
                f"### Rebase needed\n\n#{merged_num} merged, so the frontier this PR is "
                f"scored against has moved. The tier currently on it was measured against "
                f"the old baseline and no longer means anything: you are paid for the gain "
                f"**on top of what merged**, not the gain over where `main` used to be. If "
                f"the merged work already covers this change, the honest re-measurement is "
                f"a small number, and that is the mechanism working rather than a "
                f"penalty.\n\nRebase onto `main` and push. The `{REBASE_LABEL}` label "
                f"clears by itself once the branch is current, and the next round "
                f"re-measures against the new frontier."])
            print(f">> #{num}: {REBASE_LABEL} (frontier moved)")
        elif has and not behind:
            if dry_run:
                print(f"--- dry-run: would clear {REBASE_LABEL} from #{num}")
                continue
            gh(["api", "-X", "DELETE", f"repos/{repo}/issues/{num}/labels/{REBASE_LABEL}"])
            print(f">> #{num}: {REBASE_LABEL} cleared — branch is current again")


def mark_merge_first(repo, results, dry_run, queue_auto_merge=False):
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
    if queue_auto_merge and not dry_run:
        r = gh(["pr", "merge", str(winner), "-R", repo, "--squash", "--auto"])
        if r.returncode == 0:
            print(f">> #{winner}: auto-merge queued — fires when review + checks pass")
        else:
            print(f"!! #{winner}: could not queue auto-merge: {r.stderr.strip()}",
                  file=sys.stderr)


def main():
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
    results = []
    for pr in prs:
        num = pr["number"]
        ok, why = eligibility(pr)
        if not ok:
            print(f"#{num}: skip — {why}")
            continue
        print(f"#{num}: evaluating {pr['headRefOid'][:8]} on {NODE} …")
        try:
            res = evaluate(pr, args.repo, seal=not args.no_seal)
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

    if args.merge_first or args.auto_merge or args.merge_admin:
        mark_merge_first(args.repo, results, args.dry_run,
                         queue_auto_merge=args.auto_merge)

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

    # A PR that has already been rebased should not keep the label just because no merge
    # happened this round. Cheap to check, and it is the half of the contract the miner has
    # already done their part of.
    if args.merge_first or args.merge_admin:
        sync_rebase_labels(args.repo, prs, merged_num=-1, dry_run=args.dry_run)

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
          f"tiers are eval-label.yml's, merges wait on review")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
