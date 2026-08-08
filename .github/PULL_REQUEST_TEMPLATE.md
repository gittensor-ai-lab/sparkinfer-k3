## Summary

<!-- What this PR adds or changes, and why. One or two lines. -->


## What kind of change is this?

<!-- Tick one. Most PRs in this repo are harness/docs and need no GPU at all. -->

- [ ] **Harness / configs / docs / CI** — verified entirely by the `ci` workflow, no node needed
- [ ] **Perf-bearing code** (`runtime/`, `kernels/`, `moe/`, `server/`, `CMakeLists.txt`) — needs a node run below
- [ ] **Pin change** (`reference.lock`, a `*.sha256` manifest) — needs the backing `bench/results/*.json` committed

---

## Node run

> Only required for perf-bearing changes. CI covers syntax, the no-GPU test suite, configs
> and the `--dry-run` plans — but it cannot run an 802 GiB model, so it cannot tell whether
> this moved the numbers. The `node-attestation` job adds a `needs-node-run` label when
> perf-bearing files change with no node ticked. It **labels only — it never closes a PR.**
>
> Tick a box only if you actually ran it. A pinned baseline must be backed by the
> `bench/results/*.json` the sweep emitted, and the `lock` CI job enforces that.

- [ ] Tested on **8× H200** (`sm_90`)
- [ ] **Prefill measured at 32k** on 8× H200 — this is the scored metric
- [ ] **No 128k decode regression** on 8× H200 — within 1% of the frontier
- [ ] **Blocked — do not evaluate yet.** Something below has to resolve first. Say what, under
      [Known problems](#known-problems).

> **The first three are required for a perf-bearing PR, and the evaluation loop skips a PR
> that is missing any of them.** The tier is earned on prefill; decode is a guard. A prefill
> gain bought by giving decode back has not moved the engine forward, it has moved work
> around, so the round refuses to score it.
>
> **Tick a box only once you have the number, and put the number in the table.** The loop
> checks the `after (this PR)` cell, not the tick: `*awaits eval round*`, a blank, or
> `no change — no decode path is touched` all read as *not measured* and the PR is skipped.
> A round costs ~25 GPU-minutes per PR, so it measures changes that are already known to do
> something — it is not the place to find out whether yours does.
>
> **Tick `Blocked` if you found a problem with your own PR.** It is not a black mark; it is
> the most useful thing you can tell the loop, and it costs you nothing — the round skips
> you instead of spending the node re-deriving a number you already have and already said
> not to act on. Untick it when the problem resolves. **Reorganising this section to explain
> a problem is what to avoid**: the attestation scan keys on the `## Node run` heading and
> the box wording, so a rewritten section reads as *no node run at all*, and a PR that did
> more work than anyone gets labelled as though it had touched no GPU.

**Prefill tok/s @ 32k** — the scored metric.

| | before (main) | after (this PR) |
|---|--:|--:|
| prefill @ 32k | | |

> **Do not copy a frontier number out of this template, an older PR, or a comment.** It goes
> stale every time something merges, and a stale one has already cost real work: three PRs
> were optimised against a pin that understated `main` by 31%, and every one of them measured
> honestly and still landed under the bar.
>
> Read `KIMI_K3_H200X8_IQ1S_SPARKINFER_32K_PP` from
> [`bench/scripts/reference.lock`](../bench/scripts/reference.lock) **on `main`, today** — and
> better, **measure `main` yourself in the same session as your own arm**. That is what #133
> did, and its self-measured `before` landed within 0.2% of what the node independently found.
>
> **Your number has to beat that frontier by more than 2%** or the PR is skipped: below the
> significance gate it scores `none` even if it is exact, so re-deriving it on the node
> changes nothing.

**Decode tok/s @ 128k** — the guard, not the tier. Must stay within **1%** of the frontier.

| | before (main) | after (this PR) |
|---|--:|--:|
| decode @ 128k | | |

**Decode at other contexts** — optional, useful evidence, never scored:

| context | before (main) | after (this PR) |
|---|--:|--:|
| 128 | | |
| 512 | | |
| 4k | | |
| 32k | | |

<!-- Prefill at 32k became the scored metric on 2026-08-05. Decode at 128k was the right
     thing to score while sparkinfer was 18x behind llama.cpp there; it is now well ahead
     and the untouched gap is ingestion, where llama.cpp is still several times faster
     because it batches the prompt and we do not. -->

<!-- Paste the sweep output backing the numbers above. Isolated-kernel microbenchmarks are
     welcome as extra evidence but do NOT substitute for an end-to-end before/after. -->

```text
# paste bench/scripts/kimi_k3_baseline.sh output here (before -> after)
```

## Known problems

<!-- Anything you found that you have not resolved: a parity failure, a case you did not
     cover, a number you cannot yet explain, a diagnostic still in flight.

     Write it here rather than restructuring the sections above — those are machine-read.

     Declaring a defect in your own PR is not held against you. Finding it before the node
     does is worth more to this repo than a clean-looking body, and every rule in this
     template exists because something reached main without someone saying it out loud. -->

_None._

## Checklist

- [ ] `python3 bench/scripts/test_kimi_k3_baseline.py` passes
- [ ] `bench/scripts/kimi_k3_baseline.sh --node <node> --dry-run` resolves
- [ ] If a baseline was pinned: the emitted `bench/results/*.json` is committed alongside it
