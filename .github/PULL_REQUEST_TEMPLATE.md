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

> **All three are required for a perf-bearing PR, and the evaluation loop skips a PR that is
> missing any of them.** The tier is earned on prefill; decode is a guard. A prefill gain
> bought by giving decode back has not moved the engine forward, it has moved work around,
> so the round refuses to score it.
>
> **Tick a box only once you have the number, and put the number in the table.** The loop
> checks the `after (this PR)` cell, not the tick: `*awaits eval round*`, a blank, or
> `no change — no decode path is touched` all read as *not measured* and the PR is skipped.
> A round costs ~25 GPU-minutes per PR, so it measures changes that are already known to do
> something — it is not the place to find out whether yours does.
>
> **Your prefill number must beat the pinned frontier by more than 2%**, or the PR is
> skipped: below that it scores `none` even if exact, so re-deriving it on the node changes
> nothing. The frontier moves as PRs merge — read the current
> `KIMI_K3_H200X8_IQ1S_SPARKINFER_32K_PP` in
> [`bench/scripts/reference.lock`](../bench/scripts/reference.lock) rather than a number
> quoted in an older PR, and prefer measuring `main` yourself over trusting the pin.

**Prefill tok/s @ 32k** — the scored metric. `llama.cpp` does **143.88** here; `main` does
**40.35**, because there is no batched prefill and every prompt token goes through the
single-token decode step.

| | before (main) | after (this PR) |
|---|--:|--:|
| prefill @ 32k | | |

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
     thing to score while sparkinfer was 18x behind llama.cpp there; it is now 3.08x ahead
     (56.82 vs 18.44) and the untouched gap is ingestion, where we are 3.57x BEHIND. -->

<!-- Paste the sweep output backing the numbers above. Isolated-kernel microbenchmarks are
     welcome as extra evidence but do NOT substitute for an end-to-end before/after. -->

```text
# paste bench/scripts/kimi_k3_baseline.sh output here (before -> after)
```

## Checklist

- [ ] `python3 bench/scripts/test_kimi_k3_baseline.py` passes
- [ ] `bench/scripts/kimi_k3_baseline.sh --node <node> --dry-run` resolves
- [ ] If a baseline was pinned: the emitted `bench/results/*.json` is committed alongside it
