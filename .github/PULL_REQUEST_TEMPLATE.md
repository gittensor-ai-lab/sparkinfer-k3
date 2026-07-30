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

- [ ] Tested on **8× H200** (`sm_90`) — M1
- [ ] Tested on **8× B200** (`sm_100`) — M2
- [ ] Tested on **4×+ B300** (`sm_103`) — M3

**Decode tok/s** — from `bench/scripts/kimi_k3_baseline.sh --node <node>`, UD-Q2_K_XL:

| context | before (main) | after (this PR) |
|---|--:|--:|
| 128 | | |
| 512 | | |
| 4k | | |
| 32k | | |

**Prefill pp tok/s** — same command, same node:

| context | before (main) | after (this PR) |
|---|--:|--:|
| 4k | | |
| 32k | | |

**1M context** (`--longctx`, 1 rep — capability probe, not a median):

| | loads | decode tok/s | prefill pp |
|---|:-:|--:|--:|
| this PR at 1,048,576 | | | |

<!-- Paste the sweep output backing the numbers above. Isolated-kernel microbenchmarks are
     welcome as extra evidence but do NOT substitute for an end-to-end before/after. -->

```text
# paste bench/scripts/kimi_k3_baseline.sh output here (before -> after)
```

## Checklist

- [ ] `python3 bench/scripts/test_kimi_k3_baseline.py` passes
- [ ] `bench/scripts/kimi_k3_baseline.sh --node <node> --dry-run` resolves
- [ ] If a baseline was pinned: the emitted `bench/results/*.json` is committed alongside it
