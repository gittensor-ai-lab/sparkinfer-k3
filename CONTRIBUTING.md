# Contributing to sparkinfer

sparkinfer is the engineering arm of **SN74 on Gittensor**. Contributions are rewarded
for **real, verified inference-speed engineering** — not benchmark gaming. This guide is
how to make a contribution that counts.

## Built through Gittensor

Gittensor helps power SPARKINFER through SN74: the project receives subnet emissions,
contributors submit source PRs, the evaluator rebuilds those PRs on a real 8x H200 node,
and rewards are assigned from verified marginal speedups that keep correctness intact.
You do not need to be in Discord or understand the subnet internals to contribute, but the
source of the incentive loop is clear: SPARKINFER is built through **SN74 on Gittensor**.

## Principles

- **Source-required & reproducible.** The validator builds your PR from source. No
  opaque prebuilt images — the shipped prebuilt binaries are a *run* convenience, not a
  submission format.
- **Correctness first.** A faster kernel that changes the model's output is worth zero.
  Every change is gated against a frozen reference (see *Accuracy gate* below).
- **General, not overfit.** This repo targets **one model: Kimi K3** (2.8T, 93 layers,
  896 routed experts) at **UD-IQ1_S** on **8x H200**. "Not overfit" here means the win must
  hold across *context lengths* (128 -> 128k) and survive the accuracy gate — not a
  cherry-picked shape. A kernel that only helps ctx 128 has not moved the case anyone runs.
- **Hopper only.** Targets `sm_90` (8x H200). CUDA 12.8+ (13 works). `sm_100`
  (Blackwell / RTX Spark) is **not** a target here — that is the parent repo's track.

## Before you open a PR

```bash
# 1. build + tests
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90 && cmake --build build -j && ctest --test-dir build
python3 bench/scripts/test_kimi_k3_baseline.py          # 78 tests, no GPU needed

# 2. does it still generate? (8x H200, all 93 layers, text in text out)
export KIMI_K3_MODEL=$KIMI_K3_MODELS_DIR/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf
bench/scripts/kimi_k3_run.sh "The capital of France is" 32 0,1,2,3,4,5,6,7

# 3. speed AND accuracy in one command — this is what produces your verdict
bench/scripts/kimi_k3_eval.sh --node h200x8 --seal
```

`bench.sh` and `accuracy.sh` are the **inherited Qwen3-30B track** and do not apply here.
K3's harness is `kimi_k3_*`.

**Accuracy gate, and it runs FIRST.** `kimi_k3_eval.sh` compares your build's logits
against the captured llama.cpp reference (`bench/refdata/`) on identical weights and
identical token ids. `label.py` REJECTs below **top-1 0.90** or **mean KL 0.20** no matter
how fast the run was — a speedup that erodes parity is not a speedup worth taking.

Current measured parity on `main`: **top-1 100%, mean KLD 4.05e-03**. Note that KLD is
~400x the 1e-5 same-implementation bar, from a known and accepted cause (K3 keeps f32
activations where ggml quantizes them before a quantized mat-vec). Do not go hunting it
as a bug; do not make it worse. If `compute-sanitizer` is available, your kernels
must be clean (0 errors).

## How rewards work (SN74 on Gittensor)

**Speedup-only.** You're paid for the **verified marginal speedup** your PR adds over the
current best ("frontier"), not your rank — so "copy the leader + ε" pays ≈ ε. Your PR is built and benchmarked on the **same 8x H200 node** as the recorded frontier,
against the pinned reference in `bench/scripts/reference.lock`, so speed differences between eval
machines can't inflate or hide your result.

**There is no autonomous eval bot in this repo — yet.** The round workflow the parent
repo runs (queue every PR against the same `main`, label the winner `merge-first`,
auto-merge it, tag the rest `needs-rebase` and re-grade them) does **not** exist here.
`eval/pr_eval_bot.py` is inherited and Qwen-shaped: it drives `bench.sh`/`accuracy.sh`,
not the K3 harness.

What actually happens today:

1. A maintainer runs `bench/scripts/kimi_k3_eval.sh --node h200x8 --seal` on the node.
   That measures, scores, seals the verdict into a Polaris receipt, and publishes the run
   to [`sparkinfer-k3-log`](https://github.com/gittensor-ai-lab/sparkinfer-k3-log).
2. They post `/eval RESULT_JSON {…}` on the PR.
3. `.github/workflows/eval-label.yml` **re-derives** the tier from the reported
   measurements rather than trusting the reported label, verifies the receipt against the
   log, and applies exactly one `eval:*` label.

So: **one open PR at a time, evaluated when a maintainer runs it.** No auto-merge, no
`merge-first`/`needs-rebase`/`re-evaluate`, no oldest-first polling. Keep your branch
rebased on `main`; the frontier moves when something merges, and your gain is measured
against the merged frontier.

The applied labels are **lowercase** — `eval:xs` … `eval:xl`, plus `eval:baseline` for the
first verified entry on a node/quant. The tier is never set by hand. A speedup is scored the same wherever it lands (`kernels/`, `runtime/`, `moe/`); there is **no
per-subsystem budget**. Tiers are bands of **% speedup over the frontier** — `XS` 2–3.5%, `S`
3.5–6%, `M` 6–10%, `L` 10–18%, `XL` >18% (a gain under 2% is within measurement noise → `none`).
Because they scale with the frontier, every tier stays reachable as decode speed grows.

**Non-speedup PRs are welcome — but score 0.** Bug fixes, refactors, tests, benchmarks, docs,
and tooling are appreciated and we'll review and merge good ones, but SN74 emits only for
verified speedups, so they earn no reward. (The eval/scoring harness is maintainer-owned — see
*Maintainer-owned paths* below.)

**Evaluation is opt-in and proof-gated.** A node eval happens only when you tick
**`- [x] Tested on 8x H200`** *and* fill the template's **decode tok/s** table with a real
end-to-end improvement (`after > before`, from `bench/scripts/kimi_k3_baseline.sh` — not an
isolated-kernel microbenchmark). Fill the **128k** row: that is the configuration this repo runs.
- Box not ticked on a perf-bearing diff → the `node-attestation` job labels `needs-node-run`.
  It **labels only — it never closes a PR.**
- Box ticked with an empty or placeholder table → a maintainer will not queue the run.
There is **no override** — a tier is only ever applied from a real measured run.

> ⚠️ Tick that box **only if you actually ran it on the node you ticked** and pasted the benchmark log.
> Checking it without testing is false attestation — it is treated as gaming and the account will
> be **blocked** (added to the denylist), the same as sybil farming.

### One open PR at a time

**You may have exactly ONE open pull request.** Open a second while the first is still
open and it is closed automatically, newest first, so the one under review survives.

This is not only anti-spam. The eval scores a PR against the **merged** frontier, so two
open PRs from one author are both measured against a baseline the other is about to move —
the marginal-gain number only means something if they land one at a time.

### Anti-gaming (how submissions are kept honest)

`copycat-guard.yml` fingerprints every PR diff against merged history on open and on push,
so gaming is caught automatically without anyone having to look:

- **Copycatting.** Re-submitting an earlier PR's diff — *even with a few extra lines bolted on to
  look original or slip past the evaluator* — is flagged by diff-containment fingerprint. A first
  copycat is acted on IMMEDIATELY — there is no strike accumulation and no 5-day freeze.
  Enforcement is tiered by how much of an earlier PR your diff contains:

  | containment | what happens |
  |---|---|
  | **>= 90%** | account added to the denylist and the PR closed. Effectively permanent. |
  | **80-90%** | explanatory comment + the PR is auto-closed. You can open a real one next. |
  | **70-80%** | `llm-judge` label + a semantic review. **Never closes, never blocks** — a maintainer decides. |
  | < 70% | ignored. |

  The gap between the tiers is deliberate: a missed copycat costs one PR's emission, a
  false block costs a real contributor their access permanently. The 70-80% band exists
  because overlap alone cannot separate a renamed copycat from an independent contributor
  who touched the same hot function, so that band is reviewed rather than punished.
  Logged in [`.github/copycats.json`](.github/copycats.json) / [`COPYCATS.md`](.github/COPYCATS.md).
- **Sybil / duplicate-account farming** (one operator pushing under multiple GitHub identities, or
  shadowing others' work) is blocked outright; evidence is recorded in [`.github/FLAGGED.md`](.github/FLAGGED.md).
- **No override.** There is no way to force-evaluate around the gate — not even for a maintainer.
  Real, original, frontier-advancing work is the only thing that scores.

## Maintainer-owned paths (eval, scoring & governance)

The evaluation harness and scoring config are **maintainer-owned** and must not be changed
in a contributor PR. They decide labels and emissions and are the trust anchor validators
rely on — so a change here, however well-intentioned, can't ride in on the same PR it would
score. These paths are protected:

| Path | What |
|---|---|
| `eval/` | inherited Qwen-era bot + GPU runner (dead in this repo, still guarded) |
| `bench/scripts/label.py` | the scoring function — decides your tier |
| `bench/scripts/kimi_k3_eval.sh`, `kimi_k3_baseline.sh`, `_kimi_k3.sh` | the measurement chain that produces tok/s |
| `bench/scripts/kimi_k3_attest.py` | receipt sealing / Polaris attestation |
| `bench/scripts/compare_logits.py`, `check_reference_lock.py` | the accuracy gate and pin verifier |
| `bench/scripts/reference.lock`, `*.sha256` | the pinned frontier + weight manifests |
| `bench/results/`, `dashboard/data.json` | recorded runs and the frontier ledger |
| `.gittensor/` | intra-repo emission weights |
| `.github/` | CI, `CODEOWNERS`, and this guard |

Everything else under `bench/scripts/` (including `kimi_k3_run.sh` and the Qwen-era
`bench.sh`/`accuracy.sh`) is **not** guarded — those are convenience and legacy, not trust anchors.

**Enforcement.** A required **`sensitive-paths-guard`** check automatically fails any PR from a
non-maintainer that touches these paths, and `CODEOWNERS` requires maintainer review — so such
PRs **cannot merge**, regardless of content. The evaluator also grades with the harness pinned
to the protected branch, so editing it in a PR never affects that PR's own score.

**Improving the harness is welcome — just not via a direct PR.** Open an issue or discussion
describing the change; if a maintainer agrees, they'll land it (with credit). Keep your own PRs
scoped to `kernels/`, `runtime/`, and `moe/` — that's the rewarded optimization work.

## Style & scope

- Match the surrounding code (portable CUDA is the production path; CuTe/tensor-core is
  the opt-in ceiling). Keep kernels readable and commented where non-obvious.
- Reference the bench + accuracy numbers in your PR description (before → after).
- Keep changes focused; one optimization per PR makes the measured delta attributable.

By contributing you agree your work is licensed under the repository's [MIT License](LICENSE).

<!-- check-name verification probe; see PR #29 -->
