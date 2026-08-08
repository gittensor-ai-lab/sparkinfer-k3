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
python3 bench/scripts/test_kimi_k3_baseline.py          # 117 tests, no GPU needed

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
identical token ids. `label.py` REJECTs below **top-1 0.95** or **mean KL 0.05** no matter
how fast the run was — a speedup that erodes parity is not a speedup worth taking.

In practice **top-1 is pass/fail, not a 5% tolerance.** Each probe dumps one logit row, so
top-1 is `argmax_ref == argmax_ours` — a boolean — and the suite takes the worst depth. Any
bar in (0, 1] behaves the same, and all 48 top-1 values in the sealed log are exactly `1.0`.
Treat it as *the argmax must match at every depth*, and treat **KL as the graded gate**: it
sums over all 163,840 vocab entries, so it moves long before an argmax flips (merged runs sit
at 0.004–0.008 against the 0.05 bar).

**It is measured at seven context depths.** The evaluator probes 4, 128, 256, 512, 1024,
2048 and 4096 tokens — nested prefixes of one document — and takes the **worst**
depth, not the average. Until 2026-08-04 the gate was a single 4-token prompt graded on
one next-token distribution, which could not distinguish "correct" from "correct only
while the KV cache is nearly empty". If your change touches the KV cache, attention,
routing or the LM head, expect the deep probes to be the ones that catch it.

References at 8192, 16384 and 32768 are also committed and can be switched on with
`K3_PARITY_DEPTHS`; they are off by default because the deep pass runs through the decode
path one token at a time, so 32768 costs 812 s per measured build against 136 s for 4096.
Ingestion runs at decode speed (42.80 tok/s measured) — see #119.

Two consequences for you:

- A regression at one depth fails the gate even if the other nine are perfect.
- Parity is also **ratcheted against `main` measured in the same round, per depth**, and
  reported on every PR. **Below the 0.05 bar a ratio is never a regression** — it is
  annotated so drift is visible, but it does not block a merge. `accuracy-regression` is
  applied only when a depth is both ≥2.0x main **and** at or over the bar, at which point
  `label.py` REJECTs on the absolute value anyway.

Note that KLD is ~400x the 1e-5 same-implementation bar, from a known and accepted cause
(K3 keeps f32 activations where ggml quantizes them before a quantized mat-vec). Do not go
hunting it as a bug; do not make it worse. If `compute-sanitizer` is available, your
kernels must be clean (0 errors).

**What the gate still does not prove.** 4096 is not the 131,072 you are scored at. The
reference must come from llama.cpp and capture cost grows with depth, so the untested gap
is "everything past 4k" — 32k with the opt-in depths on — not "nothing". See
`bench/refdata/README.md`.

## How rewards work (SN74 on Gittensor)

**Speedup-only.** You're paid for the **verified marginal speedup** your PR adds over the
current best ("frontier"), not your rank — so "copy the leader + ε" pays ≈ ε. Your PR is built and benchmarked on the **same 8x H200 node** as the recorded frontier,
against the pinned reference in `bench/scripts/reference.lock`, so speed differences between eval
machines can't inflate or hide your result.

**The eval bot is `eval/k3_eval_bot.py`.** It finds eligible PRs, measures each on the
pinned 8x H200 node, seals the run, publishes the receipt to
[`sparkinfer-k3-log`](https://github.com/gittensor-ai-lab/sparkinfer-k3-log), and posts the
measurements. It never decides a tier. With `--merge-first` it marks the round's largest
measured gain; with `--merge-admin` it merges that winner and labels the rest
`needs-rebase` (see *Competing PRs* below).

`eval/pr_eval_bot.py` is the inherited Qwen evaluator and does **not** run here — it
refuses a `-k3` repository outright, because its greenlight looks for an RTX 5090 checkbox
this template cannot contain and it closes any PR that fails it.

What happens today:

1. A maintainer runs `bench/scripts/kimi_k3_eval.sh --node h200x8 --seal` on the node.
   That measures, scores, seals the verdict into a Polaris receipt, and publishes the run
   to [`sparkinfer-k3-log`](https://github.com/gittensor-ai-lab/sparkinfer-k3-log).
2. They post `/eval RESULT_JSON {…}` on the PR.
3. `.github/workflows/eval-label.yml` **re-derives** the tier from the reported
   measurements — and from `reference.lock` on the protected branch, not from the payload —
   rather than trusting the reported label, then applies exactly one `eval:*` label.

   **Receipt verification is currently OFF.** The workflow only checks a receipt when the
   `REQUIRE_EVAL_RECEIPT` repository variable is set to `1`, and it is not set. It prints a
   notice saying so on every run. Until that is switched on, a tier proves the arithmetic
   was recomputed on trusted inputs; it does not prove the measurement happened.

So: **one open PR at a time, evaluated when a maintainer runs it.** No auto-merge, no
`merge-first`/`needs-rebase`/`re-evaluate`, no oldest-first polling. Keep your branch
rebased on `main`; the frontier moves when something merges, and your gain is measured
against the merged frontier.

The applied labels are **lowercase** — `eval:xs` … `eval:xl`, plus `eval:baseline` for the
first verified entry on a node/quant. The tier is never set by hand. A speedup is scored the same wherever it lands (`kernels/`, `runtime/`, `moe/`); there is **no
per-subsystem budget**. **Two different denominators, and mixing them up is expensive:**

- **Significance gate — % over the frontier.** The gain must beat **2% of the current
  frontier**, or the label is `none` (inside measurement noise).
- **Tier — the WORSE of two bases, so it can never exceed your measured speedup.** Past the
  gate the tier is `min(delta / llama_ref, delta / frontier)`, where `llama_ref` is pinned in
  `bench/scripts/reference.lock` (**18.4435 tok/s** for `h200x8` / `UD-IQ1_S` **at the scored
  128k context**): `XS` <3.5%, `S` 3.5–6%, `M` 6–10%, `L` 10–18%, `XL` >18%. Any verified
  gain that clears the gate floors at `XS`.

  **On the current scored metric (prefill @32k) the llama term is disabled** and the tier is
  `delta / frontier` — see [The scored metric is PREFILL at 32k](#the-scored-metric-is-prefill-at-32k-from-2026-08-05)
  for why. Everything below describes the anchor as it applies to decode.

  Both halves matter. While the frontier is **below** llama.cpp the llama term is smaller, so
  an un-optimized baseline cannot mint `XL`s from low-hanging fruit. Once the frontier is
  **past** llama.cpp — which K3 now is, at 2.2× — the frontier term is smaller, so a tier
  costs the full percentage over the *current best*. In practice that means **`XL` needs 18%
  over main**, not 18% of a reference main already beat.

  This changed on 2026-08-04. Tier credit used to be capped at *twice* the measured gain,
  which was invisible until the frontier passed 2× llama.cpp and then became the whole rule —
  `XL` was costing 9%. The K3 ladder was re-scored from the sealed receipts; see #122.

### The scored metric is PREFILL at 32k (from 2026-08-05)

**The tier is earned on prompt ingestion of 32,768 tokens.** Decode at 131,072 is still
measured every round, but as a **regression guard**, not the tier basis: if your PR drops
decode more than **1%** below the pinned frontier, the round refuses to score it.

Decode was the right thing to score while sparkinfer was 18× behind llama.cpp there. It is
now 3.08× ahead, and the untouched gap is ingestion:

| @ 32k | tok/s | |
|---|--:|---|
| llama.cpp | **143.88** | batches the prompt |
| sparkinfer | **98.80** | **1.46× behind** — it batches the prompt too, since #148 |

Prompt ingestion was one forward per token until #148 landed batched ingestion on
2026-08-06 (69.02 → 98.80 at 32k). The gap to llama.cpp is now 1.46×, not 3.57×, and what
is left is the batching's own efficiency rather than its absence.

Two consequences worth knowing before you start:

- **The llama anchor is OFF here, so the tier is `delta / frontier`.** The bands are the
  same percentages you would read off decode: `S` +3.5%, `M` +6%, `L` +10%, `XL` +18% over
  the current prefill frontier.

  The anchor is off because 143.88 is not a mature version of what we do — llama.cpp
  **batches** the prompt and sparkinfer walks it token by token, so `delta / 143.88` sizes a
  gain against the feature we have not built rather than against the work in the PR. And
  because the buckets are fractions of the reference, leaving it on made a tier cost **2.7×
  more on prefill than on decode** (`L` = +27.1% over the frontier vs +10.0%) purely because
  llama's two metrics are 7.8× apart — 18.44 decode against 143.88 prefill. Nothing about
  prefill work is 2.7× harder; that number came from llama.cpp's shape, not from ours.

  `pct_of_llama` is still recorded on every run, and `reference.lock` still pins 143.88 — it
  stopped being the tier basis, not the target. **This flips back the round after batched
  prefill (#137) lands**, because at that point the two engines are doing the same thing and
  the anchor means what it says again.
- **Decode and prefill share kernels.** Batching the prompt will move decode. The 1% guard
  bounds how far, and it is a refusal rather than a tier — a prefill gain bought by giving
  decode back has not moved the engine forward, it has moved work around.

### The decode guard is measured at 128k

Everything about decode is measured at **ctx 131,072**, not at short context. Until 2026-08-01 it
was not: `kimi_k3_tp_bench` hardcoded `max_ctx=64`, so the `reference.lock` slot said 128,
this guide said 128k, and the hardware saw 64. Every tier awarded before that came from the
last of those three.

It matters because the two engines diverge almost entirely at length — 8× H200, UD-IQ1_S,
same weights:

| | ctx 64 | ctx 131,072 | lost |
|---|--:|--:|--:|
| llama.cpp | 18.32 | 18.44 | ~0% |
| sparkinfer, as first measured | 10.34 | 1.00 | **−90%** |

llama.cpp keeps a compressed MLA cache (`kv_lora` 512, f16) and holds its rate essentially
flat with depth; sparkinfer reduced over 576 f32 per token in a kernel with one block per
head. Scoring at 64 hid an **18×** gap behind a 1.8× one — and pointed the incentive at a
context nobody runs.

**That headroom has largely been taken.** The frontier is now **29.97 tok/s** — past
llama.cpp rather than behind it — so with the basis at 18.4435 the bands are absolute
tok/s, and they no longer look brutal:

| tier | gain over the frontier |
|---|--:|
| `XS` | +0.60 tok/s (the 2% significance gate, which now binds first) |
| `S` | +0.65 (+2.2%) |
| `M` | +1.11 (+3.7%) |
| `L` | +1.84 (+6.2%) |
| `XL` | +3.32 (+11.1%) |

Note the gate, not the tier, is what a small PR has to clear first: 2% of a 29.97 frontier
is 0.60 tok/s, which is above the `S` and comparable to the `M` threshold. A real but small
win now scores `none` — see #71, twice.

sparkinfer at 128k now sits at **163% of llama.cpp**, and the remaining headroom is against
the HBM bandwidth roofline rather than against llama.cpp — so an `XL` is still asking for
work that is on the table, but the low-hanging fruit is gone. Anchoring to llama.cpp is what stops
an immature frontier minting `XL`s from low-hanging fruit, and means the same tok/s of real
work earns the same tier however fast the frontier already is. The verdict JSON reports both
— `pct_over_frontier` is the honest measured speedup, `pct_of_llama` is the tier basis, and
`scored_context` records which context earned it.

**How 128k is measured, precisely.** K3 has no prefill path, so genuinely filling 131,072
positions is ~10 h of sequential `forward_token` calls. The harness instead allocates the KV
cache at full size, leaves it **zeroed**, and seeks position (`--ctx` / `--seek`). Decode
cost is data-independent — the MLA reduction is dense whether the entries are activations or
zeros — so the **timing** is faithful. The **correctness** gate is unaffected by this because
it runs separately, without those flags, against the llama.cpp capture in `bench/refdata/`.
A PR that stubs `--seek` would score ctx-64 decode as 128k (~10×), so the harness refuses
any run where the bench did not announce the seek it performed.

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
A tier is only ever applied from a real measured run: the numbers come from the node, and
`eval-label.yml` recomputes the tier from them against the protected `reference.lock`, so a
posted label cannot set a payout.

**Maintainers do have an override on merging.** `eval/k3_eval_bot.py --merge-admin` merges
the round's `merge-first` winner with admin rights, bypassing the approving-review
requirement on `main`. It refuses to merge a PR carrying `hold`, `copycat`, `penalty`,
`needs-rebase` or similar, one with no `eval:*` tier, one touching maintainer-owned paths,
or one whose merge state is not demonstrably clean — but within those bounds it lands a
change with no human reading it. Said plainly because a contributor is entitled to know
what can merge their work, and what cannot.

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

**Enforcement.** `sensitive-paths-guard` is a **required status check on `main`**: it fails
any PR from a non-maintainer touching these paths, and a failing required check means the PR
**cannot merge**, regardless of content. It runs on `pull_request_target`, so it runs on fork
PRs too and cannot be disabled by editing the workflow in your own branch. `CODEOWNERS`
additionally **requests** maintainer review on those paths — that part is advisory; the status
check is the hard gate. The evaluator also grades with the harness pinned to the protected
branch, so editing it in a PR never affects that PR's own score.

### What `main` requires

Six checks must pass: `sensitive-paths-guard`, `shell syntax`, `python compile + unit tests`,
`configs + docs`, `harness pins`, and `reference.lock provenance`. Branches
must also be **up to date with `main`** before merging — your gain is scored as marginal over
the current frontier, so it has to be measured against current `main`.

Two things to expect if you contribute **from a fork**:

- The five `ci.yml` checks show as pending until a maintainer clicks **Approve and run
  workflows**. This is deliberate: `python compile + unit tests` executes test files from your
  branch, and GitHub gates running outside code. `sensitive-paths-guard` runs immediately
  without approval.
- When `main` moves, rebase and push; the required checks re-run.

**Improving the harness is welcome — just not via a direct PR.** Open an issue or discussion
describing the change; if a maintainer agrees, they'll land it (with credit). Keep your own PRs
scoped to `kernels/`, `runtime/`, and `moe/` — that's the rewarded optimization work.

## Style & scope

- Match the surrounding code (portable CUDA is the production path; CuTe/tensor-core is
  the opt-in ceiling). Keep kernels readable and commented where non-obvious.
- Reference the bench + accuracy numbers in your PR description (before → after).
- Keep changes focused; one optimization per PR makes the measured delta attributable.

By contributing you agree your work is licensed under the repository's [MIT License](LICENSE).
