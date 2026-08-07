# SparkInfer-K3 — technical reference

Everything the [README](../README.md) links to: how the runtime is built, how the model
is sharded, how the baseline was measured, and how a PR gets scored.

---

## It works — a real run, on a real node

```
$ bench/scripts/kimi_k3_run.sh "Q: What is the capital of Japan? A:" 24 0,1,2,3,4,5,6,7

prompt : Q: What is the capital of Japan? A:
ids    : 48 25 5071 387 276 10484 318 10417 30 401 25
devices: 0,1,2,3,4,5,6,7   predict: 24

model: 93/93 layers, vocab 163840, 8 stage(s)
decode: 23 forward passes in 7.459 s = 3.08 tok/s

generated text:  Tokyo. Q: What is the capital of France? A: Paris. Q: What is the capital of Germany?
```

Correct answers, and it picks up the few-shot pattern on its own. No chat template, no
sampling parameters — greedy argmax over raw completions.

### Are all 8 GPUs actually working? Yes, and here is the proof

Not "we launched with 8 devices" — three independent signals from one run:

**1. Each rank owns a disjoint band of the 896 experts.** Printed at init:

```
[k3-tp] rank 0: device 0, experts [  0,112)     rank 4: device 4, experts [448,560)
[k3-tp] rank 1: device 1, experts [112,224)     rank 5: device 5, experts [560,672)
[k3-tp] rank 2: device 2, experts [224,336)     rank 6: device 6, experts [672,784)
[k3-tp] rank 3: device 3, experts [336,448)     rank 7: device 7, experts [784,896)
```

`kimi_k3_tp_load_check` proves those bands **tile exactly**: 16351 checks, every byte of
a sharded tensor claimed by exactly one rank. A gap would silently drop an expert; an
overlap would double-count it. Both leave a model that runs and emits fluent text.

**2. All eight cards fill with weights.** `nvidia-smi` sampled through a load, starting
from 0 MiB on every card:

```
02:49:43   0:73567MiB   1:4MiB      2:4MiB      3:4MiB      ...     <- rank 0 loading
02:50:28   0:125999MiB  1:125999MiB 2:125999MiB 3:8743MiB   ...     <- ranks 1-3 filling
02:51:59   0:125999MiB  1:125999MiB ... 6:125999MiB 7:118653MiB     <- all 8 resident
```

~123 GiB per card. The model is 553 GiB and no single H200 holds 141.

**3. The collective count is exact.** `collectives/token 92` — one all-reduce per MoE
layer, 93 layers minus the leading dense block. A missing reduce leaves a partial expert
sum; an extra one multiplies a complete tensor by `tp_size`. Neither crashes, so the
count is asserted rather than eyeballed.

---

## How the TP module works

The design decision that makes it customizable: **the layer is split at exactly the
points a collective belongs, and the driver owns the interleaving.**

```
  for each layer:
      ranks 0..7  ──►  Attn         (attention, all ranks in parallel)
      ranks 0..7  ──►  FfnPartial   (router → routed_down → THIS RANK'S experts)
                              │
                       ALL-REDUCE   ← 3584-wide, sums the 8 partial expert sums
                              │
      ranks 0..7  ──►  FfnFinish    (routed_norm → routed_up → shared experts)
```

Every rank runs the same router over the same replicated weights, so all eight agree on
the same top-16 experts — then each evaluates only the ones it holds and contributes
zero for the rest. The all-reduce turns eight partial sums back into exactly what one
GPU with all 896 experts would have computed. Verified to **1.127e-07** of peak — below
float32 epsilon, i.e. one ulp.

**Why a phase split and not a callback.** NCCL needs every rank's call enqueued inside
one group before any launches, and sparkinfer drives all ranks from a single host
thread. A `reduce()` callback fired mid-forward deadlocks — measured on this node, hung
at the first collective with no output. The driver has to own the interleaving.

**Why the reduce is at 3584 and not 7168.** `ffn_routed_norm` is an RMS norm, and
rms_norm is not linear: `rms_norm(Σ partial) ≠ Σ rms_norm(partial)`. The sum has to
complete *before* it. Getting this wrong is invisible — the model stays fluent.

### Where to customize

Each of these is a seam, not a rewrite:

| knob | what it changes | where |
|---|---|---|
| `ShardPolicy` | `ExpertsOnly` (experts banded, attention replicated) vs `Full` | [`kimi_k3.h`](../runtime/include/sparkinfer/models/kimi_k3.h) |
| `SPARKINFER_TP_BACKEND` | `nccl` \| `peer` \| `multimem` — all three validated on 8× H200 | [`collective.h`](../runtime/include/sparkinfer/tp/collective.h) |
| shard rules | per-tensor Row / Col / Expert / Replicate, 230 CPU tests | [`weight_plan.cpp`](../runtime/src/tp/weight_plan.cpp) |
| expert bands | contiguous today; strided would trade load balance for locality | [`shard.cpp`](../runtime/src/tp/shard.cpp) |
| reduce points | `K3LayerPhase` — move a collective by moving one call | [`kimi_k3_tp.cpp`](../runtime/src/models/kimi_k3_tp.cpp) |
| dtype | f32 today (K3's residual stream is f32 by design); bf16 path exists | [`collective.h`](../runtime/include/sparkinfer/tp/collective.h) |

The shard math is **CUDA-free and unit-tested without a GPU** — 4972 checks on
`shard.cpp`, 230 on `weight_plan.cpp`, 44 on backend selection. TP bugs do not live in
the collective (NCCL is correct); they live in the band arithmetic, which is exactly the
part you can verify on a laptop before burning node hours.

### The honest limit of the current policy

`ExpertsOnly` shards the 896 experts (531 of 553 GiB) and **replicates attention**. So
attention runs redundantly on all eight cards. After the MoE dispatch got 3× faster,
that replicated attention became the entire serial term and TP scaling fell from 2.44×
to ~1.1×. `ShardPolicy::Full` is declared and deliberately **not** enabled — the loader
would shard weights the executor still indexes at full width. That is the next lever.

---

## The model

| | |
|---|---|
| Params | 2.8T total · typed `2.8T.A50B` by the reference implementation |
| Layers | 93 — **24 MLA** (full attention) + **69 KDA** (linear, recurrent) |
| Hidden / vocab | 7168 / 163840 |
| Context | 1,048,576 |
| MoE | **896 routed experts**, top-16, 2 shared · latent MoE at **3584**, expert FFN 3072 |
| Attention (full) | MLA — `q_lora 1536`, `kv_lora 512`, **NoPE-only**, sigmoid output gate |
| Attention (linear) | KDA — 96 heads × 128, conv kernel 4, full-rank gate, `gate_lower_bound −5.0` |
| Activation | `situ` replaces SwiGLU everywhere (`β 4.0`, `linear β 25.0`) |
| Extras | Cross-layer residual attention, `block_size 12` |
| Vision | MoonViT-3d — 27 layers, 1024 wide, **non-square fused QKV** (1536 ≠ n_embd), patch 14 |

Three traps that produce silently wrong output rather than an error — all encoded in [`bench/configs/models/kimi_k3.yaml`](../bench/configs/models/kimi_k3.yaml):

- **`full_attn_layers` is 1-indexed.** The converter tests `(il + 1) in full_attn_layers`. Off by one and you get garbage, not a crash.
- **MLA is stored as MQA.** `head_count_kv = 1`, `key_length = kv_lora + qk_rope = 576`; per-layer `head_count_kv == 0` is what marks a KDA layer.
- **Routed experts live in a down-projected space.** `expert_latent_length 3584`, not `hidden_size 7168`. Size expert GEMMs off `hidden_size` and you're wrong by 2×.

---

## Why the baseline is a fork

Every other model in the SparkInfer family is benchmarked against `ggml-org/llama.cpp` at a pinned commit. Kimi K3 cannot be.

**Upstream llama.cpp cannot load this model at all.** It asserts `n_expert <= LLAMA_MAX_EXPERTS`, and upstream's cap is 512. K3 has 896. There is no upstream number to compare against, so the reference is [`unslothai/llama.cpp`](https://github.com/unslothai/llama.cpp) PR #48, pinned in [`bench/scripts/reference.lock`](../bench/scripts/reference.lock).

Four things in that fork are load-bearing, not cosmetic:

1. `LLAMA_MAX_EXPERTS 1024` — without it, the model asserts at load.
2. The `LLM_ARCH_KIMI_K3` graph — hybrid KDA + MLA, latent MoE, `situ`, cross-layer attention residual, MLA output gate, full-rank KDA gate.
3. `graph_max_nodes = max(n_tokens × 160, 64 × n_tensors)` — the generic `×40` budget shared by other hybrid archs is exhausted at ubatch 3840.
4. Four **required-not-defaulted** KV keys (`expert_latent_length`, `attn_res.block_size`, both `situ` betas). Silently defaulting them loads cleanly and emits garbage — exactly what a baseline must refuse to do.

Because the pin is a PR head on a fork, GitHub refuses fetch-by-sha. The harness fetches the ref and then **asserts** it resolves to the pinned commit, so a force-push **fails the run** instead of quietly moving the baseline.

---

## The target · 8× H200, UD-IQ1_S

**Hopper is the default target, not a stepping stone to Blackwell.** H200 is the
fastest silicon that is actually rentable at scale today, and by a wide margin the
cheapest per GB of HBM — 141 GB a card, eight cards, 1123 GiB, on hardware you can get
this afternoon. A runtime that only pays off on B200/B300 is a runtime almost nobody
can run. So the default node profile is `h200x8` and the default quant is `UD-IQ1_S`.

**UD-IQ1_S is the default quant; UD-Q2_K_XL is still the accuracy target.** Those are
different claims and both matter:

| | UD-IQ1_S (default) | UD-Q2_K_XL (target) |
|---|---|---|
| size | 553 GiB | 802 GiB |
| shards | 14 | 19 |
| top-1 vs lossless | 78.9% | 90.4% |
| on 8× H200 | ✅ fits, **measured** | ✅ fits, not yet run |

UD-Q2_K_XL is the accuracy knee and remains what the project is ultimately judged on.
But the *default* is what runs when nobody passes a flag, and defaulting to an 802 GiB
download that is on no machine means every fresh invocation dies before it does
anything. UD-IQ1_S is what is resident, what the llama.cpp reference was measured on,
and what [`bench/refdata/`](../bench/refdata)'s reference logits were captured against.
`PRIMARY_QUANT=UD-Q2_K_XL` switches to the target.

The reference.lock slots carry the quant in their **name**
(`KIMI_K3_H200X8_IQ1S_LLAMA_128`) so a number measured under one can never be read as
the other — IQ1_S decodes faster, and pinning it into an unqualified slot would
understate every future gain forever.

| quant | GiB | 8× H200 | 8× B200 | 4× B300 | top-1 |
|---|---:|:-:|:-:|:-:|---:|
| **UD-IQ1_S** | **553** | ✅ | ✅ | ✅ | 78.9% |
| UD-IQ2_XXS | 662 | ✅ | ✅ | ✅ | 84.1% |
| **UD-Q2_K_XL** | **802** | ✅ | ✅ | ✅ | **90.4%** |
| UD-Q4_K_XL | 1407 | ❌ | ❌ | 8× only | — |

All weights in HBM is a hard requirement — a partially-offloaded baseline is not
reproducible, so `kimi_k3_check_fits` **refuses** to run one, and it prices the KV
cache in rather than assuming flat headroom.

### Node profiles

| node | per-GPU | total | arch | status |
|---|---:|---:|---|---|
| **`h200x8`** *(default)* | 141 GB | 1128 GiB | `sm_90` Hopper | **measured** |
| `b200x8` | 180 GB | 1440 GiB | `sm_100` Blackwell | untested — and no longer compile-gated in CI |
| `b300x4` | 288 GB | 1152 GiB | `sm_103`* Ultra | untested |

\* `sm_103` is unverified. `detect_arch` reads `compute_cap` at run time and wins, so a
wrong label mis-files a result but does not mis-build. It is deliberately **not** in the
CMake arch list — an arch the installed toolkit does not recognise breaks configure for
every other node.

---

## Evaluation — llama.cpp is the baseline

Correctness is **agreement with llama.cpp on identical weights**, and speed is scored
against it too. One command produces both plus the tier:

```bash
bench/scripts/kimi_k3_eval.sh --node h200x8 --frontier <merged best>
```

It emits the `RESULT_JSON` contract [`bench/scripts/label.py`](../bench/scripts/label.py)
already scores for the other models, so K3 needs no second scoring path:

- **Correctness gate first.** top-1 ≥ 0.95 and KL ≤ 0.05 against the captured reference,
  else `REJECT` — regardless of speed. A speedup that erodes parity is not a speedup.
  Top-1 is effectively pass/fail (one logit row per probe, so it is a boolean); KL is the
  graded gate.
- **Significance gate.** The gain must beat 2% of the frontier, else `none`.
- **Tier is the worse of two bases** — `min(delta/llama_ref, delta/frontier)` — so it can
  never exceed the speedup actually measured. Below llama.cpp the anchor binds, so an
  un-optimized model cannot mint `XL`s from low-hanging fruit; past llama.cpp (K3 is at
  2.2×) the frontier binds, so `XL` costs a real 18% over main.

A node run posts its verdict to a PR with `/eval RESULT_JSON {...}`;
[`.github/workflows/eval-label.yml`](../.github/workflows/eval-label.yml) **re-derives** the
tier from the reported measurements rather than trusting the reported label, and honours
the command only from maintainers.

---

## Layout & scoring

| Path | What |
|---|---|
| [`bench/`](../bench) | **the baseline** — K3 harness, arch/target configs, eval + accuracy scripts |
| [`docs/`](../docs) | [`kimi-k3-baseline.md`](kimi-k3-baseline.md) — how to run it, and every trap in the arch |
| [`kernels/`](../kernels) | CUDA kernels — flash-decode, decode GEMV, fused MoE FFN, GEMM, RMSNorm, RoPE, GGUF dequant |
| [`runtime/`](../runtime) | scheduler, paged KV cache, CUDA-graph decode, native GGUF loading, model forward |
| [`moe/`](../moe) | sync-free MoE router + expert dispatch |
| [`server/`](../server) | OpenAI-compatible HTTP API (`BUILD_SERVER=ON`) |

`kernels/` and `runtime/` now carry a native K3 path (KDA + MLA decode, latent MoE, `situ`, cross-layer residual, expert-parallel dispatch). `moe/` and `server/` are still Qwen-shaped.

**Scoring is speedup-only.** SN74 pays verified marginal speedups labeled **XL / L / M / S / XS**. Sub-2% gains are never aggregated across contexts. See [`.gittensor/weights.json`](../.gittensor/weights.json).

---

## Build

Requires **CUDA Toolkit 12.8+**. Pick the arch for your milestone node:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90    # M1  H200   Hopper
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=100   # M2  B200   Blackwell datacenter
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=103   # M3  B300   Blackwell Ultra — see caveat
cmake --build build -j
ctest --test-dir build
```

The default arch list in `CMakeLists.txt` is `89;90;100;120;121`. **103 is not in it**, and adding it requires a toolkit that recognises the target — otherwise configure fails for every node, not just B300. Confirm with `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` before you add it.

The baseline scripts build the pinned llama.cpp reference themselves into `.llamacpp-k3/` — a separate checkout from the upstream pin, so the two references never fight over one tree.

---

## CI

CI is built around one honest limit: **no hosted runner will ever run an 802 GiB model.** So it gates what is cheap here and expensive to discover on a rented node, and it does not pretend to gate anything else.

| job | what it proves |
|---|---|
| `shell` | `bash -n` over **every** tracked script + no CRLF in source |
| `python` | `py_compile` over every tracked file + every discovered `test_*.py` (76 K3 tests) |
| `configs` | all YAML parses; the derived arch arithmetic is self-consistent; relative links resolve |
| `plans` | `--dry-run` resolves for all three nodes; the fork pin does **not** leak into the upstream harness |
| `lock` | **every pinned baseline traces to a committed `bench/results/*.json`** |

That last job is the one that matters. The repo's whole claim is that a native runtime gets scored against a reference someone else can reproduce — which collapses the moment a number can be typed in, because downstream a hand-filled baseline is indistinguishable from a measured one. `check_reference_lock.py` makes that unfakeable: `0` means not measured and is always fine, but any non-zero value must match a recorded sweep for the same node *and* context.

Two more, both path-filtered so harness PRs don't pay for them:

- **`build-gate`** — compiles for `sm_90` and `sm_100` on changes to `kernels/`, `moe/`, `runtime/`, `server/`, `CMakeLists.txt`. (`sm_103` is deliberately absent; see the arch caveat above.)
- **`pin-audit`** — weekly, plus on pin changes. The baseline is a PR head on someone else's fork and the weights live in someone else's HF repo; neither is immutable. Catches a force-push or a re-uploaded quant *before* you're paying for a node.

`node-attestation` labels PRs that change perf-bearing code with no node run attested. It **labels only — it never closes a PR**, unlike the RTX-5090 gate it replaces.

---

## Automated evaluation

Correctness for K3 is **agreement with the reference on identical weights** — nothing else. There is no second independent K3 implementation, and even the target quant's own top-1 against full precision is 90.4%, so absolute quality numbers are meaningless. (M3's stretch goal — UD-Q8_K_XL on 8× B300 — would give this repo its first *in-house* lossless reference, instead of citing published figures.)

Two reference-server flags are mandatory, not tuning:

- `--no-context-shift` — K3 is a hybrid recurrent arch; llama.cpp cannot context-shift or restore slots for it, and a long eval dies mid-run without it.
- `--no-jinja` — the gate posts raw token ids; a chat template would prepend tokens the candidate never saw.

Details: [`eval/`](../eval) · **[EVAL-TRUST.md](../EVAL-TRUST.md)** (Polaris TDX receipts, reproducible from source today).
