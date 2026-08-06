![SparkInfer K3 — Kimi K3 inference on NVIDIA H200, 2.8T parameters, 896 experts, MLA + KDA](docs/k3.png)

# SP⚡RKINFER-K3 · Powered by SN74

### The largest open-weight model in the world, running on one node you can rent today.

**Kimi K3** is 2.8 trillion parameters — hybrid KDA + MLA attention, 896 routed experts,
native vision, 1M context. It is the frontier of open weights.

**SparkInfer-K3 runs it on a single 8× H200 box.** Text in, text out, all 93 layers, 128k
context, at speeds that beat the only other engine that will load it.

```
$ bench/scripts/kimi_k3_run.sh "def fibonacci(n):" 48 0,1,2,3,4,5,6,7

    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```

---

## The numbers

**8× H200 · UD-IQ1_S · 553 GiB of weights · same box, same weights, both engines measured there.**

| | llama.cpp | **sparkinfer-k3** | |
|---|--:|--:|--:|
| **decode @ 128k context** | 18.44 tok/s | **56.82 tok/s** | **3.08× faster** |
| **prefill @ 32k context** | 143.88 tok/s | **69.02 tok/s** | 2.08× behind |

Decode is where a long conversation actually lives, and it is the metric the project is
scored on. K3 started **18× behind** llama.cpp there on 2026-08-01. It is now 3.08× ahead.

Prefill was one forward per token until 2026-08-06. Batching the prompt took the pinned
frontier from **40.35 → 69.02 tok/s** in two days, and the batched driver that merged after
that round measures **98.80** on the node — real, reproducible, and not yet sealed by a
round, so the pin above stays at the last attested value. llama.cpp still leads here and
that gap is the current work.

Every number above is measured on a rented 8× H200, pinned in
[`bench/scripts/reference.lock`](bench/scripts/reference.lock), and backed by a sealed
receipt in the [append-only log](https://github.com/gittensor-ai-lab/sparkinfer-k3-log) —
reproducible without trusting us.

---

## Why this exists

A 2.8T model does not fit the assumptions the mainstream serving stacks are built on. K3's
architecture is new — a KDA/MLA hybrid with a latent MoE — and running it at all means a
loader, attention kernels, and an expert-parallel shard policy written for *this* model.
Today the practical options are llama.cpp and this.

That matters if you are:

**Running models locally.** One node, no API, no per-token bill, no data leaving the
building. The largest open model there is, on hardware you can actually rent.

**An enterprise with data you cannot send anywhere.** Legal, medical, defence, finance —
where "we don't send it to a third party" is the requirement, not a preference.

**Building private AI products.** Frontier-class weights you control, with an engine you
can read, patch, and benchmark yourself.

**Working on inference itself.** Every kernel, shard policy and scheduling decision is in
this repo, measured against a pinned reference, with the receipts published.

---

## Quickstart

```bash
export KIMI_K3_MODELS_DIR=/workspace/models_k3     # needs ~560 GiB free

# fetch the default quant (UD-IQ1_S, 553 GiB, 14 shards)
bench/scripts/kimi_k3_baseline.sh --download --decode-only

# GENERATE. text in, text out, 93 layers across 8 GPUs
export KIMI_K3_MODEL=$KIMI_K3_MODELS_DIR/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf
bench/scripts/kimi_k3_run.sh "The capital of France is" 32 0,1,2,3,4,5,6,7

# SCORE against llama.cpp — speed tier + KL/top-1 parity in one command
bench/scripts/kimi_k3_eval.sh --node h200x8

# plan anything with no GPU, no network, no weights
bench/scripts/kimi_k3_baseline.sh --node h200x8 --dry-run
```

`kimi_k3_run.sh` needs `llama-tokenize` for the text → ids step: K3 ships a **tiktoken
BPE**, not a `tokenizer.json`, so the GGUF's own vocab is the only encoder correct by
construction. The script finds it or tells you how to build it, rather than falling back
to a tokenizer that would silently produce different ids.

Tests need no GPU: `python3 bench/scripts/test_kimi_k3_baseline.py`.

---

## What works

| | |
|---|---|
| ✅ **Native K3 runtime** | `kimi-k3` GGUF loader, KDA + MLA decode, latent MoE, `situ`, cross-layer residual |
| ✅ **Multi-GPU** | Tensor-parallel (expert-parallel) **and** layer-split pipeline. Agree to 1.85e-09 |
| ✅ **End to end** | text → ids → 93 layers × 8 GPUs → text |
| ✅ **128k context** | the scored configuration, not a ceiling claim |
| ✅ **Batched prefill** | a chunk of prompt tokens through the kernels together, bit-identical to the per-token walk |
| ✅ **Verified accuracy** | top-1 and KL against llama.cpp on identical weights and ids, every round |
| ✅ **Sealed receipts** | every score bound to code, weights and build in an append-only log |
| ❌ **Vision** | `mmproj` wired up, no image benchmark yet |

---

## Documentation

| | |
|---|---|
| [**Technical reference**](docs/technical.md) | how the TP module works, the model, the shard policy, the baseline, scoring, build and CI |
| [**Contributing**](CONTRIBUTING.md) | what gets measured, what gets rejected, and how a PR is scored |
| [**Miner guide**](docs/miner-guide.md) | the SN74 workflow end to end |
| [**Baseline method**](docs/kimi-k3-baseline.md) | how the llama.cpp reference was measured |
| [**Tensor parallel**](docs/tensor-parallel.md) | shard policies and collectives |
| [**Changelog**](CHANGELOG.md) | every change, with the numbers |

---

## Powered by SN74 — moving at the speed of ⚡

Contributors submit PRs; the eval verifies correctness and speed **on an 8× H200 node**;
[SN74 on Gittensor](https://gittensor.io/miners/repository?name=gittensor-ai-lab%2Fsparkinfer)
rewards verified marginal speedups.

1. Pick a narrow bottleneck in the Hopper path.
2. Submit a PR with source changes and benchmark evidence.
3. The eval builds `main` and the PR on the same 8× H200 node.
4. Correctness vs llama.cpp (top-1 ≥ 0.95, KL ≤ 0.05) gates before any speed tier.
5. The strongest verified improvement scores; regressions are labelled and refused.
6. The frontier merges; the [dashboard](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/) updates.

> **Fewer models. Deeper optimization. Faster evolution.**

---

## What's next

**Shard attention.** `ShardPolicy::ExpertsOnly` shards the experts and replicates
attention, so TP scaling collapsed to ~1.1× once the MoE got 3× faster. Attention is now
the whole serial term — `ShardPolicy::Full` is the main lever.

**Close the prefill gap.** llama.cpp is 2.08× ahead at the pinned 32k frontier. Batching landed; its
efficiency is the open question.

**Beyond 128k.** The model supports 1M. The KV cache and the MLA split plan are what stand
between here and there.

**Blackwell.** A later delta on a runtime that already works — not the thing that makes it
work.

---

## Contributing

Source-required and reproducible. Before a PR: `bench/scripts/kimi_k3_baseline.sh
--dry-run` and `python3 bench/scripts/test_kimi_k3_baseline.py`. Never hand-fill a
`reference.lock` baseline — downstream, a hand-filled number is indistinguishable from a
measured one. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) · [Changelog](CHANGELOG.md)
