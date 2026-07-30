![sparkinfer banner](docs/sparkinfer.png)

# SP⚡RKINFER-K3 · Powered by SN74

**Kimi K3 inference. 2.8T parameters, one node.**

Kimi K3 is the largest open-weight model anyone can actually run: **2.8T total parameters**, hybrid KDA + MLA attention, **896 routed experts**, 1M context, native vision. SparkInfer-K3 is the runtime track for that model class — where the bottleneck stops being one card's bandwidth and becomes expert residency across eight. Continuously optimized by competition at **[SN74 on Gittensor](https://gittensor.io/miners/repository?name=gittensor-ai-lab%2Fsparkinfer)** and **Kernel Design Agents**.

> **Fewer models. Deeper optimization. Faster evolution.**

## Status — read this first

This repo currently contains **the measuring stick, not the runtime.**

| | |
|---|---|
| ✅ **Evaluation baseline** | Pinned llama.cpp reference, sharded-weight verification, accuracy reference server, 35 tests |
| ✅ **Architecture spec** | Every hparam traced to `config.json` or the reference implementation's source |
| ❌ **Native K3 runtime** | No `kimi-k3` GGUF loader. `runtime/` loaders are Qwen-shaped |
| ❌ **sm_90 kernel path** | `kernels/` targets Blackwell `sm_120`/`sm_121`; H200 is Hopper |
| ❌ **Measured numbers** | The sweep has not been run on hardware. Every reference slot ships `0`/`null` |

Nothing here claims a speedup, because nothing here has been measured yet. That is deliberate: a baseline you can't reproduce is worse than no baseline. Full detail in [`docs/kimi-k3-baseline.md`](docs/kimi-k3-baseline.md).

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

Three traps that produce silently wrong output rather than an error — all encoded in [`bench/configs/models/kimi_k3.yaml`](bench/configs/models/kimi_k3.yaml):

- **`full_attn_layers` is 1-indexed.** The converter tests `(il + 1) in full_attn_layers`. Off by one and you get garbage, not a crash.
- **MLA is stored as MQA.** `head_count_kv = 1`, `key_length = kv_lora + qk_rope = 576`; per-layer `head_count_kv == 0` is what marks a KDA layer.
- **Routed experts live in a down-projected space.** `expert_latent_length 3584`, not `hidden_size 7168`. Size expert GEMMs off `hidden_size` and you're wrong by 2×.

## Why the baseline is a fork

Every other model in the SparkInfer family is benchmarked against `ggml-org/llama.cpp` at a pinned commit. Kimi K3 cannot be.

**Upstream llama.cpp cannot load this model at all.** It asserts `n_expert <= LLAMA_MAX_EXPERTS`, and upstream's cap is 512. K3 has 896. There is no upstream number to compare against, so the reference is [`unslothai/llama.cpp`](https://github.com/unslothai/llama.cpp) PR #48, pinned in [`bench/scripts/reference.lock`](bench/scripts/reference.lock).

Four things in that fork are load-bearing, not cosmetic:

1. `LLAMA_MAX_EXPERTS 1024` — without it, the model asserts at load.
2. The `LLM_ARCH_KIMI_K3` graph — hybrid KDA + MLA, latent MoE, `situ`, cross-layer attention residual, MLA output gate, full-rank KDA gate.
3. `graph_max_nodes = max(n_tokens × 160, 64 × n_tensors)` — the generic `×40` budget shared by other hybrid archs is exhausted at ubatch 3840.
4. Four **required-not-defaulted** KV keys (`expert_latent_length`, `attn_res.block_size`, both `situ` betas). Silently defaulting them loads cleanly and emits garbage — exactly what a baseline must refuse to do.

Because the pin is a PR head on a fork, GitHub refuses fetch-by-sha. The harness fetches the ref and then **asserts** it resolves to the pinned commit, so a force-push **fails the run** instead of quietly moving the baseline.

## Hardware · 8× H200 SXM

The smallest useful K3 quant is **553 GiB**. That fits no Blackwell edge card, and no eight of them. 8× H200 SXM (141 GB each, 1051 GiB total) is the smallest node that holds the weights entirely in HBM — a hard requirement, since a partially-offloaded baseline isn't reproducible.

| quant | GiB | fits 8× H200 | top-1 vs lossless |
|---|---:|---|---:|
| **UD-IQ1_S** | **553** | ✅ default | 78.9% |
| UD-IQ1_M | 604 | ✅ | 81.2% |
| UD-IQ2_XXS | 662 | ✅ | 84.1% |
| **UD-Q2_K_XL** | **802** | ✅ accuracy knee | 90.4% |
| UD-Q8_K_XL | 1453 | ❌ | lossless |

Two caveats stated plainly, because they cap the ceiling:

- **H200 is Hopper (`sm_90`), not Blackwell.** The CMake arch list includes 90, so a source build compiles — but no sm_90 kernel path is tuned. Treat any future SparkInfer number on this node as untuned until that work lands.
- **Hopper has no FP4 tensor core.** K3's routed experts ship as MXFP4 and the useful GGUFs are 1–2 bit, so every low-bit expert GEMM dequantizes before the MMA. llama.cpp takes the same path, so the comparison is *fair* — the ceiling is simply lower than on Blackwell.

### Memory budget · UD-IQ1_S

Derived from the reference implementation's own sizing functions, not estimated:

| item | size |
|---|---:|
| weights | 553 GiB |
| MLA KV @ 32k | 0.84 GiB |
| MLA KV @ 128k | 3.38 GiB |
| MLA KV @ **1M** | **27.0 GiB** |
| KDA recurrent state | 0.43 GiB / sequence |
| headroom @ 1M | ~470 GiB |

**KV is not the constraint.** 24 MLA layers × 576 × 2 B = 27,648 B/token, K-only (MLA allocates no V cache), so the entire 1M-token budget is 27 GiB. The pressure on this model is expert residency and compute buffers — which is where a native runtime has to win.

## Quickstart

On the 8× H200 node — the scripts detect arch, build the pinned reference for `sm_90`, fetch the 14 shards, and refuse to run if the weights won't fit in HBM:

```bash
export KIMI_K3_MODELS_DIR=/workspace/models_k3     # needs ~600 GiB free
export PRIMARY_QUANT=UD-IQ1_S                      # or UD-Q2_K_XL for the accuracy knee

# plan the whole run — no network, no GPU, no weights
bench/scripts/kimi_k3_baseline.sh --dry-run

# fetch + build + sweep decode/prefill at 128 / 512 / 4k / 32k
bench/scripts/kimi_k3_baseline.sh --download

# accuracy reference for accuracy_compare.py
bench/scripts/kimi_k3_reference_server.sh --ctx 8704
```

Results land in `bench/results/`. Tests need no GPU: `python3 bench/scripts/test_kimi_k3_baseline.py`.

K3 ships a **tiktoken vocab, not `tokenizer.json`**, so eval prompt ids come from the GGUF itself via `gen_eval_prompt.py --gguf` — strictly more correct than a side-loaded tokenizer, and cheap (`vocab_only`, metadata not weights).

## Powered by SN74 — moving at the speed of ⚡

Contributors submit PRs; the bot verifies correctness and speed on real RTX 5090 hardware; SN74 rewards verified marginal speedups. **15 releases in 3 weeks** — from first llama.cpp beat to **+86% decode / +127% prefill @ 32k on Qwen3.6 SOTA**.

1. Pick a narrow bottleneck in the Blackwell decode path.
2. Submit a PR with source changes and benchmark evidence.
3. The bot builds `main` and the PR on the same RTX 5090.
4. Correctness vs llama.cpp; guards at 128 / 512 / 4k / 16k / 32k decode.
5. Strongest context improvement scores; regressions get `regression-*` labels.
6. Frontier merges; the [dashboard](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/) updates.

Miner workflow: [`docs/miner-guide.md`](docs/miner-guide.md).

## Roadmap

### Milestone 1 · Now — a reference nobody can dispute

- Pinned fork baseline, sharded-weight verification, fits-in-HBM refusal — **landed**
- Run the sweep on 8× H200 and pin the measured decode/prefill numbers — **pending hardware**
- Vision baseline via `llama-mtmd-cli` + `mmproj` (wired up, no image benchmark yet)

### Milestone 2 · Next — load it natively

- `kimi-k3` GGUF loader: 93 layers, per-layer KDA/MLA layout, 896 experts
- KDA linear-attention decode — full-rank gate, conv state, f32 recurrent state
- MLA NoPE decode with output gate · latent MoE expert dispatch · `situ` · attn residual
- Multi-GPU weight sharding — 553 GiB fits no single card at any quant

### Milestone 3 · Then — beat the reference

- `sm_90` kernel path, then the same correctness-first eval loop the Qwen frontier uses
- Expert residency and compute-buffer pressure, not KV, is the target

## Layout & scoring

| Path | What |
|---|---|
| [`bench/`](bench) | **the baseline** — K3 harness, arch/target configs, eval + accuracy scripts |
| [`docs/`](docs) | [`kimi-k3-baseline.md`](docs/kimi-k3-baseline.md) — how to run it, and every trap in the arch |
| [`kernels/`](kernels) | CUDA kernels — flash-decode, decode GEMV, fused MoE FFN, GEMM, RMSNorm, RoPE, GGUF dequant |
| [`runtime/`](runtime) | scheduler, paged KV cache, CUDA-graph decode, native GGUF loading, model forward |
| [`moe/`](moe) | sync-free MoE router + expert dispatch |
| [`server/`](server) | OpenAI-compatible HTTP API (`BUILD_SERVER=ON`) |

`kernels/`, `runtime/`, `moe/` and `server/` are inherited from SparkInfer and are **Qwen-shaped today** — they are the surface Milestone 2 lands on, not K3 support.

**Scoring is speedup-only.** SN74 pays verified marginal speedups labeled **XL / L / M / S / XS**. Sub-2% gains are never aggregated across contexts. See [`.gittensor/weights.json`](.gittensor/weights.json).

## Build

Requires **CUDA Toolkit 12.8+**. H200 is `sm_90`:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
ctest --test-dir build
```

The baseline scripts build the pinned llama.cpp reference themselves into `.llamacpp-k3/` — a separate checkout from the upstream pin, so the two references never fight over one tree.

## Automated evaluation

Correctness for K3 is **agreement with the reference on identical weights** — nothing else. There is no second independent K3 implementation, and UD-IQ1_S's own top-1 against full precision is 78.9%, so absolute quality numbers are meaningless.

Two reference-server flags are mandatory, not tuning:

- `--no-context-shift` — K3 is a hybrid recurrent arch; llama.cpp cannot context-shift or restore slots for it, and a long eval dies mid-run without it.
- `--no-jinja` — the gate posts raw token ids; a chat template would prepend tokens the candidate never saw.

Details: [`eval/`](eval) · **[EVAL-TRUST.md](EVAL-TRUST.md)** (Polaris TDX receipts, reproducible from source today).

## Contributing

Source-required and reproducible. Before a PR: `bench/scripts/kimi_k3_baseline.sh --dry-run` and `python3 bench/scripts/test_kimi_k3_baseline.py`. Never hand-fill a `reference.lock` baseline — downstream, a hand-filled number is indistinguishable from a measured one. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) · [Changelog](CHANGELOG.md)
