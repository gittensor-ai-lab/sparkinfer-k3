![sparkinfer banner](docs/sparkinfer.png)

# SP⚡RKINFER-K3 · Powered by SN74

**Kimi K3 inference. 2.8T parameters, 1M context, one node.**

Kimi K3 is the largest open-weight model anyone can actually run: **2.8T total parameters**, hybrid KDA + MLA attention, **896 routed experts**, 1M context, native vision. SparkInfer-K3 is the runtime track for that model class — where the bottleneck stops being one card's bandwidth and becomes expert residency across the node. Target: **UD-Q2_K_XL at the full 1M context**, on H200 → B200 → B300. Continuously optimized by competition at **[SN74 on Gittensor](https://gittensor.io/miners/repository?name=gittensor-ai-lab%2Fsparkinfer)** and **Kernel Design Agents**.

> **Fewer models. Deeper optimization. Faster evolution.**

## Status — read this first

This repo currently contains **the measuring stick, not the runtime.**

| | |
|---|---|
| ✅ **Evaluation baseline** | Pinned llama.cpp reference, 19-shard verification, accuracy reference server, 58 tests |
| ✅ **Architecture spec** | Every hparam traced to `config.json` or the reference implementation's source |
| ✅ **Node + fit model** | H200/B200/B300 profiles, context-aware fits-in-HBM refusal, per-node reference slots |
| ❌ **Native K3 runtime** | No `kimi-k3` GGUF loader. `runtime/` loaders are Qwen-shaped — M2 work |
| ❌ **Tuned kernel path** | `kernels/` targets `sm_120`/`sm_121`; the milestone nodes are `sm_90`/`sm_100`/`sm_103` |
| ❌ **Vision** | `llama-mtmd-cli` + `mmproj` wired up, no image benchmark — M4 work |
| ❌ **Measured numbers** | No sweep has run on hardware. Every reference slot ships `0`/`null` |

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

## The target · UD-Q2_K_XL at 1M context

One quant, one context, three nodes. **UD-Q2_K_XL (802 GiB)** is the target because 90.4% top-1 against the lossless reference is the accuracy knee — UD-IQ1_S is 249 GiB smaller but drops to 78.9%, which is too lossy to be the thing an optimization frontier is judged on.

| quant | GiB | shards | 8× H200 | 8× B200 | 4× B300 | top-1 vs lossless |
|---|---:|---:|:-:|:-:|:-:|---:|
| UD-IQ1_S | 553 | 14 | ✅ | ✅ | ✅ | 78.9% |
| UD-IQ1_M | 604 | 15 | ✅ | ✅ | ✅ | 81.2% |
| UD-IQ2_XXS | 662 | 16 | ✅ | ✅ | ✅ | 84.1% |
| **UD-Q2_K_XL** | **802** | **19** | ✅ | ✅ | ✅ | **90.4%** |
| UD-Q4_K_XL | 1407 | 32 | ❌ | ❌ | 8× only | — |
| UD-Q8_K_XL | 1453 | 34 | ❌ | ❌ | 8× only | lossless |

✅ = fits with the full **1,048,576**-token KV cache resident. All weights in HBM is a hard requirement — a partially-offloaded baseline isn't reproducible, so `kimi_k3_check_fits` **refuses** to run one.

| node | per-GPU | total | arch | headroom @ 1M |
|---|---:|---:|---|---:|
| **M1** 8× H200 SXM | 141 GB | 1128 GiB | `sm_90` Hopper | 274 GiB |
| **M2** 8× B200 SXM | 180 GB | 1440 GiB | `sm_100` Blackwell | 586 GiB |
| **M3** 4× B300 | 288 GB | 1152 GiB | `sm_103`* Ultra | 298 GiB |

\* **`sm_103` is unverified here.** `detect_arch` reads `compute_cap` at run time and wins, so a wrong label mis-files a result but doesn't mis-build. The CMake arch list is `89;90;100;120;121` — no 103 — and it is deliberately *not* added pre-emptively, because an arch the installed toolkit doesn't recognise breaks configure for every other node. Confirm on first contact, then add it.

### Memory budget · UD-Q2_K_XL

Derived from the reference implementation's own sizing functions, not estimated:

| item | size |
|---|---:|
| weights | 802 GiB |
| MLA KV @ 32k | 0.84 GiB |
| MLA KV @ 128k | 3.38 GiB |
| MLA KV @ **1M** | **27.0 GiB** |
| KDA recurrent state | 0.43 GiB / sequence |
| compute buffers (allowance) | 24 GiB |
| **total @ 1M** | **~854 GiB** |

**KV is not the constraint.** 24 MLA layers × 576 × 2 B = 27,648 B/token, K-only (MLA allocates no V cache), so the entire 1M-token budget is 27 GiB — 3% of the weights. The pressure on this model is expert residency and compute buffers, which is where a native runtime has to win.

The fit check prices context in rather than assuming flat headroom, which is what lets it tell you that 8× B200 holds UD-Q4_K_XL at 32k but *not* at 1M — and that 4× H200 can't hold even UD-IQ1_S once compute buffers are counted, despite 553 < 564.

### Why Blackwell is a milestone, not a footnote

**Hopper has no FP4 tensor core.** K3's routed experts ship as MXFP4 (group size 32, 4-bit) and the target GGUF is 2-bit, so on M1 every low-bit expert GEMM must dequantize to bf16 before the MMA. llama.cpp does the same on every backend, so the M1 comparison is *fair* — but removing that dequant is the M2/M3 opportunity, and it can't be measured until there's an M1 number to measure against.

## Quickstart

The scripts detect arch and GPU count, build the pinned reference, fetch the 19 shards, and refuse to run if the weights plus the 1M KV cache won't fit in HBM:

```bash
export KIMI_K3_MODELS_DIR=/workspace/models_k3     # needs ~900 GiB free

# plan the whole run — no network, no GPU, no weights
bench/scripts/kimi_k3_baseline.sh --node h200x8 --dry-run

# M1: fetch + build + sweep decode/prefill at 128 / 512 / 4k / 32k
bench/scripts/kimi_k3_baseline.sh --node h200x8 --download

# ...then close the milestone by probing 128k / 256k / 1M (1 rep — a 1M prefill is
# minutes per rep, so it's a capability probe, not a median)
bench/scripts/kimi_k3_baseline.sh --node h200x8 --longctx

# M2 / M3 are the same command against a different node
bench/scripts/kimi_k3_baseline.sh --node b200x8 --download --longctx
bench/scripts/kimi_k3_baseline.sh --node b300x4 --download --longctx

# accuracy reference for accuracy_compare.py
bench/scripts/kimi_k3_reference_server.sh --ctx 8704
```

`--node` picks the `reference.lock` prefix (`KIMI_K3_H200X8_LLAMA_*` and friends), so **M1/M2/M3 cannot overwrite each other's reference** — they run the same quant at the same context, so the node is the only variable and a shared slot would conflate them. It also warns when the detected box doesn't match the claimed profile.

Results land in `bench/results/`. Tests need no GPU: `python3 bench/scripts/test_kimi_k3_baseline.py` (58 tests).

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

Same model, same quant, same 1M context throughout. M1–M3 change **only the node**, so each delta is attributable to hardware rather than to a moving target. M4 is the one capability milestone.

### M1 · 8× H200 · 1M context — a reference nobody can dispute

`KIMI_K3_NODE=h200x8` · [`targets/kimi_k3_ud_q2kxl_h200x8.yaml`](bench/configs/targets/kimi_k3_ud_q2kxl_h200x8.yaml)

- Pinned fork baseline, 19-shard verification, context-aware fits-in-HBM refusal — **landed**
- Sweep 8× H200 and pin measured decode/prefill at 128/512/4k/32k — **pending hardware**
- Prove 1M context loads and decodes with all weights in HBM (`--longctx`)

### M2 · 8× B200 · 1M context — the Blackwell delta

`KIMI_K3_NODE=b200x8` · [`targets/kimi_k3_ud_q2kxl_b200x8.yaml`](bench/configs/targets/kimi_k3_ud_q2kxl_b200x8.yaml)

- Like-for-like H200 → B200 on identical weights: how much of K3's cost *is* the FP4 dequant
- Native FP4 expert GEMM on `sm_100` vs llama.cpp's dequant-then-MMA
- At 1.8 TB/s NVLink5 (2× H200), is sharding experts rather than layers worth it
- First native work lands here: `kimi-k3` GGUF loader, KDA decode, MLA NoPE + output gate, latent MoE dispatch, `situ`, cross-layer attn residual, multi-GPU sharding

### M3 · 4×+ B300 · 1M context — fewer, fatter GPUs

`KIMI_K3_NODE=b300x4` · [`targets/kimi_k3_ud_q2kxl_b300x4.yaml`](bench/configs/targets/kimi_k3_ud_q2kxl_b300x4.yaml)

- 288 GB/GPU means **four** cards hold it where M1/M2 need eight — ~23 layers each, not ~12
- Does the reduced NVLink traffic beat the coarser MLA-heavy/KDA-heavy split imbalance?
- Confirm `compute_cap`, correct the arch label, and only then add it to the CMake arch list
- Stretch: at 8× B300 (2304 GiB) the lossless UD-Q8_K_XL fits — an **in-repo** quality reference instead of citing published figures

### M4 · Vision — MoonViT-3d

[`targets/kimi_k3_vision_m4.yaml`](bench/configs/targets/kimi_k3_vision_m4.yaml)

Runs on whichever node is available; the tower is 27 layers at 1024 wide, so it doesn't change the fit maths. `llama-mtmd-cli` and the `mmproj` GGUF are already wired up — the missing piece is the benchmark.

- Image-encode latency and output-token count vs resolution (both dynamic for this projector)
- Image-prompt logit agreement against the reference on identical weights
- A text-only regression guard: loading the `mmproj` must not move the M1–M3 numbers
- Deferred: MMMU-Pro / MathVision scored evals, and the video path — correctness before quality

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

## Automated evaluation

Correctness for K3 is **agreement with the reference on identical weights** — nothing else. There is no second independent K3 implementation, and even the target quant's own top-1 against full precision is 90.4%, so absolute quality numbers are meaningless. (M3's stretch goal — UD-Q8_K_XL on 8× B300 — would give this repo its first *in-house* lossless reference, instead of citing published figures.)

Two reference-server flags are mandatory, not tuning:

- `--no-context-shift` — K3 is a hybrid recurrent arch; llama.cpp cannot context-shift or restore slots for it, and a long eval dies mid-run without it.
- `--no-jinja` — the gate posts raw token ids; a chat template would prepend tokens the candidate never saw.

Details: [`eval/`](eval) · **[EVAL-TRUST.md](EVAL-TRUST.md)** (Polaris TDX receipts, reproducible from source today).

## Contributing

Source-required and reproducible. Before a PR: `bench/scripts/kimi_k3_baseline.sh --dry-run` and `python3 bench/scripts/test_kimi_k3_baseline.py`. Never hand-fill a `reference.lock` baseline — downstream, a hand-filled number is indistinguishable from a measured one. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) · [Changelog](CHANGELOG.md)
