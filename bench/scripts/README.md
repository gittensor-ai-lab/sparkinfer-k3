# sparkinfer bench & accuracy harness

Turnkey scripts for a fresh NVIDIA Blackwell box (`sm_120` RTX 5090 / PRO 6000,
`sm_121` RTX Spark / Jetson Thor). They auto-detect the GPU arch, build what's
missing, fetch the model, and print results — no manual path-passing.

**Prereqs:** CUDA 12.8+ (or 13), CMake ≥ 3.20, a C++17 compiler, `git`, and
`pip install huggingface_hub tokenizers` (the accuracy script also needs `curl`).

## Quickstart

```bash
# 1) Decode throughput (downloads Qwen3-30B-A3B Q4_K_M on first run)
bench/scripts/bench.sh --download

# 2) Head-to-head vs llama.cpp on the same GGUF + same GPU (builds llama.cpp once)
bench/scripts/bench.sh --download --compare

# 3) Accuracy gate vs llama.cpp (token-match / KL / perplexity)
bench/scripts/accuracy.sh --download
```

Use your own model instead of `--download`:
```bash
bench/scripts/bench.sh /path/to/model.gguf --tokens 256 --compare
```

## Prebuilt binaries (no toolkit needed)

To avoid compiling, the scripts first try the **newest matching prebuilt binary**
published in the GitHub releases
([latest releases](https://github.com/gittensor-ai-lab/sparkinfer/releases)).
The default is `PREBUILT_TAG=latest`, which scans releases for the newest
`linux-x86_64-cuda13-sm<arch>` tarball matching your detected GPU arch. If that
prebuilt is missing or incompatible (different arch like sm_121, older
driver/CUDA, older glibc), the scripts **automatically fall back to a source
build**. Order of preference: existing local `build/` → prebuilt → source build.

Force a source build with `NO_PREBUILT=1`. Pin a specific release when you want
reproducibility:
```bash
PREBUILT_TAG=v0.2.0 bench/scripts/bench.sh --download
```

Manual use of a release bundle:
```bash
gh release download --repo gittensor-ai-lab/sparkinfer \
  --pattern 'sparkinfer-*-linux-x86_64-cuda13-sm120.tar.gz'
tar xzf sparkinfer-*-linux-x86_64-cuda13-sm120.tar.gz
./sparkinfer-bin/run qwen3_gguf_bench model.gguf 128
```

## What you get

`bench.sh` → sparkinfer decode tok/s + VRAM (and, with `--compare`, the llama.cpp
`tg128` number on the same card).

`accuracy.sh` → the correctness gate:
```
token-match (top-1)   : 100/100 = 1.000   (bar >= 0.90)
mean KL(llama||spark) : 0.136 nats
PPL sparkinfer        : 6.13   (exact)
PPL llama.cpp         : 7.76   (top-k+floor; inflated — see accuracy results doc)
```

## Using the accuracy gate for optimization (no silent regressions)

The same `score` tool gates an optimization against the **previous** sparkinfer build,
not just llama.cpp — expect **~100% top-1 + KL ≈ 0**:
```bash
build/runtime/qwen3_gguf_score model.gguf 20 <token-ids...>   # baseline, save output
# ... apply your kernel optimization, rebuild ...
build/runtime/qwen3_gguf_score model.gguf 20 <token-ids...>   # compare argmax + logprobs
```

## Knobs (env vars)

| var | default | purpose |
|---|---|---|
| `ARCH` | auto (`compute_cap`) | CUDA arch, e.g. `121` for RTX Spark |
| `MODELS_DIR` | `./models` | where the GGUF + tokenizer live |
| `MODEL_REPO` / `MODEL_FILE` | Qwen3-30B-A3B GGUF | model to fetch |
| `LLAMACPP_DIR` | `./.llamacpp` | reuse an existing llama.cpp checkout/build |
| `NO_PREBUILT` | `0` | set `1` to skip prebuilt binaries and build from source |
| `PREBUILT_TAG` | `latest` | newest matching prebuilt release; set a tag like `v0.2.0` to pin |
| `PREBUILT_URL` | auto | override with an exact prebuilt tarball URL |
| `SPARKINFER_EVAL_PREFILL_CHECK` | `1` | H3: batched vs token-loop prefill fidelity veto (`qwen3_gguf_prefill_check`) |
| `SPARKINFER_EVAL_PREFILL_CHECK_PREFIX` | `512` | prefix length for H3 |
| `SPARKINFER_EVAL_PREFILL_CHECK_TOP1_BAR` | `0.80` | H3 veto if TOP1 below this |
| `SPARKINFER_EVAL_PREFILL_CHECK_KL_BAR` | `0.05` | H3 veto if mean KL above this |
| `LLAMACPP_REF` | *(empty)* | ref to fetch when the pinned commit isn't fetchable by sha (fork/PR heads); the sha is still asserted |
| `LLAMACPP_EXTRA_TARGETS` | *(empty)* | extra llama.cpp targets to build, e.g. `llama-tokenize llama-mtmd-cli` |

## Kimi K3 baseline (8× H200)

Kimi K3 is 2.8T / A50B with **896 routed experts** — upstream `ggml-org/llama.cpp` asserts
`n_expert <= LLAMA_MAX_EXPERTS` (512) and cannot load it, so the reference engine is
`unslothai/llama.cpp` PR #48, pinned in `reference.lock` as `KIMI_K3_LLAMACPP_*`. Since a
fork PR head isn't fetchable by sha, `_common.sh` fetches `LLAMACPP_REF` and then asserts it
resolves to the pinned commit — a force-push fails the run instead of moving the baseline.

```bash
bench/scripts/kimi_k3_baseline.sh --dry-run            # plan only; no network, no GPU
bench/scripts/kimi_k3_baseline.sh --download           # 14 shards + sm_90 build + sweep
bench/scripts/kimi_k3_reference_server.sh --ctx 8704   # accuracy reference for accuracy_compare.py
```

The smallest useful quant (UD-IQ1_S) is 553 GiB, so `kimi_k3_check_fits` **refuses** to run on
a node that can't hold the weights in HBM — a partially-offloaded baseline isn't reproducible.
K3 also ships no `tokenizer.json` (tiktoken vocab), so prompt ids come from the GGUF via
`gen_eval_prompt.py --gguf <model.gguf>` + `llama-tokenize` instead of `ensure_tokenizer`.

sparkinfer has **no kimi-k3 loader yet**; this is the baseline half of the eval only. Details,
memory budget and architecture traps: [`docs/kimi-k3-baseline.md`](../../docs/kimi-k3-baseline.md).
Env: `PRIMARY_QUANT`, `KIMI_K3_MODELS_DIR`, `KIMI_K3_NGL/_BATCH/_UBATCH/_SPLIT_MODE/_EXTRA_FLAGS`.

## DFlash speculative decode (Qwen3.6)

Block-diffusion draft (`z-lab/Qwen3.6-35B-A3B-DFlash` safetensors) + target GGUF:

```bash
# Correctness: greedy DFlash must match AR (SPEC_AGREE = 100%)
bench/scripts/dflash_accuracy.sh /path/to/Qwen3.6-35B-A3B.gguf /path/to/Qwen3.6-35B-A3B-DFlash

# Throughput + mean accept τ
build/runtime/qwen3_gguf_dflash_bench target.gguf draft_dir 64 <token-ids...>
```

Env: `SPARKINFER_DFLASH=1` selects DFlash from `generate()` when a draft is attached;
`SPARKINFER_DFLASH_CAPTURE=1` logs capture setup. CB/server multi-accept is deferred until
SPEC_AGREE and a decode speedup are proven on the single-stream path.

## Automatic PR evaluation

`evaluate.sh` grades one submission (build → correctness → 128/512/4k/16k/32k speed → `label.py`).
Prefill `eval-prefill:{tier}` is sized by **TTFT reduction %** vs same-box main at the winning
context (`TTFT ≈ ctx / pp_tps`; `SPARKINFER_LABEL_LOWER_IS_BETTER=1`), not raw pp %-over-frontier.
For Qwen3.5 and Qwen3.6, the headline is the **max tier** of single-seq TTFT and CB mixed-load
`long_ttft_s` (`qwen3_gguf_cb_bench`: 4×decode + 8k interrupt). Dashboards still get `prefill_tps` as pp tok/s.
`evaluate_dual.sh` wraps it for **dual-model scoring**: it builds once, then scores **Qwen3.6-35B-A3B**
(the primary target, 128/512/4k for now) and guards **Qwen3-30B-A3B** against regression (full sweep +
accuracy) — any Qwen3 speed drop <98% of its main or broken llama.cpp parity REJECTs the submission.
Each model uses its own `MODELS_DIR` (different tokenizers) and weight-sha pin (`reference.lock`).
See [`eval/README.md`](../../eval/README.md) for the vast.ai orchestration and `--dual`.

Files: `bench.sh`, `accuracy.sh`, `accuracy_compare.py`, `evaluate.sh`, `evaluate_dual.sh`, `label.py`,
`eval_text.txt`, `reference.lock`, `_common.sh`,
`kimi_k3_baseline.sh`, `kimi_k3_reference_server.sh`, `_kimi_k3.sh`.
Results from reference runs live in [`../results/`](../results).
