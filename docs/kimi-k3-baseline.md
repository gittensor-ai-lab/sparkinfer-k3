# Kimi K3 evaluation baseline — 8× H200

Kimi K3 (Moonshot AI) is 2.8T total / ~50B active, hybrid KDA + MLA, 896 routed experts,
1M context, native vision. This document covers **the baseline half of the eval only**:
producing a pinned, reproducible llama.cpp reference number on an 8× H200 node.

sparkinfer cannot run this model yet. `runtime/` loaders are Qwen-shaped
(`runtime/src/models/qwen35.cpp`, `runtime/include/sparkinfer/models/qwen_config.h`) and
there is no `kimi-k3` GGUF loader, no KDA decode path, and no sm_90 kernel path. The work
this branch lands is the *measuring stick*, so that native work has a same-box number to
beat instead of a claim.

## Why the baseline is a fork, not upstream llama.cpp

This is the one thing worth being precise about, because it is unusual: every other model
in this repo is benchmarked against `ggml-org/llama.cpp` at a pinned commit.

**Upstream cannot load Kimi K3 at all.** `llama-model.cpp` asserts
`hparams.n_expert <= LLAMA_MAX_EXPERTS`, and upstream's cap is 512. K3 has 896 routed
experts. There is no upstream number to compare against, so the reference has to be
[`unslothai/llama.cpp`](https://github.com/unslothai/llama.cpp) PR #48
(`kimi-k3-fullsize-vision`), pinned in [`bench/scripts/reference.lock`](../bench/scripts/reference.lock):

| pin | value |
|---|---|
| `KIMI_K3_LLAMACPP_REPO` | `https://github.com/unslothai/llama.cpp` |
| `KIMI_K3_LLAMACPP_REF` | `refs/pull/48/head` |
| `KIMI_K3_LLAMACPP_COMMIT` | `efc8bc38f0a9950cbb10ccef2cf48b951c39d3b2` |
| `KIMI_K3_LLAMACPP_BASE_COMMIT` | `cf67f0d24511864d2d3da0769108fd6fc16d00d1` (`kimi-k3-text-base`) |

Four things in the fork are load-bearing, not cosmetic:

1. **`LLAMA_MAX_EXPERTS 1024`** — without it the model asserts at load.
2. **`LLM_ARCH_KIMI_K3` graph** (`src/models/kimi-k3.cpp`) — hybrid KDA (linear, recurrent)
   + MLA (full) attention, plus five things Kimi-Linear-48B does not have: cross-layer
   residual attention, latent MoE, the `situ` activation replacing SwiGLU everywhere, an
   MLA sigmoid output gate, and a full-rank KDA gate.
3. **`graph_max_nodes(n_tokens) = max(n_tokens * 160, 64 * n_tensors)`** — the generic
   `n_tokens * 40` budget used by the other hybrid archs is exhausted at ubatch 3840.
   PR #48 lifts K3 out of that shared case specifically.
4. **Required-not-optional KV keys** — PR #48's first commit (`47c5bbdf`) turns
   `expert_latent_length`, `attn_res.block_size`, `situ_beta` and `situ_linear_beta` from
   silently-defaulted into `ml.get_key(...)` + `GGML_ASSERT`. That change is the difference
   between "loads and emits garbage" and "refuses to load a bad GGUF" — which is exactly
   the property a *baseline* needs.

Because the pin is a PR head on a fork, GitHub refuses fetch-by-sha. `_common.sh` therefore
fetches `LLAMACPP_REF` and then asserts the result resolves to `LLAMACPP_COMMIT`. **A
force-push to the PR branch fails the run** rather than silently changing the baseline.

## Why 8× H200 and not a Blackwell box

| quant | GiB | fits 8× H200 (1051 GiB)? | reported top-1 vs Q8 |
|---|---:|---|---:|
| UD-IQ1_S | 553 | yes ← default | 78.9% |
| UD-IQ1_M | 604 | yes | 81.2% |
| UD-IQ2_XXS | 662 | yes | 84.1% |
| UD-Q2_K_XL | 802 | yes | 90.4% |
| UD-Q8_K_XL | 1453 | no | lossless |

The smallest useful K3 quant is 553 GiB. That does not fit an RTX 5090 (32 GB), a PRO 6000
(96 GB), or eight of either. 8× H200 SXM at 141 GB each is the smallest node that holds
the weights entirely in HBM, which is a hard requirement: a partially-offloaded baseline is
not reproducible and `kimi_k3_check_fits` refuses to run one.

Two caveats to state plainly:

- **H200 is Hopper (sm_90), not Blackwell.** Everything in `kernels/` targets sm_120/sm_121.
  The top-level `CMakeLists.txt` arch list already includes 90 so a source build *compiles*,
  but no sm_90 kernel path is tuned. Treat any future sparkinfer number on this node as
  untuned until that work lands. See [`bench/configs/h200_sxm.yaml`](../bench/configs/h200_sxm.yaml).
- **Hopper has no FP4 tensor core.** K3's routed experts ship as MXFP4 and the useful GGUFs
  are 1–2 bit, so every low-bit expert GEMM dequantizes before the MMA. llama.cpp takes the
  same path, so the comparison is fair — but the ceiling is lower than on Blackwell.

## Run it

```bash
# 0. from the repo root, on the 8x H200 box
export KIMI_K3_MODELS_DIR=/workspace/models_k3     # needs ~600 GiB free
export PRIMARY_QUANT=UD-IQ1_S                       # or UD-Q2_K_XL for the accuracy knee

# 1. sanity-check the plan without touching the network or the GPUs
bench/scripts/kimi_k3_baseline.sh --dry-run

# 2. fetch the 14 shards + build the pinned fork for sm_90, then sweep
bench/scripts/kimi_k3_baseline.sh --download

# 3. results land in bench/results/kimi_k3_<quant>_baseline_<stamp>.json
```

The sweep runs `llama-bench` at contexts `128 512 4096 32768`, 3 reps, taking the median
`samples_ts` for the prefill (`pp`) and decode (`tg`) rows separately. Override with
`--ctx "128 512 4096 32768 131072"`, `--reps N`, `--decode-only`, `--prefill-only`.

Pin the measured numbers into `reference.lock` (`KIMI_K3_LLAMA_128` … `_32K_PP`) once you
have them. They ship as `0` = **not measured**; nothing in the harness may hand-fill them,
because downstream a hand-filled baseline is indistinguishable from a measured one.

### Accuracy reference

```bash
bench/scripts/kimi_k3_reference_server.sh --ctx 8704     # backgrounds, waits for /health
python3 bench/scripts/accuracy_compare.py <candidate_score.txt> - <ids_file> http://localhost:8081 64
bench/scripts/kimi_k3_reference_server.sh --stop
```

Two K3-specific server flags are mandatory, both added by `kimi_k3_server_flags`:

- `--no-context-shift` — K3 is a hybrid recurrent arch; llama.cpp cannot context-shift or
  restore slots for it, and a long eval dies mid-run without this.
- `--no-jinja` — `accuracy_compare.py` posts raw token ids to `/completion`; applying the
  chat template would prepend tokens the candidate never saw.

Note what the accuracy gate can and cannot tell you here. There is no second independent
K3 implementation, and UD-IQ1_S's own top-1 against full precision is 78.9% — so absolute
quality numbers are meaningless. The only meaningful signal is **agreement with the
reference on identical weights**.

### Tokenization

K3 has no HF `tokenizer.json`; its vocab is a tiktoken BPE (`tiktoken.model` +
`tokenization_kimi.py`), so `tokenizers.Tokenizer.from_file` has nothing to load and
`ensure_tokenizer` does not apply. `gen_eval_prompt.py` gained a `--gguf` backend that
shells out to `llama-tokenize` and reads the vocab straight out of the GGUF:

```bash
python3 bench/scripts/gen_eval_prompt.py "$SEED" - bench/scripts/eval_corpus.txt \
  --gguf "$KIMI_K3_MODELS_DIR/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf" \
  --llama-tokenize .llamacpp-k3/build/bin/llama-tokenize
```

This is strictly more correct than a side-loaded `tokenizer.json` — it is the exact vocab
both engines will run — and it is cheap: `llama-tokenize` opens the model `vocab_only`, so
it reads metadata, not 553 GiB of weights. `llama-tokenize` is built automatically via
`LLAMACPP_EXTRA_TARGETS`.

## Memory budget (UD-IQ1_S, 8× H200)

Derived from the fork's own sizing functions, not estimated:

| item | size | source |
|---|---:|---|
| weights | 553 GiB | unsloth published |
| MLA KV @ 32k | 0.84 GiB | 24 MLA layers × 576 × 2 B/token, **K only** — `is_mla()` sets `has_v = false` |
| MLA KV @ 128k | 3.38 GiB | ” |
| MLA KV @ 1M | 27.0 GiB | ” |
| KDA recurrent state | 0.43 GiB / sequence | 69 layers × (128²×96 + 3×3×96×128) × 4 B — `n_embd_s` + `n_embd_r` |
| headroom @ 1M | ~470 GiB | |

The takeaway: **KV is not the constraint.** The entire 1M-token budget is 27 GiB. The
memory pressure on this model is expert residency and compute buffers, which is where a
native runtime would have to win.

## Architecture reference

[`bench/configs/models/kimi_k3.yaml`](../bench/configs/models/kimi_k3.yaml) carries the full
spec, every field traced to either `moonshotai/Kimi-K3/config.json` or the fork's
`conversion/kimi_k3.py` / `src/models/kimi-k3.cpp`. Three traps worth calling out:

- **`full_attn_layers` is 1-indexed.** The converter uses `(il + 1) in full_attn_layers`.
  An off-by-one produces garbage silently, which is why the YAML key is named
  `full_attention_layers_1based`. 24 MLA layers, 69 KDA layers, 93 total.
- **MLA is encoded as MQA.** The converter sets `num_key_value_heads = 1` and
  `key_length = kv_lora_rank + qk_rope_head_dim = 576`; per-layer `head_count_kv == 0` is
  what marks a KDA layer. K3 is NoPE-only, so llama.cpp reports `LLAMA_ROPE_TYPE_NONE`.
- **Routed experts run in a down-projected space.** `expert_latent_length = 3584`, not
  `hidden_size = 7168`; expert FFN is 3072 wide. Sizing expert GEMMs off `hidden_size` will
  be wrong by 2×.

## Tests

```bash
python3 bench/scripts/test_kimi_k3_baseline.py     # 35 tests, no GPU / weights needed
```

Covers the things that are cheap to get wrong here and expensive to discover on a rented
node: the fork pin actually reaching the builder (and *not* leaking into the Qwen evals),
14-shard path and completeness handling, the fits-in-HBM refusal, and the GGUF tokenizer
backend.

## Not done on this branch

- No sparkinfer `kimi-k3` loader — see `runtime.required_work` in the model YAML.
- No sm_90 kernel path.
- No vision-path baseline. `llama-mtmd-cli` + `mmproj-BF16.gguf` are wired up in
  `_kimi_k3.sh` (`kimi_k3_mmproj`, `LLAMACPP_EXTRA_TARGETS`) but there is no image benchmark
  or MMMU-style gate yet.
- No measured numbers. Everything in `reference.lock` ships as `0`, and every sparkinfer
  slot in the target YAML is `null`, until the sweep has actually run on real hardware.
