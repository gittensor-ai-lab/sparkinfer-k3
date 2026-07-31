# Kimi K3 evaluation baseline — UD-Q2_K_XL at 1M context

Kimi K3 (Moonshot AI) is 2.8T total / ~50B active, hybrid KDA + MLA, 896 routed experts,
1M context, native vision. This document covers **the llama.cpp reference half of the eval**. The sparkinfer half — measuring the runtime and emitting the scoring verdict — is `bench/scripts/kimi_k3_eval.sh`; this doc covers what it is measured *against*:
producing a pinned, reproducible llama.cpp reference for **UD-Q2_K_XL at the full 1M
context**, across the three milestone nodes.

| | node | arch | total HBM | headroom @ 1M |
|---|---|---|---:|---:|
| **M1** | 8× H200 SXM 141 GB | `sm_90` Hopper | 1128 GiB | 274 GiB |
| **M2** | 8× B200 SXM 180 GB | `sm_100` Blackwell | 1440 GiB | 586 GiB |
| **M3** | 4×+ B300 288 GB | `sm_103`* Ultra | 1152 GiB | 298 GiB |
| **M4** | any of the above | — | — | vision adds ~1 GiB |

\* unverified — see [Milestone nodes](#milestone-nodes) below.

M1–M3 change **only the node**: same quant, same context, same flags, so each delta is
attributable to hardware rather than a moving target. M4 is the one capability milestone.

sparkinfer runs this model. `runtime/src/models/kimi_k3.cpp` carries a native K3 loader and
forward (KDA + MLA decode, latent MoE, `situ`, cross-layer residual), and
`runtime/src/models/kimi_k3_tp.cpp` runs it tensor-parallel across 8 GPUs. All 93 layers,
text in and text out, via `bench/scripts/kimi_k3_run.sh`.

**The reference still matters more than ever, because sparkinfer currently LOSES to it:**

    llama.cpp    18.32 tok/s      8x H200, UD-IQ1_S, ctx 128
    sparkinfer    3.55 tok/s      tensor-parallel, same box, same weights  -> 5.2x slower

Accuracy on identical weights and ids: top-1 100%, top-5 100%, mean KLD 4.05e-03. So the
measuring stick did its job — it gave the native work a same-box number to beat, and that
number is not yet beaten. See `bench/results/kimi_k3_tp_scaling_h200x8_iq1s.md` and
`bench/results/kimi_k3_e2e_h200x8_iq1s.md`.

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

## Why UD-Q2_K_XL

| quant | GiB | shards | fits @ 1M on M1 / M2 / M3 | reported top-1 vs Q8 |
|---|---:|---:|---|---:|
| UD-IQ1_S | 553 | 14 | yes / yes / yes | 78.9% |
| UD-IQ1_M | 604 | 15 | yes / yes / yes | 81.2% |
| UD-IQ2_XXS | 662 | 16 | yes / yes / yes | 84.1% |
| **UD-Q2_K_XL** | **802** | **19** | **yes / yes / yes ← target** | **90.4%** |
| UD-Q4_K_XL | 1407 | 32 | no / no / 8× B300 only | — |
| UD-Q8_K_XL | 1453 | 34 | no / no / 8× B300 only | lossless |

802 GiB at 90.4% top-1 is the accuracy knee. UD-IQ1_S is 249 GiB smaller but drops to
78.9%, which is too lossy to be the thing an optimization frontier is judged on — a 2%
kernel win means little next to a 12-point quality gap in the weights being measured.

Every quant above fits *some* node, but only with the weights **entirely in HBM**. That is a
hard requirement: a partially-offloaded baseline is not reproducible, so
`kimi_k3_check_fits` refuses to run one. It prices the KV cache in rather than assuming flat
headroom, which is why it can tell you that 8× B200 holds UD-Q4_K_XL at 32k but not at 1M —
and that 4× H200 cannot hold even UD-IQ1_S once compute buffers are counted, despite
553 < 564.

## Milestone nodes

### M1 · 8× H200 SXM — [`h200_sxm.yaml`](../bench/configs/h200_sxm.yaml)

- H200 is Hopper (sm_90), not Blackwell. Everything in `kernels/` targets sm_120/sm_121.
  The top-level `CMakeLists.txt` arch list already includes 90 so a source build *compiles*,
  but no sm_90 kernel path is tuned. Treat any future sparkinfer number here as untuned.
- **Hopper has no FP4 tensor core.** K3's routed experts ship as MXFP4 and the target GGUF
  is 2-bit, so every low-bit expert GEMM dequantizes before the MMA. llama.cpp takes the
  same path, so the comparison is fair — but it caps the ceiling, which is what M2 measures.

### M2 · 8× B200 SXM — [`b200_sxm.yaml`](../bench/configs/b200_sxm.yaml)

- First Blackwell datacenter node here, and the first with native FP4 tensor cores.
  Removing the dequant-before-MMA step is the M2 opportunity — a genuine gap, not a
  micro-opt.
- Same generation as `kernels/` but a **different target**: B200 is sm_100, the kernels are
  sm_120/sm_121. They do not run unmodified and the tensor-core/TMA shapes differ.
- NVLink5 at 1.8 TB/s per GPU (2× H200) changes whether sharding *experts* rather than
  layers is worth it. Worth re-measuring, not assuming.

### M3 · 4×+ B300 — [`b300.yaml`](../bench/configs/b300.yaml)

- 288 GB/GPU means **four** cards hold the target at 1M, where M1/M2 need eight. That is
  ~23 layers per GPU instead of ~12: fewer cross-GPU expert dispatches per token, but a
  coarser balance between the 24 MLA layers (all the KV cache) and the 69 KDA layers (all
  the recurrent state). Whether 4×288 beats 8×180 on decode is an open measurement.
- ⚠ **`sm_103` is unverified here.** `detect_arch` reads `compute_cap` at run time and wins,
  so a wrong label mis-files a result but does not mis-build. Confirm with
  `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`, override the profile's
  expectation with `KIMI_K3_B300_ARCH=<n>`, and only then add 103 to the CMake arch list —
  adding an arch the installed toolkit does not recognise breaks configure for every node,
  not just B300.
- Stretch: at 8× B300 (2304 GiB) the lossless UD-Q8_K_XL fits at 1M, which would give this
  repo its first **in-house** quality reference instead of citing published figures.

### M4 · Vision — [`kimi_k3_vision_m4.yaml`](../bench/configs/targets/kimi_k3_vision_m4.yaml)

Runs on whichever node is available — the MoonViT-3d tower is 27 layers at 1024 wide, so it
does not change the fit maths. `llama-mtmd-cli` and `mmproj-BF16.gguf` are already wired up
via `LLAMACPP_EXTRA_TARGETS`; the missing piece is the benchmark. Note that image encode is
a one-shot cost per image, not a per-token rate, so decode tok/s is the wrong primary
metric — see the config's `required_work` for what actually needs measuring.

## Run it

```bash
# 0. from the repo root, on the node
export KIMI_K3_MODELS_DIR=/workspace/models_k3     # needs ~560 GiB free (UD-IQ1_S default)

# 1. sanity-check the plan without touching the network or the GPUs
bench/scripts/kimi_k3_baseline.sh --node h200x8 --dry-run

# 2. fetch the 14 shards + build the pinned fork, then sweep the scored contexts (128..128k)
bench/scripts/kimi_k3_baseline.sh --node h200x8 --download

# 3. close the milestone: probe 128k / 256k / 1M
bench/scripts/kimi_k3_baseline.sh --node h200x8 --longctx

# 4. results land in bench/results/kimi_k3_<quant>_<node>_baseline_<stamp>.json
```

M2 and M3 are the same command with `--node b200x8` / `--node b300x4`.

The scored sweep runs `llama-bench` at contexts `128 512 4096 32768 131072`, 3 reps, taking the
median `samples_ts` for the prefill (`pp`) and decode (`tg`) rows separately. Override with
`--ctx "..."`, `--reps N`, `--decode-only`, `--prefill-only`.

`--longctx` is **separate and opt-in** because a 1M-token prefill is minutes of wall-clock
per rep. It runs at 1 rep and is recorded as `"kind": "probe"` rather than a median — it
answers "does 1M work, and at what rate", which is what the milestone requires. Results
without it print a reminder that the milestone is not closed.

### Pinning the result

`--node` selects the `reference.lock` prefix, so the three milestones cannot overwrite each
other:

| node | prefix |
|---|---|
| `h200x8` | `KIMI_K3_H200X8_LLAMA_*` |
| `b200x8` | `KIMI_K3_B200X8_LLAMA_*` |
| `b300x4` | `KIMI_K3_B300X4_LLAMA_*` |

This matters precisely *because* M1–M3 are identical except for the node: a single shared
set of slots would silently conflate three hardware generations into one number. Each set
has `_128 _512 _4K _32K _1M` plus `_4K_PP _32K_PP _1M_PP`.

All of them ship as `0` = **not measured**. Nothing in the harness may hand-fill them,
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
K3 implementation, and even the target quant's own top-1 against full precision is 90.4% — so absolute
quality numbers are meaningless. The only meaningful signal is **agreement with the
reference on identical weights**.

### Tokenization

K3 has no HF `tokenizer.json`; its vocab is a tiktoken BPE (`tiktoken.model` +
`tokenization_kimi.py`), so `tokenizers.Tokenizer.from_file` has nothing to load and
`ensure_tokenizer` does not apply. `gen_eval_prompt.py` gained a `--gguf` backend that
shells out to `llama-tokenize` and reads the vocab straight out of the GGUF:

```bash
python3 bench/scripts/gen_eval_prompt.py "$SEED" - bench/scripts/eval_corpus.txt \
  --gguf "$KIMI_K3_MODELS_DIR/UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf" \
  --llama-tokenize .llamacpp-k3/build/bin/llama-tokenize
```

This is strictly more correct than a side-loaded `tokenizer.json` — it is the exact vocab
both engines will run — and it is cheap: `llama-tokenize` opens the model `vocab_only`, so
it reads metadata, not 802 GiB of weights. `llama-tokenize` is built automatically via
`LLAMACPP_EXTRA_TARGETS`.

## Memory budget (UD-Q2_K_XL @ 1M)

Derived from the fork's own sizing functions, not estimated:

| item | size | source |
|---|---:|---|
| weights | 802 GiB | unsloth published |
| MLA KV @ 32k | 0.84 GiB | 24 MLA layers × 576 × 2 B/token, **K only** — `is_mla()` sets `has_v = false` |
| MLA KV @ 128k | 3.38 GiB | ” |
| MLA KV @ 1M | 27.0 GiB | ” |
| KDA recurrent state | 0.43 GiB / sequence | 69 layers × (128²×96 + 3×3×96×128) × 4 B — `n_embd_s` + `n_embd_r` |
| compute buffers | 24 GiB (allowance) | per-GPU, scales with ubatch; generous flat figure until characterised |
| **total @ 1M** | **~854 GiB** | what `kimi_k3_check_fits` requires |

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

## CI

CI is shaped by one limit: no GitHub-hosted runner will ever load an 802 GiB model. So it
gates what is cheap here and expensive on a rented node, and claims nothing more.

| job | what it proves |
|---|---|
| `shell` | `bash -n` over **every** tracked script; no CRLF in source |
| `python` | `py_compile` over every tracked file; every discovered `test_*.py` |
| `configs` | YAML parses; derived arch arithmetic self-consistent; links resolve |
| `plans` | `--dry-run` resolves for all three nodes; fork pin does not leak upstream |
| `lock` | every pinned baseline traces to a committed `bench/results/*.json` |
| `build-gate` | compiles for `sm_90` + `sm_100` (path-filtered) |
| `pin-audit` | weekly: PR #48 head + published shard counts still match the pins |

The `lock` job is the load-bearing one. `0` means not measured and is always allowed; a
non-zero baseline must match a recorded sweep for the same node **and** context, with prefill
and decode not interchangeable. Without it, `reference.lock` is just a text file somebody can
type a favourable number into — and downstream that is indistinguishable from a measurement.

`node-attestation` labels `needs-node-run` when perf-bearing code changes with no node run
attested. It labels only; it never closes a PR.

## Tests

```bash
python3 bench/scripts/test_kimi_k3_baseline.py     # 76 tests, no GPU / weights needed
python3 bench/scripts/check_reference_lock.py      # pinned baselines have measurements
python3 bench/scripts/audit_baseline_pins.py       # external pins vs reality (network)
```

Covers the things that are cheap to get wrong here and expensive to discover on a rented
node: the fork pin actually reaching the builder (and *not* leaking into the Qwen evals),
shard paths and completeness for all six published quants, the three node profiles and their
per-node `reference.lock` prefixes, the context-aware fits-in-HBM refusal at every
node/quant/context combination, and the GGUF tokenizer backend.

## Not done yet

- **One measured number.** `KIMI_K3_H200X8_IQ1S_LLAMA_128 = 18.3210` is pinned and backed by a committed sweep JSON (8x H200, UD-IQ1_S, ctx 128, 1 rep, decode only). Every OTHER `KIMI_K3_*_LLAMA_*` in `reference.lock` still ships
  as `0` and every sparkinfer slot in every target YAML is `null`. No sweep has run.
- **No sparkinfer `kimi-k3` loader** (M2) — see `runtime.required_work` in the model YAML.
- **No tuned kernel path** for sm_90, sm_100 or sm_103.
- **B300's arch is a guess** (M3). `arch_verified: false` in `b300.yaml` says so; confirm
  `compute_cap` before trusting any B300 label or adding 103 to the CMake arch list.
- **No vision benchmark** (M4). `llama-mtmd-cli` + `mmproj-BF16.gguf` are wired up in
  `_kimi_k3.sh` (`kimi_k3_mmproj`, `LLAMACPP_EXTRA_TARGETS`), but nothing invokes them and
  there is no MMMU-style gate.
