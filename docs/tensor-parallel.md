# Tensor parallelism — foundation layer

Kimi K3 at the target quant is **802 GiB**. No single GPU holds it, so multi-GPU is
not an optimization here — it is the precondition for sparkinfer running K3 at all.

This document covers **what has landed** (the shard math, the collective
abstraction, the NCCL backend, the hardware validation tool) and **what has not**
(the forward-path integration). Read the status table before trusting anything.

## Status

| | |
|---|---|
| ✅ **Shard math** | `runtime/src/tp/shard.cpp` — CUDA-free, **4972 checks passing** |
| ✅ **Backend selection** | `runtime/src/tp/backend_select.cpp` — CUDA-free, **44 checks passing** |
| ✅ **Tensor→rule mapping** | `runtime/src/tp/weight_plan.cpp` — CUDA-free, **230 checks passing** |
| ✅ **NCCL collective** | bf16 **and f32** — **validated exact on 8× H200, every rank** |
| ✅ **Fast collectives** | peer-oneshot + multimem — **validated on 8× H200 (bf16 only)** |
| ✅ **Validation tool** | `tp_allreduce_check` — **run; numbers in `bench/results/`** |
| ✅ **Reduce points** | corrected: the MoE collective is at `expert_latent`, not hidden — see below |
| ⚠️ **f32 fast collectives** | peer-oneshot/multimem are bf16-only; K3 f32 falls back to NCCL |
| ❌ **Forward integration** | no model forward issues a collective yet |
| ❌ **Sharded weight loading** | the loader still slices by LAYER RANGE, never by row/col/expert |

### Two corrections this table used to hide

**1. The MoE collective is at `expert_latent` (3584), not hidden (7168), and it lands
BEFORE `routed_norm` — not after `routed_up`.** The routed experts are `ExpertShard`, so
each rank's dispatch accumulator is a *partial* sum over the top-16. `ffn_routed_norm` is
an RMS norm and rms_norm is not linear, so the cross-rank sum has to complete first. The
earlier plan put a single FFN reduce after `routed_up`, which was wrong twice: it skipped
the reduce the experts need, and by that point `routed_norm`/`routed_up`/`shexp` are all
`Replicate`, so every rank already held the complete tensor — reducing it **multiplies the
FFN output by tp_size**. Exactly the failure `kimi_k3_decode_plan.h` warns is unspottable
by eye at 186 reduces per token. `kimi_k3_decode_plan_cpu_test.cpp` now asserts it, and
that assertion fails on the old plan.

Note `rule_needs_reduce()` cannot express this: it answers `ColShard` only, and
`ExpertShard` is *also* used for `attn_k_b`/`attn_v_b`, which band the head axis — heads
concatenate rather than sum, so those genuinely need no collective. The distinction is the
**op**, not the rule, which is why the assertion is on `MoeDispatch`.

**2. K3 needs an f32 all-reduce, and the collective was bf16-only.** K3 runs an f32
residual stream deliberately (every kernel is transcribed from a float64 reference and
validated near machine epsilon). Routing its activations through `allreduce_bf16` would
truncate to ~8 mantissa bits **186 times per token, at the layer boundary** — undoing the
executor's numerics. `allreduce_f32_group()` now exists; `supports_f32()` reports it, and
`make_collective(..., need_f32=true)` downgrades a bf16-only fast backend to NCCL *before*
the 20-minute weight load rather than failing at the first collective.

The split is deliberate. TP bugs do not live in the collective — NCCL is correct.
They live in the shard arithmetic, and that is exactly the part that can be
verified without a GPU. So the shard math is tested to death here, and the CUDA
plumbing is written to be validated by one command on a real node.

## Why NCCL first

The custom communication engine at `/home/speedy/sn14/cacheon-sglang-miner/cuda/`
has two collectives that should beat NCCL on this workload:

| backend | mechanism | why it should win |
|---|---|---|
| `peer-oneshot` | every rank reads all peers, sums in FP32, **in-kernel flag barrier** | one kernel per rank, no host events, no cross-stream graph edges |
| `multimem` | `multimem.ld_reduce` — the **NVSwitch** does the reduction | one reduced load per rank instead of N peer loads; needs sm_90+ and NVLS |

**Both are now vendored** under `runtime/csrc/tp/`, and `select_backend()` honours
an explicit request for either on hardware that can run it. Two things remain true
and matter:

1. **Neither is validated on hardware.** `peer_oneshot_allreduce.h` says *"UNTESTED
   on the 8x H200 target as committed"*; `multimem_allreduce.h` says *"UNTESTED on
   hardware as committed"*. That is not selection's problem to solve — the
   protection is that `make_collective()` falls back to NCCL when construction
   fails, and that `tp_allreduce_check` must pass before a forward trusts either.
   **NCCL remains the default**, so nothing gets a fast collective by accident.
2. **Attribution.** `tp_allreduce.cuh` is *"Adapted from vLLM
   `csrc/custom_all_reduce.cuh` (Apache-2.0, Copyright the vLLM project)"* — the
   `Signal` layout, the dual-counter barrier, the acquire/release flag ops and the
   packed FP32 reduce. Apache-2.0 and MIT are compatible; §4 requires retaining the
   copyright notice and stating the file was modified. Both are done: the header
   keeps its attribution, and [`NOTICE`](../NOTICE) records it. The rest of the
   comm engine is first-party code.

### Two calling modes, because the fast backends need it

NCCL reduces the caller's own buffer in place, per rank. The fast backends
**cannot** — only multicast-bound (multimem) or peer-registered (one-shot)
allocations can back their loads, so the buffer has to belong to the collective:

```
Mode A  NCCL, TP=1 no-op        allreduce_bf16(buf, count, rank, stream)
Mode B  peer-oneshot, multimem  write reduce_in(rank) -> allreduce_group(count, streams)
                                -> read reduce_out(rank)
```

`Collective::owns_buffers()` says which. Mode B also launches **one kernel per
rank across every stream at once**, with the cross-rank barrier inside the kernel —
no host events, no cross-stream graph edges. That barrier mechanism, not the
reduce arithmetic, is what these backends exist to change.

Hiding Mode B behind a copy into/out of a caller buffer would add two 14 KiB
device-to-device copies per collective — **372 extra copies per K3 token** — and
erase the entire reason for using them. So `allreduce_bf16()` returns `false` on a
Mode B backend rather than silently staging, which makes a mis-wired forward fail
immediately instead of paying that cost 186 times per token.

## The shard plan

Two rules, fixed once in `shard.h` so the forward never re-derives them:

```
ROW shard (split the OUTPUT dim)   Wq  Wk  Wv  gate  up  lm_head
  rank owns a band of output rows -> produces a partial activation covering
  only its own outputs. No collective: the shard boundary IS the activation
  boundary.

COL shard (split the INPUT dim)    Wo  down
  rank owns a band of input columns -> produces a FULL-width partial sum.
  Needs an all-reduce. This is why TP costs exactly two collectives per layer.
```

Row-then-col is what makes it work with no redistribution mid-layer: rank *r*'s
`Wq` output rows line up exactly with rank *r*'s `Wo` input columns.
`test_row_and_col_shard_compose` pins that alignment.

### Three traps the tests exist to close

**1. KV heads often don't divide.** Qwen3.6 has 2 kv heads; a Kimi K3 GGUF stores
MLA as MQA with exactly **1**. At `tp_size 8` there is no way to give each rank a
whole kv head, and splitting one is *incorrect* — a kv group must be visible to
every query head attending through it. So when `n_kv_heads < tp_size` the kv
projections are **replicated** on every rank (`kv_replicated = true`) and only the
query side is sharded. The naive `n_kv_heads / tp_size` yields **0**, producing a
model with no keys on any rank that still loads and still emits text.

**2. Every axis must divide, not just the interesting one.** K3 has 896 experts and
`896 % 7 == 0`, so an expert-only check happily accepts `tp_size 7` — but K3 has 96
query heads and `96 % 7 != 0`. `shard_dims()` rejects the whole shape and names the
offending field. My own first draft of the test made exactly this mistake and the
test caught it.

**3. Bands must tile exactly.** A gap in the expert partition silently drops an
expert's contribution; an overlap double-counts it. Either way the all-reduce
combine is wrong and the output is plausible. `test_expert_bands_tile_exactly`
walks every rank at every viable `tp_size` and asserts each expert is owned once.

### MoE: shard experts, not expert width

An expert is already a natural unit — each rank runs whole experts for the tokens
routed to them, and the combine folds into the existing after-FFN all-reduce.
Sharding each expert's intermediate width instead would make every rank touch
every expert, multiplying weight traffic by `tp_size`. That fallback exists (for
shapes where experts don't divide but the width does) and is reported via
`experts_sharded = false`, but it is not the default.

## What the collective costs

K3, `tp_size 8`, 93 layers:

```
2 collectives/layer x 93 layers   = 186 all-reduces per decoded token
hidden 7168 x 2 bytes (bf16)      = 14 KiB per collective
```

14 KiB is nothing for NVLink. **186 round-trips per token is the entire decode
budget.** So this is latency-bound, `tp_allreduce_check` reports microseconds per
call rather than GB/s, and the per-call overhead of the collective mechanism —
not its throughput — is what decides whether TP=8 scales.

## Validate before trusting

```bash
cmake -B build -DSPARKINFER_TP=ON -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j2 --target tp_allreduce_check
./build/runtime/tp_allreduce_check 8 200
```

It prints probed capabilities, the selected backend, a correctness verdict, and
per-call latency at 14 KiB / 112 KiB / 896 KiB / 7 MiB. Exit 0 only if the
reduction is exact **on every rank**.

Correctness is checked first and cannot pass by accident: rank *r* fills its buffer
with `r + 1`, so the only correct sum is `tp*(tp+1)/2`. A missing rank, a
double-counted rank, or a rank reducing the wrong buffer all produce a different
total. And every rank is verified, not just rank 0 — a collective that reduces
correctly on one rank and not another is a real bug, and checking only rank 0 is
how it ships.

`SPARKINFER_TP_BACKEND=nccl|peer|multimem|none` overrides the request; an unknown
value falls back to NCCL with a warning rather than killing a run that took twenty
minutes to load 802 GiB.

## Build

`SPARKINFER_TP` is **OFF by default**. `src/tp/*.cpp` always compiles — the shard
math and selection are plain C++ — so the default single-GPU build is unchanged and
`TP=1` takes the same code path it always did. `SingleDeviceCollective` makes every
collective call a no-op that succeeds, so the forward can be written with
unconditional all-reduce calls and stay bit-identical at `TP=1`. That equivalence
is what makes it safe to land this before any of it runs on 8 GPUs.

`-DSPARKINFER_TP=ON` without NCCL is a **hard CMake error**, not a downgrade:
asking for multi-GPU and silently getting a single-GPU binary would let a "TP"
benchmark measure one card.

## Next, in order

1. **Run `tp_allreduce_check` on the 8× H200 node.** Nothing below is worth
   starting until the collective is proven correct and its latency is known.
2. **Sharded weight loading.** `load_gguf()` currently loads to one device; it
   needs to walk its tensors through `plan_for()` and place each rank's slice. The
   decisions are done and tested — the slicing primitives (`row_shard`,
   `col_shard`, `col_shard_row_offset`) and the per-tensor rules
   (`weight_plan.h`) — so what remains is copy plumbing, not design.

   Two things the plan will tell the loader that it must handle rather than
   ignore: a **fused `attn_qkv` with replicated kv is refused** (`Rule::Unknown`
   with a note), because q shards while kv replicates and that is not one rule —
   the loader has to split the fused tensor first. And any tensor with no rule is
   a **hard load error**, so a K3 tensor sparkinfer has no loader for (`attn_kv_b`,
   `attn_res_proj`) stops the load instead of being mis-sharded.
3. **Forward integration.** Insert the two all-reduces per layer at the points
   `ReducePoint` enumerates, and shard the lm_head with a cross-rank argmax
   (each rank's local max + index is all that needs exchanging, so that
   collective is tiny).
4. **CUDA-graph capture across ranks.** The existing decode path captures a graph;
   an 8-stream capture with collectives in it is where the host-event vs
   in-kernel-barrier distinction starts to matter, and where `peer-oneshot`
   would earn its keep — after step 1 has validated it.

Note that none of this makes sparkinfer run **K3**: there is still no `kimi-k3`
GGUF loader (see `bench/configs/models/kimi_k3.yaml` → `runtime.required_work`).
TP is exercisable against Qwen today, which is the right way to validate it.
