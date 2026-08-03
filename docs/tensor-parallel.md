# Tensor parallelism — foundation layer

Kimi K3 at the target quant is **802 GiB**. No single GPU holds it, so multi-GPU is
not an optimization here — it is the precondition for sparkinfer running K3 at all.

This document covered "what has landed vs what has not" when nothing ran. **It all runs
now**: sharded loading, a phase-split forward, and an end-to-end tensor-parallel decode of
all 93 layers on 8x H200. What remains is not integration but SCOPE — only the experts are
sharded. Read the status table, then the limit at the bottom.

## Status

| | |
|---|---|
| ✅ **Shard math** | `runtime/src/tp/shard.cpp` — CUDA-free, **4972 checks passing** |
| ✅ **Backend selection** | `runtime/src/tp/backend_select.cpp` — CUDA-free, **44 checks passing** |
| ✅ **Tensor→rule mapping** | `runtime/src/tp/weight_plan.cpp` — CUDA-free, **230 checks passing** |
| ✅ **NCCL collective** | bf16 **and f32** — **validated exact on 8× H200, every rank** |
| ✅ **Fast collectives** | peer-oneshot + multimem — peer validated on 8× H200 (bf16 and f32); multimem now has both dtypes |
| ✅ **Validation tool** | `tp_allreduce_check` — **run; numbers in `bench/results/`** |
| ✅ **Reduce points** | corrected: the MoE collective is at `expert_latent`, not hidden — see below |
| ✅ **f32 fast collectives** | peer-oneshot and multimem both reduce f32; K3 no longer falls back to NCCL for either |
| ✅ **Forward integration** | `kimi_k3_tp.cpp` — phase-split forward, one collective per MoE layer |
| ✅ **Sharded weight loading** | `upload_sliced()` consumes `plan_tensor_residency()`'s StridedCopy |
| ✅ **End to end** | 93 layers on 8x H200, text in text out, matches the pipeline to 1.85e-09 |
| ❌ **`ShardPolicy::Full` (lm_head)** | vocab RowShard still unsupported — Full today is Attn + dense FFN only |
| ✅ **Dense FFN under Full** | leading dense gate/up/down banded; one extra all-reduce/token |

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
MEASURED, ShardPolicy::ExpertsOnly:
  1 collective/MoE layer x 92 MoE layers = 92 all-reduces per decoded token
  expert_latent 3584 x 4 bytes (f32)     = 14 KiB per collective
  58.7 us/call on 8x H200 (NCCL)         = ~5.4 ms/token of pure collective
```

The 186 figure this section used to quote assumed a fully-sharded K3 — two collectives
per layer over all 93. `ExpertsOnly` replicates attention, so the attention reduce does
not exist and the leading dense layer has no expert dispatch: **92, not 186.**

f32, not bf16: K3's residual stream is f32 by design, so the payload is 3584 x 4 B rather
than 7168 x 2 B — the same 14 KiB, for a different reason.

Still latency-bound — 512x the data costs 1.45x the time — so `tp_allreduce_check` reports
microseconds per call rather than GB/s. But the measured 5.4 ms/token is **under 7% of the
281.6 ms token**, so the collective is NOT the bottleneck. Launch overhead in the ~30
unfused kernels per layer is.

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

Steps 1-3 of the old list are done: the collective is validated on hardware, the loader
shards, and the forward issues the reduce. What is left:

1. **`ShardPolicy::Full` — dense FFN done; lm_head vocab shard still open.** Attention
   bands and the leading dense FFN are sharded under Full (default with
   `SPARKINFER_K3_SHARD_DENSE` on). Vocab RowShard for lm_head is still refused by the
   loader allowlist — the forward indexes full vocab on rank 0.
2. **Batched prefill.** There is none; prompt ingestion is one forward per prompt token.
3. **CUDA-graph capture across ranks.** ~30 unfused launches per layer x 93 layers, and
   the profile is launch/occupancy-bound at 13x off roofline. An 8-stream capture with
   collectives inside is where the host-event vs in-kernel-barrier distinction starts to
   matter, and where `peer-oneshot` would finally earn its keep.
<<<<<<< HEAD
4. ~~**f32 fast collectives.**~~ **Done.** peer-oneshot and multimem both reduce f32;
   `SPARKINFER_TP_BACKEND=multimem` no longer downgrades to NCCL for K3's residual stream.
=======
4. **f32 fast collectives.** peer-oneshot carries an f32 path; multimem is still bf16-only
   on this branch (see the multimem-f32 PR).
>>>>>>> cd676f8 (feat(k3): ShardPolicy::Full — band the leading dense FFN)

## The limit of ExpertsOnly, and why it is now the whole story

`ShardPolicy::ExpertsOnly` bands the 896 routed experts (531 of UD-IQ1_S's 553 GiB) and
**replicates everything else, including attention**. That was the right first cut: it
captures essentially all of the memory win, needs one collective per layer instead of two,
and — because no activation changes width — lets the forward run at full `cfg` dims on
every rank with no per-rank shape threading.

Then the MoE dispatch got ~3x faster, and the arithmetic inverted:

| | before the MoE speedup | after |
|---|---:|---:|
| tp=1, 16 layers | 196.65 ms/token | **60.66** |
| tp=8, 16 layers | 80.61 ms/token | **54.33** |
| **tp=8 vs tp=1** | **2.44x** | **~1.1x** |

The replicated attention did not get faster and did not get sharded, so it became the
entire serial term. `ShardPolicy::Full` is declared and deliberately NOT enabled: the
loader would shard weights the executor still indexes at full width, which reads past the
end of the slice. Enabling it means threading per-rank head counts through the KDA and MLA
kernels and sizing the recurrent state per rank.

Absolute context: llama.cpp does **18.32 tok/s** on this box against sparkinfer's **3.55**.
TP is not what closes that gap on its own.
