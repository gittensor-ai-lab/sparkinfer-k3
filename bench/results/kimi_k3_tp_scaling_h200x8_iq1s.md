# Kimi K3 UD-IQ1_S — tensor-parallel decode scaling, 8× H200

> **SUPERSEDED IN PART — read this first.** These numbers predate the MoE dispatch
> optimization (`BLOCK` 128→32 and the IQ lattice tables moved `__constant__`→`__device__`),
> which made the sharded half ~3× faster and therefore **changed the scaling conclusion**:
>
> | | before | after |
> |---|---:|---:|
> | tp=1, 16 layers | 196.65 ms/token | **60.66** |
> | tp=8, 16 layers | 80.61 ms/token | **54.33** |
> | tp=8 vs tp=1 | **2.44×** | **~1.1×** |
> | tp=8, 93 layers | 439.10 ms/token | **281.6** (3.55 tok/s) |
>
> The 2.44× below is still what was measured at the time, and the Amdahl analysis is still
> the right way to read it. But the conclusion inverted: once the MoE got 3× faster, the
> **replicated attention** that `ShardPolicy::ExpertsOnly` does not shard became the whole
> serial term. Sharding attention (`ShardPolicy::Full`) is now the lever, not the MoE.
>
> Absolute context: llama.cpp does **18.32 tok/s** on this box. sparkinfer is still slower.

First TP numbers for this model. Expert-parallel (`ShardPolicy::ExpertsOnly`): the 896
routed experts are banded across ranks, everything else replicated. Produced by
`runtime/examples/kimi_k3_tp_bench.cpp`; correctness is established separately by
`kimi_k3_tp_check` before any of this is worth reading.

## Node

8× NVIDIA H200 SXM, `sm_90`, 143771 MiB each, all-pairs NV18 NVSwitch.
NCCL 2.30.4+cuda13.2 (`NCCL_NVLS_ENABLE=0` — `cuMulticastBindMem` fails 401 without
Fabric Manager in this container). fp32 executor, single stream per rank, no CUDA-graph
capture.

## Measured — 16 layers, 8 tokens, warm-up token discarded

| tp | ms/token | tok/s | speedup | ms/layer | VRAM rank0 | collectives/token |
|---:|---------:|------:|--------:|---------:|-----------:|------------------:|
| 1  | 196.65   |  5.09 |  1.00×  | 12.290   | 94.08 GiB  | 0                 |
| 2  | 115.13   |  8.69 |  1.71×  |  7.195   | 54.57 GiB  | 15                |
| 4  |  92.48   | 10.81 |  2.13×  |  5.780   | 34.35 GiB  | 15                |
| 8  |  80.61   | 12.41 |  2.44×  |  5.038   | 24.25 GiB  | 15                |

15 collectives/token = 15 MoE layers (layer 0 is the leading dense block, which has no
expert dispatch and therefore no collective).

## Reading

**2.44× at tp=8, and that is close to the ceiling — not a disappointment.** Only the
MoE dispatch is sharded; attention is replicated and runs redundantly on every rank.
Solving Amdahl against the measured tp=8 point gives a parallel fraction of **0.67**,
consistent with the single-GPU profile's 69.7% MoE share. The remaining ~30% is
attention, which `ShardPolicy::ExpertsOnly` does not touch by construction. Getting
past ~2.6× requires sharding attention, not tuning the collective.

**The collective is not the bottleneck.** At tp=8 the measured floor is 92 × 58.7 µs ≈
5.4 ms/token of pure NCCL (`tp_allreduce_h200x8_20260730.md`), against 80.6 ms/token
measured over 16 layers — under 7%. Latency work belongs on the ~30 unfused kernel
launches per layer and on CUDA-graph capture, not on the reduce.

**tp=2 beats its own Amdahl prediction (1.71× against ~1.5×), and that is real.**
Sharding the experts also shrinks each rank's expert working set 2×, so the IQ1_S
in-kernel decode gets better locality. Speedup here is not purely a FLOP division;
part of it is that a smaller resident expert stack reads better. This effect fades as
tp grows and the fixed replicated cost dominates.

**VRAM does not scale 1/tp, by design.** 94.08 → 24.25 GiB is 3.9×, not 8×, because the
non-expert tensors are replicated on every rank. Independently measured on the real
file, the replication premium is **1.44×** of the model, i.e. ~99 GiB/rank for the full
553 GiB at tp=8 — comfortable in 141 GB cards, and the reason ExpertsOnly is a viable
policy rather than a stopgap.

## Extrapolation, and why it is only that

| tp | ms/token @ 93 layers | tok/s |
|---:|---------------------:|------:|
| 1  | 1143.0 | 0.87 |
| 2  |  669.2 | 1.49 |
| 4  |  537.5 | 1.86 |
| 8  |  468.5 | 2.13 |

**These are scaled per-layer costs, not measurements.** No 93-layer run has happened —
the download was incomplete when this was taken, and at tp=1 the full model does not fit
on one card at all, so a 93-layer single-GPU baseline cannot be produced on this node in
any case. Per-layer cost is the only quantity that compares across configurations.

The scaling also assumes layers are interchangeable, and they are not: the first 16
layers carry a different KDA/MLA mix than the full 93 (69 KDA / 24 MLA overall), and
MLA layers are the cheaper branch (4.0% of decode across 24 layers in the single-GPU
profile). Treat the 93-layer column as an order-of-magnitude, and replace it with a
real sweep once all 14 shards are resident.

## Not measured

Prefill. There is no K3 prefill path at all — prompt ingestion is a decode loop, one
forward per prompt token. At 1M context that is the dominant cost and none of these
numbers speak to it.
