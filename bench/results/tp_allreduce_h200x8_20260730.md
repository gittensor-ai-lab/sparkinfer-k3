# TP all-reduce validation — 8× H200 SXM

First measured numbers in this repository. Produced by
`runtime/examples/tp_allreduce_check.cpp`; correctness is verified before any timing.

## Node

| | |
|---|---|
| GPUs | 8× NVIDIA H200, `sm_90`, 143771 MiB each = **1123.2 GiB** total |
| interconnect | all-pairs `NV18` (full NVSwitch fabric), 26.562 GB/s per link |
| driver | 595.71.05 — reports `cudaDriverVersion 13020` (CUDA **13.2**) |
| toolkit | nvcc 13.3.73 |
| NCCL | 2.30.4+cuda13.2 |
| host | 240 cores, 2011 GiB RAM, container |
| commit | `ab0a584` + the group-allreduce fix |

## Correctness

Both backends **PASS** at every payload, on **every rank**. Rank *r* writes `r + 1`, so
the only correct sum is `8·9/2 = 36` — no partial or double-counted reduction lands
there by accident.

## Latency (200 iters, µs per call)

`µs/token` = per-call × 186, i.e. K3's 93 layers × 2 collectives. This is the floor TP
adds to decode **before any compute**.

| payload | | NCCL | peer-oneshot | winner |
|---|---|---:|---:|---|
| 7168 | **14.0 KiB — K3 decode** | 47.41 | **23.62** | **peer 2.01×** |
| 57344 | 112.0 KiB | 48.57 | **23.71** | peer 2.05× |
| 458752 | 896.0 KiB | 52.77 | **26.51** | peer 1.99× |
| 3670016 | 7168.0 KiB — prefill ubatch 512 | **69.07** | 162.93 | **nccl 2.36×** |
| | **µs/token at 14 KiB** | **8818** | **4394** | |

## What the numbers say

**Latency-bound, confirmed.** NCCL costs 47.67 µs at 14 KiB and 69.14 µs at 7 MiB —
512× the data for 1.45× the time. Bandwidth is irrelevant at decode size; per-call
overhead is the entire cost. Reporting GB/s here would have looked healthy while
decode starved.

**There is a crossover, and it is where the one-shot algorithm predicts.** A one-shot
has every rank read all N peers, so it moves N× the payload: free at 14 KiB, fatal at
7 MiB. NCCL's ring/tree moves ~2× regardless and wins once the message is large. So
**decode wants peer-oneshot, prefill wants NCCL** — a size-based dispatch, not a
single global choice.

**A hypothesis that turned out to be wrong, recorded because it saves someone else
the experiment.** NCCL's group path called `cudaSetDevice` once per rank per
collective — 8 switches × 186 collectives/token — which looked like obvious host
overhead on the critical path. Removing the redundant switches moved 14 KiB from
47.67 µs to **47.41 µs: 0.5%, i.e. nothing.** The device switches were not the
bottleneck. The change was kept anyway because restoring the caller's device is a
correctness fix in its own right (a collective that leaves a different device current
makes every later kernel launch in the forward land on the wrong GPU) — but it is not
a performance fix, and NCCL's ~47 µs at decode size is inherent to ring/tree across 8
ranks without NVLS.

**Both numbers are launch-bound, and both should improve.** 8 kernel launches from one
host thread at ~2-3 µs each accounts for most of peer-oneshot's 23.62 µs. That is what
CUDA-graph capture removes, and it is precisely where the in-kernel barrier earns its
keep: no host events and no cross-stream graph edges to serialize. Expect this number
to drop materially once the collective is captured inside the decode graph rather than
launched per call.

**NVLS/multimem is unavailable here and it matters.** All 8 devices report
`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 1`, but `cuMulticastBindMem` fails with CUDA
error 401 — Fabric Manager is not reachable from this container. NCCL hits the same wall
and must run with `NCCL_NVLS_ENABLE=0`. Multicast is the mechanism that makes small
all-reduces fast on H200 (the switch does the reduction, one load per rank instead of
N), so **part of NCCL's 47.67 µs is the absence of NVLS, not NCCL itself.** A node with
Fabric Manager exposed should be re-measured before drawing conclusions about either
backend's ceiling.

## Environment required on this node

```bash
export NCCL_NVLS_ENABLE=0     # cuMulticastBindMem fails 401 (no Fabric Manager)
export NCCL_IB_DISABLE=1      # no IB in the container; the ibv wrapper warns and stalls
export NCCL_P2P_LEVEL=NVL
```

NCCL must match the **driver's** CUDA version, not the toolkit's: 2.30.7+cuda13.3
failed `ncclCommInitAll` with "unhandled cuda error" against a 13.2 driver;
2.30.4+cuda13.2 works.

## Not measured

No model has been loaded. These are collective microbenchmarks only — there is no
sharded weight loading and no forward integration yet, so no end-to-end decode number
exists for any engine on this node.
