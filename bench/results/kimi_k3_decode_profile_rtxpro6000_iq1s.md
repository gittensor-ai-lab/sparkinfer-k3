# Kimi K3 UD-IQ1_S decode profile — RTX PRO 6000 (sm_120), single GPU

First per-phase breakdown of the fp32 correctness-first executor. Measured over 13
layers (VRAM-capped on 96 GiB), 20-token greedy decode, via the env-gated
SPARKINFER_K3_PROFILE=1 phase timer (cudaEvent pairs around the attention and FFN
branches, drained lazily so the hot loop takes no extra sync).

## Throughput

    16.9 tok/s at 13 layers = 59.1 ms/token = ~4.5 ms/layer
    extrapolates to ~422 ms/token (2.4 tok/s) at the full 93 layers, single GPU

## Where the time goes

    phase        share    note
    ffn_moe      69.7%    top-16 expert dispatch (IQ1_S), the two moe_* kernels
    attn_kda     22.8%    Q8_0 projections q/k/v/o/g + conv + delta scan
    attn_mla      4.0%    q_lora + absorbed attention (only 24 of 93 layers)
    ffn_dense     3.5%    leading dense layer only (1 of 93)

## Reading

NOT bandwidth-bound. A layer reads ~600 MB (KDA attention ~500 MB of Q8_0 weights,
MoE experts ~100 MB of the top-16 at IQ1_S). At this card's ~1.8 TB/s that is ~0.33
ms/layer at roofline; we measure 4.5 ms — 13x off. So the cost is LATENCY and
OCCUPANCY, not bytes: tiny GEMV kernels (one block per output row), no fusion across
the ~30 kernels per layer, and per-launch overhead that dominates when each kernel
moves little data and finishes in microseconds.

## Optimization priority for the H200 pass

1. MoE expert dispatch (70%). moe_gate_up_situ_kernel and moe_down_combine_kernel
   currently launch one block per (output row, expert). With top_k=16 that is a lot
   of small blocks reading IQ1_S with an in-kernel decode. Targets: coalesce the
   IQ1_S block reads, parallelize across the 16 experts within a grid rather than
   serializing the combine, and fuse gate+up (they share x and the same expert row
   stride).

2. KDA attention (23%). The five Q8_0 projections go through k3_proj_f32 (one block
   per row, fp32 activation). The runtime already has faster Q8_0 paths for Qwen
   (launch_mmvq_q80 / dp4a) — but those take a bf16/int8 activation, so using them
   means either an f32 mmvq variant or accepting the activation quantization K3's
   f32 design avoided. A judgment call for the perf pass, not a free win.

Both are decode-latency problems, so CUDA-graph capture of the whole per-token
sequence (the Qwen path does this) is likely a larger lever than any single-kernel
rewrite: it removes the ~30x per-layer launch overhead across 93 layers at once.

## Caveat

fp32 throughout, single stream, no graph capture — this is the correctness baseline,
not a tuned number. The point of this profile is to say WHERE to spend the perf pass,
not to report a competitive tok/s.
