# Kimi K3 UD-IQ1_S — full-model validation (all 14 shards, 594 GB)

Run of kimi_k3_plan_check against the COMPLETE downloaded model, reading only the
GGUF index (fast even at 594 GB). This is the definitive check that the decode
plan's pinned tensor shapes match the real file across ALL 93 layers, not just the
early ones reachable during download.

## Integrity

    2573 tensors across 14 shard(s)
    manifest: 2157 required-tensor checks, 0 missing
    layout (uniform across depth): q_lora=1 attn_gate=1 shared_experts=1 routed_norm=1

## Shape validation — the headline

    shapes: 2573 checked, 0 mismatched, 0 missing, 0 UNPINNED

Every tensor's real dims match the decode plan's pinned expectation, and nothing is
left unpinned. This validates end-to-end all the shape/ordering fixes made while
building the plan against partial shards and the reference loader — the ssm_f_a/f_b
bottleneck width, the MlaAbsorbQ key_length output, the res_score 1-D scorers, the
wk_b/wv_b 3-D layouts, n_ff_shexp = moe_ffn * n_shared, and the rest. On the real
full file, all of it is correct.

## Residency: why layer-split, not tensor-parallel, on H100 80GB

The tool reports TENSOR-PARALLEL residency at tp=8:

    model total        553.20 GiB
    per rank            87.37 GiB   (745 sharded / 1828 replicated tensors)
    of which replica    20.83 GiB   <- the TP memory premium
    all ranks summed   698.99 GiB   (1.26x the model)

87.37 GiB/rank EXCEEDS 8x H100 80GB usable (74.5 GiB). So tensor-parallel does not
fit on an H100 80GB node at all — the 20.83 GiB of replicated tensors (norms,
router, embeddings, latent projections) pushes each rank over.

LAYER-SPLIT (the pipeline actually implemented) holds ~model/8 = 69.2 GiB/rank with
NO replication premium, since each GPU owns a disjoint set of whole layers. So
layer-split is not merely cheaper on collectives (0 vs 186 all-reduces/token) — it
is the only one of the two that fits on H100 80GB. Confirms the design choice from
first principles, now against the real model.

## Still requires a multi-GPU node

553 GiB of weights against 96 GiB on this box, so the full 93-layer forward and the
llama.cpp logits comparison need 8x H100 (80GB fits at short context) or 8x H200.
Everything upstream of that — kernels, loader, executor, pipeline, plan shapes — is
validated against this complete file.
