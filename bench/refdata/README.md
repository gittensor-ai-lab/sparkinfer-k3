# K3 reference logits — ground truth for validating sparkinfer's executor

The acceptance test for the K3 executor is: on the IDENTICAL quant and the IDENTICAL
token ids, sparkinfer's logits must match llama.cpp's to near machine precision (mean
KLD < 1e-5, top-1 100%). Any larger gap is a bug in sparkinfer, not a property of the
quant — see bench/scripts/compare_logits.py, which encodes that distinction.

This directory holds captured llama.cpp reference data so that comparison can be run
WITHOUT re-running the (slow) full-model llama.cpp pass each time:

    <name>.spkl        llama.cpp's next-token logits after the prompt
    <name>.ids         the prompt token ids (feed these to sparkinfer verbatim —
                       both sides must use ONE tokenizer, llama.cpp's)
    <name>.txt         prompt text + argmax next token, human-readable
    <name>.stepK.spkl  per-step logits when captured with n_predict > 0

## The parity suite: ten depths, not one point

For most of this project the gate was ONE probe: `hello`, a 4-token prompt, graded on a
single next-token distribution. That is one row of one .spkl — n=1, with the KV cache
essentially empty — while the engine is SCORED at 131,072 tokens. Everything that only
appears once the cache is populated (the F16 latent cache, per-rank CUDA graphs, the
banded LM head, 2-D MoE sharding) was gated by a measurement that barely touches it. A
change can be bit-exact at 4 tokens and wrong at 100k, and that gate passes it clean.

The suite is now eight probes by default, with two more captured and one env var away:

| probe | tokens | source | reference captured on | default |
|---|--:|---|---|:--:|
| `ctx4` | 4 | `hello.*` | llama.cpp CPU | on |
| `ctx128` … `ctx4096` | 128 … 4096 | `longctx.*` | llama.cpp CUDA | on |
| `ctx8192` … `ctx32768` | 8192, 16384, 32768 | `longctx.*` | llama.cpp CUDA | **off** |

`longctx.ctx<L>.*` are nested prefixes of ONE document, so they test genuine long-range
dependency rather than several unrelated short prompts — and because they are nested, the
executor produces all of them in a single pass (`kimi_k3_tp_bench --checkpoints`), and
llama.cpp captures all of them from a single model load.

**Why 4096 is the default.** The executor has no batched prefill, so the deep pass runs
through the decode path one token at a time and costs `max(depth) / decode_tok_s` per
measured build — paid once for main plus once per PR in a round. Measured on the 8× H200
node against main @ `be3c818`, one continuous pass:

| to depth | elapsed | |
|---|--:|---|
| 4096 | 136 s (2.3 min) | default |
| 8192 | 236 s (3.9 min) | opt-in |
| 16384 | 429 s (7.2 min) | opt-in |
| 32768 | 812 s (13.5 min) | opt-in |

Decode holds a flat **~42 tok/s** across those segments — per-token cost is dominated by
streaming the active weights, and MLA keeps the context-dependent term small — so cost is
near-linear in `max(depth)`, and 32768 triples the parity budget to buy two more depths.
Turn them on for a change that plausibly breaks only very deep:

    K3_PARITY_DEPTHS=128,256,512,1024,2048,4096,8192,16384,32768

End to end, `measure_accuracy` costs more than the decode figure because it pays **two
model loads** — the 4-token probe and the deep pass are separate `kimi_k3_tp_bench`
invocations on different prompts. Measured at the 4096 default: **352 s** per measured
build, roughly 216 s of it model loading.

**The two capture backends are not interchangeable.** `hello.spkl` was captured on CPU
(the historical run, preserved so the number this gate has always reported stays
comparable); the deep set was captured with `--ngl 999` because a 32k CPU prefill pages
the whole 554 GB per ubatch and is effectively unbounded. CPU and CUDA llama.cpp differ in
float reduction order, so an absolute KL at `ctx4` is NOT comparable to an absolute KL at
`ctx128`. This does not affect the gate: the ratchet compares a PR against main measured
in the same round against the same reference file, depth by depth, so every comparison is
within one depth.

**What this does not prove.** 4096 is not 131072, and neither is 32768. The reference has
to come from llama.cpp and the capture cost grows with depth, so this narrows the untested
gap from "everything past 4 tokens" to "everything past 4k" — 32k with the opt-in depths
on. It does not close it. No claim of verified parity at the full scored context is
supported by these files.

## Captured references

    hello.*   prompt "Hello, world!" -> 4 tokens (ids 19180 11 2695 0), full 93-layer
              UD-IQ1_S via llama.cpp (kimi-k3-fullsize-vision @ efc8bc38), CPU + mmap.
              llama.cpp's argmax next token is 1379 (' This'), logit 13.8495.
              hello.step0/1.spkl are the next two greedy steps.

    longctx.* nested prefixes of longctx.txt at 128, 256, 512, 1024, 2048, 4096, 8192,
              16384 and 32768 tokens. Same model, same quant, same llama.cpp pin
              (efc8bc38), captured with --ngl 999 across 8x H200.

    longctx.txt
              Jane Austen, "Pride and Prejudice" (Project Gutenberg ebook 1342, public
              domain). Deterministically derived: the text between the PG markers, from
              "It is a truth universally acknowledged", CRLF->LF, runs of 3+ newlines
              collapsed to 2, truncated at the last paragraph break before 220,000 bytes.
              221,248 bytes, sha256
              d8f2cbc62bdc9508f6b03dc31866a13f271925453251cb0a1cd997af29582b03.
              Natural prose on purpose: high-entropy next-token distributions are a more
              sensitive probe than code or boilerplate, where the argmax is nearly always
              certain and KL stays near zero whatever the kernel does.

## Regenerating

Needs the K3-supporting llama.cpp (branch kimi-k3-fullsize-vision @ efc8bc38) built with
CUDA, and the UD-IQ1_S model on disk.

    # the deep suite (GPU; the box must be free — this loads 554 GB across 8 GPUs)
    bench/scripts/capture_parity_refs.sh /path/to/llama.cpp \
        /workspace/models_k3/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf

    # the historical 4-token probe (CPU)
    bench/scripts/dump_ref_logits.sh <llama.cpp-dir> <model.gguf> \
        bench/refdata/hello "Hello, world." 4

Regenerating changes every absolute KL the project reports. Re-measure main afterwards; do
not compare a number captured against a new reference to one captured against the old.

## Using it

    # feed the exact same ids to sparkinfer, dump its logits at every depth
    kimi_k3_tp_bench <model.gguf> 8 93 1 \
        --ids @bench/refdata/longctx.ctx32768.ids \
        --checkpoints 128,256,512,1024,2048,4096,8192,16384,32768 \
        --logits-prefix /tmp/ours --ctx 32784
    # compare, one depth at a time
    python bench/scripts/compare_logits.py \
        bench/refdata/longctx.ctx8192.spkl /tmp/ours.ctx8192.spkl

Against llama.cpp on the IDENTICAL quant, agreement should be near-exact (mean KLD < 1e-5)
— anything more is a bug in the executor, not the quant.

## These .spkl are the answer key

They are compared ON THE CONTROLLER and are never present on the evaluation box:
`k3_eval_bot._box_build` restores only `bench/refdata/*.ids` from origin/main and deletes
any `.spkl` the PR's own checkout carried. A bench that can read `hello.spkl` can write it
back out as its own `--logits` dump — top1 1.0, KL 0.0, for a binary that computed nothing
— and nothing downstream would reject a suspiciously exact result, because label.py only
bounds top1 >= 0.90 and KL <= 0.05.

The `.ids` DO ship to the box. They are inputs, they are public in this repo, and the
32k-token prompt is ~200 KB — past `MAX_ARG_STRLEN`, so it cannot be passed as an ssh
argument and must be read from the box's own checkout. Taking them from origin/main rather
than from the PR is what stops a PR swapping in a shorter prompt and being graded on it.

The .spkl files are large (n_vocab=163840 x 4 bytes = 640 KB each) but committable as
provenance for the acceptance test — the whole point is that they are captured once and
reused. Bench .gitignore un-ignores results/kimi_k3_* and this refdata dir.
