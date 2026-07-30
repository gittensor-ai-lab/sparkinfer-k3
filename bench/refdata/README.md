# K3 reference logits — ground truth for validating sparkinfer's executor

The acceptance test for the K3 executor is: on the IDENTICAL quant and the IDENTICAL
token ids, sparkinfer's logits must match llama.cpp's to near machine precision (mean
KLD < 1e-5, top-1 100%). Any larger gap is a bug in sparkinfer, not a property of the
quant — see bench/scripts/compare_logits.py, which encodes that distinction.

This directory holds captured llama.cpp reference data so that comparison can be run
WITHOUT re-running the (slow, disk-mmap) full-model llama.cpp pass each time:

    <name>.spkl        llama.cpp's next-token logits after the prompt
    <name>.ids         the prompt token ids (feed these to sparkinfer verbatim —
                       both sides must use ONE tokenizer, llama.cpp's)
    <name>.txt         prompt text + argmax next token, human-readable
    <name>.stepK.spkl  per-step logits when captured with n_predict > 0

## Regenerating

Needs the K3-supporting llama.cpp (branch kimi-k3-fullsize-vision @ efc8bc38) built,
and the UD-IQ1_S model on disk:

    bench/scripts/dump_ref_logits.sh <llama.cpp-dir> <model.gguf> \
        bench/refdata/hello "Hello, world." 4

## Using it

    # feed the exact same ids to sparkinfer, dump its logits
    kimi_k3_generate <model.gguf> 1 $(paste -sd' ' bench/refdata/hello.ids) --logits ours.spkl
    # compare
    python bench/scripts/compare_logits.py bench/refdata/hello.spkl ours.spkl

The .spkl files are large (n_vocab=163840 x 4 bytes = 640 KB each) but committable as
provenance for the acceptance test — the whole point is that they are captured once
and reused. Bench .gitignore un-ignores results/kimi_k3_* and this refdata dir.
