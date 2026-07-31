# Kimi K3 end to end on 8× H200 — text in, text out

The first working inference of Kimi K3 through sparkinfer: a text prompt, the full
93-layer UD-IQ1_S model across 8 GPUs, and readable generated text back. Reproduce
with `bench/scripts/kimi_k3_run.sh`.

## Node

8× NVIDIA H200 SXM, `sm_90`, 143771 MiB each, all-pairs NV18 NVSwitch.
UD-IQ1_S, all 14 shards, 553.20 GiB. Layer-split pipeline (`kimi_k3_generate`),
~71.7 GiB per stage over 8 stages. fp32 executor, greedy decode, no sampling
parameters at all.

## Runs

```
prompt : The capital of France is
ids    : 1008 10484 318 15383 387
output :  Paris. The capital of France is Paris. The capital of France is Paris. ...
```

```
prompt : def fibonacci(n):
output :
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print
```

Correct factual answer on the first; a correct recursive implementation with correct
base cases and correct indentation on the second.

    decode: 39 forward passes in 12.709 s = 3.07 tok/s (325.9 ms/token, 93 layers)

## Reading

**The repetition on the first prompt is not a defect.** It is greedy argmax with no
repetition penalty, no temperature and no top-p, on a 1.5 bpw quant. Nothing in this
path samples. The second prompt shows the model is not stuck in a degenerate loop —
it terminates a function body and moves on.

**Throughput here is the pipeline path, and it is not the fast one.** Layer-split
means one GPU is busy at a time, so token latency is the SUM of the eight stages. The
tensor-parallel path reaches 281.6 ms/token on the same model (see
`kimi_k3_tp_scaling_h200x8_iq1s.md`) but is a decode benchmark with no sampling loop,
so it cannot generate text yet. The two produce the same logits — mean KLD 1.85e-09,
top-1 100% — so this is a wiring gap, not a correctness one.

## The three stages, and the one that is not sparkinfer's

1. **text → ids: llama.cpp's `llama-tokenize`, against the GGUF's own vocab.**
   K3 ships a tiktoken BPE (`tiktoken.model` + `tokenization_kimi.py`), not a
   `tokenizer.json`, so `tokenizers.Tokenizer.from_file` cannot read it and a
   side-loaded tokenizer would silently produce different ids. The vocab embedded in
   the GGUF is the one llama.cpp itself used, which makes it correct by construction.
   `--no-bos`, because `kimi_k3_generate` feeds ids verbatim and adds nothing.
   **sparkinfer has no encoder of its own.** This is the remaining external dependency.
2. **ids → ids: sparkinfer**, 93 layers across the 8 GPUs, greedy.
3. **ids → text: sparkinfer**, byte-level BPE inverse over `tokenizer.ggml.tokens`
   from the same GGUF (`runtime/src/models/kimi_k3_vocab.cpp`).

## Accuracy against llama.cpp

On the captured reference (`bench/refdata/hello.spkl`, same weights, same ids):

    mean KLD         4.045515e-03
    top-1 agreement  100.0000 %
    top-5 overlap    100.0000 %

Above the 1e-5 bar the harness wants. The remaining gap is most likely that sparkinfer
does not quantize activations before quantized mat-vecs where ggml does
(IQ1_S → Q8_K, Q8_0 → Q8_0 via `from_float`) — a deliberate design difference in K3's
f32 activation path, not an unfixed defect. It does not prevent coherent generation,
as the runs above show.

## Not covered

Prefill: prompt ingestion is a decode loop, one forward per prompt token. There is no
batched prefill path, so long prompts cost the same per token as generation. Sampling:
greedy only. Chat template: none applied — these are raw completions.
