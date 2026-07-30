#!/usr/bin/env python3
"""H1 (held-out / fuzzed eval prompt): emit a teacher-forcing token stream that a submission
CANNOT overfit, derived deterministically from a seed.

The accuracy gate used a single fixed in-repo prompt (`eval_text.txt`) — a contributor could read
it and special-case those exact tokens. Instead the eval bot passes a fresh, unpredictable seed each
run; this picks a random window of a diverse multi-domain corpus and a fuzzed token length from it.
The PR author can't know which slice will be scored, so correctness must generalize. The seed is
recorded in the eval log, so any third party reproduces the exact token stream and verdict.

Usage:  gen_eval_prompt.py <seed> <tokenizer.json> <corpus.txt> [fixed_text.txt] [--len N]
                           [--gguf model.gguf [--llama-tokenize BIN]]
Prints a space-separated list of token ids (the same format accuracy.sh feeds qwen3_gguf_score).
If <fixed_text.txt> is given AND seed=="fixed", reproduces the legacy fixed prompt (for bench.sh).

--gguf tokenizes with llama.cpp's llama-tokenize against the GGUF's own vocab instead of an
HF tokenizer.json. Required for models that ship no tokenizer.json at all: Kimi K3's vocab is
a tiktoken BPE (tiktoken.model + tokenization_kimi.py), so `tokenizers.Tokenizer.from_file`
has nothing to load. Reading the vocab out of the GGUF is also strictly more correct — it is
the exact vocab both the reference and the candidate will run — so <tokenizer.json> is ignored
when --gguf is given (pass "-" for it). llama-tokenize opens the model vocab_only, so this
costs metadata reads, not a weight load.

--len N emits a LONG stream of exactly N ids for the long-context probe (accuracy.sh's second
pass). The corpus is ~1.3k tokens, far short of the >= 8k needed to engage the int8-MMA and
sparse-KV kernels, so the stream is built by tiling seed-shuffled paragraph orders: every tile is
a fresh shuffle, so the result is long, non-degenerate, and still unpredictable to a submission —
the same H1 held-out property as the short prompt. --len is a no-op under seed=="fixed"+fixed_text,
which must stay byte-reproducible for bench.sh.
"""
import json, os, random, shutil, subprocess, sys


class _Encoded:
    """Minimal stand-in for tokenizers.Encoding — only .ids is ever used here."""
    def __init__(self, ids):
        self.ids = ids


class GgufTokenizer:
    """Tokenize via llama.cpp's llama-tokenize, reading the vocab straight out of the GGUF.

    --no-bos: the harness feeds these ids to BOTH engines verbatim, so nothing may add an
    implicit BOS on one side only. --ids prints a Python-parseable list.
    """

    def __init__(self, gguf_path, binary=None):
        self.gguf = gguf_path
        self.bin = binary or os.environ.get("LLAMA_TOKENIZE") or shutil.which("llama-tokenize")
        if not self.bin:
            raise SystemExit(
                "gen_eval_prompt: --gguf needs llama-tokenize; pass --llama-tokenize PATH "
                "or set LLAMA_TOKENIZE (built by _common.sh's LLAMACPP_EXTRA_TARGETS)")
        if not os.path.exists(self.gguf):
            raise SystemExit(f"gen_eval_prompt: no such GGUF: {self.gguf}")

    def encode(self, text):
        proc = subprocess.run(
            [self.bin, "-m", self.gguf, "--stdin", "--ids", "--no-bos"],
            input=text, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"gen_eval_prompt: llama-tokenize failed ({proc.returncode}): "
                             f"{proc.stderr.strip()[-500:]}")
        # llama-tokenize logs to stderr, but be defensive: take the last bracketed list.
        start = proc.stdout.rfind("[")
        end = proc.stdout.rfind("]")
        if start < 0 or end < start:
            raise SystemExit(f"gen_eval_prompt: llama-tokenize produced no id list: "
                             f"{proc.stdout.strip()[-500:]}")
        return _Encoded(json.loads(proc.stdout[start:end + 1]))


def load_tokenizer(tok_path, gguf, tok_bin):
    if gguf:
        return GgufTokenizer(gguf, tok_bin)
    from tokenizers import Tokenizer   # imported lazily: unavailable/unneeded for --gguf
    return Tokenizer.from_file(tok_path)


def build_long(rng, tok, paras, n_target):
    """Tile seed-shuffled paragraph orders until n_target ids, then truncate exactly."""
    ids, tile = [], 0
    while len(ids) < n_target:
        order = list(paras)
        rng.shuffle(order)
        ids.extend(tok.encode(f"\n\n[section {tile}]\n\n" + "\n\n".join(order)).ids)
        tile += 1
    return ids[:n_target]


def main():
    argv = [a for a in sys.argv[1:]]
    n_long = 0
    gguf = ""
    tok_bin = ""
    if "--len" in argv:
        i = argv.index("--len")
        n_long = int(argv[i + 1])
        del argv[i:i + 2]
    if "--gguf" in argv:
        i = argv.index("--gguf")
        gguf = argv[i + 1]
        del argv[i:i + 2]
    if "--llama-tokenize" in argv:
        i = argv.index("--llama-tokenize")
        tok_bin = argv[i + 1]
        del argv[i:i + 2]
    seed, tok_path, corpus_path = argv[0], argv[1], argv[2]
    fixed = argv[3] if len(argv) > 3 else ""
    tok = load_tokenizer(tok_path, gguf, tok_bin)

    if seed == "fixed" and fixed and not n_long:
        ids = tok.encode(open(fixed).read().strip()).ids
        print(" ".join(map(str, ids)))
        return

    rng = random.Random(seed)            # seed is a string; Random hashes it deterministically
    paras = [p.strip() for p in open(corpus_path).read().split("\n\n") if p.strip()]

    if n_long:
        print(" ".join(map(str, build_long(rng, tok, paras, n_long))))
        return

    # pick a random run of 2..5 adjacent paragraphs (held-out: which slice is unknown to the PR)
    k = rng.randint(2, min(5, len(paras)))
    start = rng.randint(0, max(0, len(paras) - k))
    ids = tok.encode(" ".join(paras[start:start + k])).ids
    # fuzz the SHAPE: score a seed-chosen window of 200..360 tokens (no fixed length to assume)
    n = rng.randint(200, 360)
    if len(ids) > n:
        s = rng.randint(0, len(ids) - n)
        ids = ids[s:s + n]
    if len(ids) < 32:                    # degenerate slice -> fall back to the whole corpus head
        ids = tok.encode(" ".join(paras)).ids[:n]
    print(" ".join(map(str, ids)))


if __name__ == "__main__":
    main()
