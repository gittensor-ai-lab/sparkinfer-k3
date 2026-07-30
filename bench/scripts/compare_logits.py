#!/usr/bin/env python3
"""Mean KLD, top-1 agreement and RMS dp between two logit streams.

The same four metrics unsloth publish, but pointed at a different question.

  Their table answers "is the quant good?" — baseline is a high-precision model.
  This answers "is our implementation right?" — baseline is llama.cpp on the
  IDENTICAL quant file.

That distinction decides what a number means. Against a high-precision baseline,
UD-IQ1_S scores mean KLD 0.5645 and 78.9% top-1, and that is EXPECTED — it is a
1.5 bpw quant. Against llama.cpp on the same file, anything above numerical noise is
a bug in our code, because both sides should be evaluating the same arithmetic on the
same weights.

WHY THIS MATTERS MORE AT IQ1_S THAN AT Q2_K_XL. At 78.9% top-1 agreement with a good
reference, about one token in five legitimately differs. So "the output looks a bit
off" carries no information: a correct IQ1_S implementation and a broken one produce
equally plausible-looking text. Reading generations cannot separate them. Exact
agreement with llama.cpp can, which is why this is the harness to optimize against.

THRESHOLDS. Two implementations of the same arithmetic in the same precision differ
only by summation order, so expect:
    mean KLD      < 1e-5      (f32 reduction-order noise)
    top-1 agree   > 99.9%     (ties near-degenerate logits are the only legit misses)
    RMS dp        < 0.1%
Anything materially worse is a defect, not tolerance. In particular a HIGH top-1
agreement with a HIGH KLD means the argmax survives but the distribution is wrong —
usually a missing scale or a dropped delta term, which is exactly the IQ1_S failure
mode (its lattice is only {-1,0,+1}; the +/-0.125 delta carries the asymmetry).

FORMAT. Raw little-endian: magic 'SPKL' | uint32 version=1 | uint32 n_tokens |
uint32 n_vocab | float32[n_tokens][n_vocab]. Deliberately trivial so both llama.cpp
(via a small patch or the --kl-divergence-base dump) and sparkinfer can emit it
without a serialization dependency.

Usage: compare_logits.py ref.spkl test.spkl [--top-k 5] [--per-token N]
"""
import argparse
import struct
import sys

import numpy as np

MAGIC = b"SPKL"


def load(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            sys.exit(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")
        version, n_tok, n_vocab = struct.unpack("<III", f.read(12))
        if version != 1:
            sys.exit(f"{path}: unsupported version {version}")
        want = n_tok * n_vocab
        a = np.fromfile(f, dtype="<f4", count=want)
        if a.size != want:
            sys.exit(f"{path}: truncated — got {a.size} floats, header says {want}")
    return a.reshape(n_tok, n_vocab)


def log_softmax(x):
    # Subtract the row max before exponentiating. Without this, K3's 163840-wide
    # logits overflow float32 for any logit above ~88.
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--top-k", type=int, default=5,
                    help="also report top-k set agreement")
    ap.add_argument("--per-token", type=int, default=0,
                    help="print the N worst-disagreeing token positions")
    args = ap.parse_args()

    ref, test = load(args.ref), load(args.test)
    if ref.shape != test.shape:
        sys.exit(f"shape mismatch: ref {ref.shape} vs test {test.shape}")
    n_tok, n_vocab = ref.shape

    lp_ref, lp_test = log_softmax(ref), log_softmax(test)
    p_ref = np.exp(lp_ref)

    # KL(ref || test), summed over vocab, averaged over tokens — the direction
    # llama.cpp reports: how much information is lost by using `test` in place of the
    # reference. It is NOT symmetric, so the argument order is part of the metric.
    kld_per_tok = (p_ref * (lp_ref - lp_test)).sum(axis=1)
    mean_kld = kld_per_tok.mean()

    top1_ref, top1_test = ref.argmax(axis=1), test.argmax(axis=1)
    top1_agree = (top1_ref == top1_test).mean()

    p_test = np.exp(lp_test)
    dp_rms = np.sqrt(((p_ref - p_test) ** 2).mean())

    k = min(args.top_k, n_vocab)
    tk_ref = np.argpartition(-ref, k, axis=1)[:, :k]
    tk_test = np.argpartition(-test, k, axis=1)[:, :k]
    topk_agree = np.mean([len(set(a) & set(b)) / k for a, b in zip(tk_ref, tk_test)])

    print(f"tokens {n_tok}  vocab {n_vocab}")
    print(f"  mean KLD        : {mean_kld:.6e}")
    print(f"  top-1 agreement : {100*top1_agree:.4f} %")
    print(f"  top-{k} overlap   : {100*topk_agree:.4f} %")
    print(f"  RMS dp          : {100*dp_rms:.6f} %")
    print(f"  max per-token KLD: {kld_per_tok.max():.6e} "
          f"(token {int(kld_per_tok.argmax())})")

    if args.per_token:
        worst = np.argsort(-kld_per_tok)[: args.per_token]
        print("\n  worst-disagreeing positions:")
        for i in worst:
            print(f"    tok {int(i):6d}  KLD {kld_per_tok[i]:.4e}  "
                  f"ref_top1 {int(top1_ref[i]):6d}  test_top1 {int(top1_test[i]):6d}")

    # Same-implementation thresholds — see the module docstring. These are NOT quant
    # quality thresholds; they are "two implementations of one arithmetic" thresholds.
    ok = mean_kld < 1e-5 and top1_agree > 0.999 and dp_rms < 1e-3
    print("\n" + ("PASS: matches the reference implementation"
                  if ok else
                  "FAIL: divergence exceeds reduction-order noise — this is a defect"))
    if not ok and top1_agree > 0.99 and mean_kld > 1e-4:
        print("  hint: top-1 mostly survives but the distribution is wrong — look for a"
              "\n        missing scale or a dropped offset (for IQ1_S, the +/-0.125 delta).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
