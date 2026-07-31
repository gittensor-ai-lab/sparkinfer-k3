#!/usr/bin/env python3
"""Seal a Kimi K3 eval verdict into a verifiable Polaris receipt.

    kimi_k3_attest.py --result RESULT.json --model FIRST_SHARD.gguf
                      [--node h200x8] [--out DIR]

WHY THIS EXISTS. `kimi_k3_eval.sh` emits a RESULT_JSON, and
`.github/workflows/eval-label.yml` turns it into an `eval:*` label. That workflow
re-derives the tier from the reported numbers, so the LABEL cannot be faked given the
numbers — but NOTHING BOUND THE NUMBERS TO A REAL RUN. Anyone with comment access could
invent a `tps` and get a matching tier computed for it, and the verdict lived only in a
PR comment, which is editable and deletable.

That was survivable when K3 was one repo among eighteen. It is not survivable now: the
registry points 100% of subnet emissions at this repo, so the unattested path is the only
path to the whole pool.

A receipt binds the verdict to its provenance — code commit, model SHA256, build hash,
llama.cpp reference commit, GPU/driver, and the measurements themselves — so a third
party can check that a score came from the claimed code on the claimed weights WITHOUT
re-running an 8x H200 job.

TWO MODES, chosen by what is configured:

    POLARIS_API_KEY set                -> Polaris TDX. Intel DCAP hardware root of
                                          trust; the receipt carries the quote and
                                          collateral so verification is offline and
                                          needs no trust in Polaris or in us.
    SPARKINFER_POLARIS_PRIVATE_KEY set -> Ed25519 fallback. Root of trust is our key,
                                          so it proves "sparkinfer signed this", not
                                          "hardware witnessed this". Weaker, and the
                                          receipt says so.
    neither                            -> attestation is still written, unsigned, and
                                          the exit code says UNSEALED. An unsigned
                                          attestation is a provenance record, not
                                          evidence — do not let it be read as one.

WHAT IS AND IS NOT ATTESTED. The benchmark runs bare-metal, NOT inside the enclave. That
is deliberate and EVAL-TRUST.md already explains why: running a GPU benchmark inside a
CC/TDX enclave degrades the very performance being measured, so you would get a
hardware-signed receipt of a CC-degraded number. What the enclave attests is the SCORING
— that these inputs, through this scoring code, yield this verdict. The link between the
bare-metal run and the sealed verdict is the provenance binding (model sha, build hash,
commit), not the enclave. Anyone reading a receipt should understand that distinction.
"""
import argparse
import hashlib
import json
import re
import os
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

try:
    from polaris.receipt import (
        AttestationBuilder, build_receipt, build_polaris_receipt,
        compute_build_hash, model_sha256, receipt_id_of,
    )
except ImportError as e:
    print(f"kimi_k3_attest: cannot import eval/polaris ({e})", file=sys.stderr)
    sys.exit(2)


def sh(*args, default=""):
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return default


def gpu_env():
    """Best-effort GPU/driver facts. Absent values stay empty rather than guessed —
    a receipt that invents a clock speed is worse than one that admits it lacked it."""
    q = sh("nvidia-smi",
           "--query-gpu=name,clocks.max.graphics,driver_version,compute_cap",
           "--format=csv,noheader", default="")
    name = arch = drv = ""
    clk = 0
    if q:
        parts = [p.strip() for p in q.splitlines()[0].split(",")]
        if len(parts) >= 4:
            name, mhz, drv, cc = parts[0], parts[1], parts[2], parts[3]
            clk = int("".join(ch for ch in mhz if ch.isdigit()) or 0)
            arch = f"sm_{cc.replace('.', '')}"
    return name, arch, drv, clk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="RESULT_JSON file from kimi_k3_eval.sh")
    ap.add_argument("--model", required=True, help="first GGUF shard (hashed into the receipt)")
    ap.add_argument("--node", default=os.environ.get("KIMI_K3_NODE", "h200x8"))
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "receipts"))
    ap.add_argument("--build-dir", default=str(ROOT / "build"))
    ap.add_argument("--eval-log", default="", help="raw harness output to archive with the run")
    ap.add_argument("--publish", action="store_true",
                    help="commit the sealed run to sparkinfer-k3-log (append-only)")
    ap.add_argument("--log-repo",
                    default=os.environ.get("KIMI_K3_LOG_REPO",
                                           "https://github.com/gittensor-ai-lab/sparkinfer-k3-log.git"))
    args = ap.parse_args()

    result = json.loads(Path(args.result).read_text())
    if "label" not in result or "tps" not in result:
        print("kimi_k3_attest: RESULT_JSON lacks label/tps — refusing to seal a "
              "verdict that is not a verdict", file=sys.stderr)
        return 2

    commit = sh("git", "-C", str(ROOT), "rev-parse", "HEAD", default="unknown")
    # The scoring scripts are pinned SEPARATELY from the tree commit: a receipt has to
    # record which scorer produced the verdict, and that is the thing a maintainer-only
    # path guards. Same list as sensitive-paths-guard.
    scoring = sh("git", "-C", str(ROOT), "log", "-1", "--format=%H", "--",
                 "bench/scripts/label.py", "bench/scripts/compare_logits.py",
                 "bench/scripts/kimi_k3_eval.sh", default="")
    build_hash = ""
    if Path(args.build_dir).is_dir():
        try:
            build_hash = compute_build_hash(args.build_dir)
        except Exception as e:
            print(f">> WARN: build_hash failed ({str(e)[:80]})", file=sys.stderr)
    else:
        print(f">> WARN: no build dir at {args.build_dir} — build_hash will be empty and "
              "verify_strict WILL reject this receipt. Run --build-dir on the eval node.",
              file=sys.stderr)

    print(f">> hashing {Path(args.model).name} (this is the slow part) ...")
    msha = model_sha256(args.model)

    name, arch, drv, clk = gpu_env()
    prov = result.get("provenance") or {}

    b = AttestationBuilder()
    b.set_code(repo="gittensor-ai-lab/sparkinfer-k3", commit=commit,
               build_hash=build_hash, scoring_scripts_commit=scoring)
    # llamacpp_commit comes from reference.lock, not the environment. The env var is
    # only set when a harness script sourced the lock first; a receipt that recorded ""
    # because a caller forgot would pass a lax verifier and fail a strict one for a
    # reason that has nothing to do with the run. Read the pin directly.
    llama_commit = os.environ.get("KIMI_K3_LLAMACPP_COMMIT", "").strip()
    if not llama_commit:
        lock = ROOT / "bench" / "scripts" / "reference.lock"
        if lock.is_file():
            m = re.search(r'KIMI_K3_LLAMACPP_COMMIT="\$\{[^:]+:-([0-9a-f]{40})\}"',
                          lock.read_text())
            if m:
                llama_commit = m.group(1)

    # eval_seed: K3's eval is deterministic on a FIXED prompt, so the "seed" that makes
    # a run reproducible is the prompt id sequence itself. Hash bench/refdata/hello.ids
    # — the exact ids the llama.cpp reference logits were captured on. A random seed here
    # would be theatre: nothing in this eval consumes one.
    eval_seed = str(prov.get("eval_seed", "")).strip()
    if not eval_seed:
        ids = ROOT / "bench" / "refdata" / "hello.ids"
        if ids.is_file():
            eval_seed = "ids:" + hashlib.sha256(ids.read_bytes()).hexdigest()[:16]

    b.set_references(model_sha256=msha, model_file=Path(args.model).name,
                     llamacpp_commit=llama_commit, eval_seed=eval_seed)
    b.set_environment(eval_mode=f"kimi_k3/{args.node}",
                      decode_tokens=int(prov.get("layers") or 0),
                      gpu_name=name, gpu_arch=arch,
                      clocks_pinned=False,          # measured: could not lock in-container
                      clock_mhz=clk, clock_spread_mhz=0, pin_target_mhz=0,
                      cuda_version=os.environ.get("CUDA_VERSION", ""),
                      driver_version=drv)
    b.set_measurements(result)
    b.set_verdict(result)
    b._timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    att = b.build()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    stamp = att["timestamp_utc"].replace(":", "").replace("-", "")
    base = outdir / f"k3_{args.node}_{stamp}"
    (base.with_suffix(".attestation.json")).write_text(json.dumps(att, indent=2, sort_keys=True))

    polaris_key = os.environ.get("POLARIS_API_KEY", "").strip()
    ed_key = os.environ.get("SPARKINFER_POLARIS_PRIVATE_KEY", "").strip()

    if polaris_key:
        from polaris.client import PolarisClient
        nonce = secrets.token_hex(16)
        print(">> sealing via Polaris TDX ...")
        try:
            resp = PolarisClient().attest_scoring(result, nonce=nonce, e2e_pubkey_b64="")
            rec = build_polaris_receipt(resp, att)
            mode = "polaris-tdx"
        except Exception as e:
            print(f"kimi_k3_attest: TDX attest failed ({str(e)[:160]})", file=sys.stderr)
            print("  NOT falling back silently — a receipt that claims a hardware root "
                  "it did not get is worse than no receipt.", file=sys.stderr)
            return 1
    elif ed_key:
        print(">> sealing via Ed25519 (software root of trust) ...")
        rec = build_receipt(att, bytes.fromhex(ed_key))
        mode = "ed25519"
    else:
        print("UNSEALED: no POLARIS_API_KEY and no SPARKINFER_POLARIS_PRIVATE_KEY.")
        print(f"  wrote provenance only: {base.with_suffix('.attestation.json')}")
        print("  This is a record, NOT evidence. Do not present it as an attested result.")
        return 3

    (base.with_suffix(".receipt.json")).write_text(json.dumps(rec, indent=2, sort_keys=True))
    rid = rec.get("receipt_id") or receipt_id_of(att)
    print(f"SEALED ({mode})")
    print(f"  receipt_id : {rid}")
    print(f"  verdict    : {result.get('label')}  tps={result.get('tps')}")
    print(f"  model_sha  : {msha[:16]}...")
    print(f"  commit     : {commit[:12]}  scoring={scoring[:12] or 'n/a'}")
    print(f"  receipt    : {base.with_suffix('.receipt.json')}")

    if args.publish:
        rc = publish(rid, result, att, rec, args.eval_log, args.log_repo)
        if rc:
            return rc
    else:
        print("  NOT published. eval-label.yml reads the receipt from the LOG, not from a "
              "comment, so an unpublished run cannot be scored. Re-run with --publish.")
    return 0


def publish(rid, result, att, rec, eval_log, log_repo):
    """Commit the sealed run to the append-only eval log.

    APPEND-ONLY IS ENFORCED HERE, not just by convention: if runs/<rid>/ already exists
    the publish is REFUSED rather than overwritten. A log whose entries can be replaced
    is not evidence of anything, and the receipt_id is a hash of the attestation — so a
    collision means either a genuine re-publish of the identical run (harmless to skip)
    or an attempt to swap the contents under a receipt someone already verified.
    """
    import shutil, tempfile
    tmp = tempfile.mkdtemp(prefix="k3log_")
    try:
        if subprocess.run(["git", "clone", "-q", "--depth", "1", log_repo, tmp]).returncode:
            print(f"kimi_k3_attest: cannot clone {log_repo}", file=sys.stderr)
            return 1
        rundir = Path(tmp) / "runs" / rid
        if rundir.exists():
            print(f"  already published: runs/{rid} exists in the log — leaving it alone.")
            return 0
        rundir.mkdir(parents=True)
        (rundir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        (rundir / "attestation.json").write_text(json.dumps(att, indent=2, sort_keys=True))
        (rundir / "receipt.json").write_text(json.dumps(rec, indent=2, sort_keys=True))
        if eval_log and Path(eval_log).is_file():
            shutil.copy(eval_log, rundir / "eval.log")

        # ledger + index: newest first, so the page needs no sort.
        entry = {
            "run_id": rid,
            "timestamp_utc": att.get("timestamp_utc", ""),
            "label": result.get("label"),
            "tps": result.get("tps"),
            "top1": result.get("top1"),
            "kl": result.get("kl"),
            "commit": att.get("code", {}).get("commit", ""),
            "attestation_type": rec.get("attestation_type", ""),
        }
        with open(Path(tmp) / "ledger.jsonl", "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        idx_p = Path(tmp) / "index.json"
        idx = json.loads(idx_p.read_text()) if idx_p.is_file() and idx_p.read_text().strip() else []
        idx.insert(0, entry)
        idx_p.write_text(json.dumps(idx, indent=2))

        g = ["git", "-C", tmp, "-c", "user.email=noreply@github.com",
             "-c", "user.name=sparkinfer-k3-eval"]
        subprocess.run(g[:3] + ["add", "-A"], check=True)
        subprocess.run(g + ["commit", "-q", "-m",
                            f"eval {rid}: {result.get('label')} tps={result.get('tps')} "
                            f"({rec.get('attestation_type','unsealed')})"], check=True)
        if subprocess.run(["git", "-C", tmp, "push", "-q", "origin", "HEAD:main"]).returncode:
            print("kimi_k3_attest: push to the log failed (token lacks write access?)",
                  file=sys.stderr)
            return 1
        print(f"  published   : {log_repo.rsplit('/',1)[-1].replace('.git','')}/runs/{rid}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
