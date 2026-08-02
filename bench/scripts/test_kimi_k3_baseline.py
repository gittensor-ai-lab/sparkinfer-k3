#!/usr/bin/env python3
"""Unit tests for the Kimi K3 baseline plumbing (_kimi_k3.sh + _common.sh helpers).

These run with no GPU and no weights — they cover the parts that are easy to get
silently wrong and expensive to discover on a rented 8x H200 node:
  - the fork pin is what actually reaches the llama.cpp builder
  - shard paths / include globs are right for a 14-part GGUF
  - the fits-in-HBM precheck rejects an under-provisioned node
  - gen_eval_prompt's GGUF tokenizer backend parses llama-tokenize --ids output

Run from the repo root:
  python3 bench/scripts/test_kimi_k3_baseline.py
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "bench" / "scripts"

PINNED_COMMIT = "efc8bc38f0a9950cbb10ccef2cf48b951c39d3b2"
PINNED_REF = "refs/pull/48/head"
PINNED_REPO = "https://github.com/unslothai/llama.cpp"


def bash(snippet: str, env: dict | None = None) -> str:
    """Source _common.sh + _kimi_k3.sh, then run snippet; return stdout."""
    script = f'source "{SCRIPTS}/_common.sh"; source "{SCRIPTS}/_kimi_k3.sh"; {snippet}'
    full_env = {**os.environ, "PATH": os.environ.get("PATH", "")}
    if env:
        full_env.update(env)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=full_env)
    if out.returncode != 0:
        raise AssertionError(f"snippet failed ({out.returncode}): {out.stderr[-2000:]}")
    return out.stdout.strip()


def bash_rc(snippet: str, env: dict | None = None) -> int:
    script = f'source "{SCRIPTS}/_common.sh"; source "{SCRIPTS}/_kimi_k3.sh"; {snippet}'
    full_env = {**os.environ}
    if env:
        full_env.update(env)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          env=full_env).returncode


class ForkPinTest(unittest.TestCase):
    """The baseline engine must be the fork, and the pin must be exact."""

    def test_sourcing_kimi_k3_switches_the_engine(self):
        self.assertEqual(bash('echo "$LLAMACPP_REPO"'), PINNED_REPO)
        self.assertEqual(bash('echo "$LLAMACPP_REF"'), PINNED_REF)
        self.assertEqual(bash('echo "$LLAMACPP_COMMIT"'), PINNED_COMMIT)

    def test_qwen_evals_keep_the_upstream_pin(self):
        # _common.sh alone must NOT be contaminated by the fork pin.
        out = subprocess.run(
            ["bash", "-c", f'source "{SCRIPTS}/_common.sh"; echo "$LLAMACPP_REPO|$LLAMACPP_REF"'],
            capture_output=True, text=True, check=True).stdout.strip()
        repo, ref = out.split("|")
        self.assertEqual(repo, "https://github.com/ggml-org/llama.cpp")
        self.assertEqual(ref, "")   # upstream pin is fetchable by sha; no ref needed

    def test_k3_uses_a_separate_checkout(self):
        # Sharing one tree with the upstream pin means a full rebuild per alternation.
        k3_dir = bash('echo "$LLAMACPP_DIR"')
        self.assertTrue(k3_dir.endswith(".llamacpp-k3"), k3_dir)

    def test_reference_lock_pins_the_base_branch_too(self):
        lock = (SCRIPTS / "reference.lock").read_text()
        self.assertIn("KIMI_K3_LLAMACPP_COMMIT", lock)
        self.assertIn(PINNED_COMMIT, lock)
        self.assertIn("KIMI_K3_LLAMACPP_BASE_COMMIT", lock)

    def test_baselines_start_unmeasured_for_every_node(self):
        # A hand-filled baseline is indistinguishable from a measured one downstream, so
        # the lock must ship zeros until kimi_k3_baseline.sh has actually run — and it must
        # ship a SEPARATE set per node, or M2 silently overwrites M1's reference.
        for node in ("H200X8", "B200X8", "B300X4"):
            for suffix in ("128", "512", "4K", "32K", "1M", "4K_PP", "32K_PP", "1M_PP"):
                var = f"KIMI_K3_{node}_LLAMA_{suffix}"
                self.assertEqual(bash(f'echo "${var}"'), "0", var)

    def test_lock_prefix_matches_node(self):
        for node, want in (("h200x8", "KIMI_K3_H200X8_LLAMA_"),
                           ("b200x8", "KIMI_K3_B200X8_LLAMA_"),
                           ("b300x4", "KIMI_K3_B300X4_LLAMA_")):
            out = subprocess.run(
                ["bash", str(SCRIPTS / "kimi_k3_baseline.sh"), "--node", node, "--dry-run"],
                capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            self.assertIn(want, out.stdout, node)


class ShardPathTest(unittest.TestCase):
    # Shard counts as published in unsloth/Kimi-K3-GGUF. Pinned, not discovered — an
    # interrupted download must fail loudly rather than look complete.
    PUBLISHED = {"UD-IQ1_S": (14, 553), "UD-IQ1_M": (15, 604), "UD-IQ2_XXS": (16, 662),
                 "UD-Q2_K_XL": (19, 802), "UD-Q4_K_XL": (32, 1407), "UD-Q8_K_XL": (34, 1453)}

    def test_default_quant_is_iq1s(self):
        # THE DEFAULT AND THE TARGET ARE DIFFERENT CLAIMS, and this pair asserts both.
        # Default = UD-IQ1_S because it is what is resident, what the llama.cpp reference
        # was measured on, and what bench/refdata's logits were captured against — so the
        # harness works with no flags. Defaulting to an 802 GiB download nobody has means
        # every fresh invocation dies at verify_model_split.
        self.assertEqual(bash("kimi_k3_quant"), "UD-IQ1_S")
        self.assertEqual(bash("kimi_k3_first_shard"),
                         "UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf")
        self.assertEqual(bash("kimi_k3_n_shards"), "14")
        self.assertEqual(bash("kimi_k3_size_gib"), "553")

    def test_target_quant_is_q2kxl(self):
        # UD-Q2_K_XL remains the accuracy knee (90.4% top-1) and what the project is
        # judged on. UD-IQ1_S is 249 GiB smaller but 78.9% top-1 — fine for making the
        # harness runnable, too lossy to be the frontier's yardstick. Reachable via
        # PRIMARY_QUANT, and it must stay fully wired.
        env = {"PRIMARY_QUANT": "UD-Q2_K_XL"}
        self.assertEqual(bash("kimi_k3_quant", env), "UD-Q2_K_XL")
        self.assertEqual(bash("kimi_k3_first_shard", env),
                         "UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf")
        self.assertEqual(bash("kimi_k3_n_shards", env), "19")
        self.assertEqual(bash("kimi_k3_include_glob", env), "*UD-Q2_K_XL*")
        self.assertEqual(bash("kimi_k3_size_gib", env), "802")

    def test_every_published_quant_resolves(self):
        for q, (shards, gib) in self.PUBLISHED.items():
            env = {"PRIMARY_QUANT": q}
            self.assertEqual(bash("kimi_k3_n_shards", env), str(shards), q)
            self.assertEqual(bash("kimi_k3_size_gib", env), str(gib), q)
            self.assertEqual(bash("kimi_k3_first_shard", env),
                             f"{q}/Kimi-K3-{q}-00001-of-{shards:05d}.gguf", q)

    def test_quant_override_changes_every_derived_path(self):
        env = {"PRIMARY_QUANT": "UD-IQ1_S"}
        self.assertEqual(bash("kimi_k3_include_glob", env), "*UD-IQ1_S*")
        self.assertTrue(bash("kimi_k3_manifest", env).endswith("kimi_k3_ud_iq1_s.sha256"))

    def test_manifest_name_matches_quant(self):
        self.assertTrue(bash("kimi_k3_manifest").endswith("kimi_k3_ud_iq1_s.sha256"))
        self.assertTrue(bash("kimi_k3_manifest", {"PRIMARY_QUANT": "UD-Q2_K_XL"})
                        .endswith("kimi_k3_ud_q2_k_xl.sha256"))

    def test_gguf_path_joins_models_dir(self):
        got = bash("kimi_k3_gguf", {"KIMI_K3_MODELS_DIR": "/mnt/w"})
        self.assertEqual(got, "/mnt/w/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf")


class NodeProfileTest(unittest.TestCase):
    """M1/M2/M3 run the same quant at the same context, so the node is the only variable —
    a shared reference slot would silently conflate them."""

    EXPECTED = {"h200x8": (90, 8, 1128), "b200x8": (100, 8, 1440), "b300x4": (103, 4, 1152)}

    def test_default_node_is_m1(self):
        self.assertEqual(bash("kimi_k3_node"), "h200x8")

    def test_each_profile(self):
        for node, (arch, gpus, gib) in self.EXPECTED.items():
            env = {"KIMI_K3_NODE": node}
            self.assertEqual(bash("kimi_k3_node_arch", env), str(arch), node)
            self.assertEqual(bash("kimi_k3_node_gpus", env), str(gpus), node)
            self.assertEqual(bash("kimi_k3_node_gib", env), str(gib), node)

    def test_b300_arch_is_overridable(self):
        # sm_103 is unverified; detect_arch wins at run time, but the profile's expectation
        # must be correctable without editing the script.
        env = {"KIMI_K3_NODE": "b300x4", "KIMI_K3_B300_ARCH": "100"}
        self.assertEqual(bash("kimi_k3_node_arch", env), "100")

    def test_unknown_node_warns_but_does_not_fail(self):
        env = {"KIMI_K3_NODE": "nonsense"}
        self.assertEqual(bash_rc("kimi_k3_check_node", env), 0)

    def test_every_target_config_declares_its_node(self):
        tdir = ROOT / "bench" / "configs" / "targets"
        for node, milestone in (("h200x8", "M1"), ("b200x8", "M2"), ("b300x4", "M3")):
            f = tdir / f"kimi_k3_ud_q2kxl_{node}.yaml"
            self.assertTrue(f.exists(), f)
            text = f.read_text()
            self.assertIn(f"milestone: {milestone}", text)
            self.assertIn(f"node:     {node}", text)
            # Each target declares a context ceiling and the KV cost that sets it.
            # NOT hardcoded to 1M any more: the measured per-rank weight share
            # (129.0 GiB on H200, not 802/8) caps UD-Q2_K_XL at 128k there, while
            # B200/B300 have the headroom for more. What must hold is that the
            # declared max_context is one the budget actually supports.
            self.assertIn("kv_at_128k: 3.38", text)
            import re as _re
            m = _re.search(r"max_context: (\d+)", text)
            self.assertIsNotNone(m, f"{node}: no max_context")
            self.assertIn(int(m.group(1)), (131072, 262144, 524288, 1048576),
                          f"{node}: odd max_context {m.group(1)}")

    def test_m4_is_a_capability_not_a_node(self):
        text = (ROOT / "bench" / "configs" / "targets" / "kimi_k3_vision_m4.yaml").read_text()
        self.assertIn("milestone: M4", text)
        self.assertIn("node:     any", text)
        # the deliverable is the benchmark that does not exist yet
        self.assertIn("benchmark_exists: false", text)

    def test_superseded_iq1s_target_is_gone(self):
        # Two files each claiming to be "the target" is how a stale reference gets scored.
        self.assertFalse((ROOT / "bench" / "configs" / "targets"
                          / "kimi_k3_ud_iq1s_h200x8.yaml").exists())


class ContextBudgetTest(unittest.TestCase):
    """All three hardware milestones target 1M context, so the fit check has to price the
    KV cache in rather than assume a flat headroom."""

    def test_kv_gib_scales_with_context(self):
        # 24 MLA layers * 576 * 2 B = 27648 B/token, K only.
        for ctx, want in ((32768, 1), (131072, 4), (262144, 7), (1048576, 27)):
            self.assertEqual(int(bash(f"kimi_k3_kv_gib {ctx}")), want, ctx)

    def test_default_max_ctx_is_128k(self):
        # 128k, not 1M: the MEASURED per-rank weight share on 8x H200 is 129.0 GiB
        # (not 802/8 = 100.3, because 32.9 GiB replicates per rank), which leaves
        # room for 128k of KV on a 140 GiB card but not 256k or beyond.
        self.assertEqual(bash('echo "$KIMI_K3_MAX_CTX"'), "131072")

    def test_headroom_includes_kv_at_the_default_ctx(self):
        # 4 GiB KV @128k + 24 GiB compute buffers + 1 GiB state
        self.assertEqual(int(bash("kimi_k3_headroom_gib")), 29)
        # ...and still prices 1M correctly when asked for it explicitly.
        self.assertEqual(int(bash("kimi_k3_headroom_gib", {"KIMI_K3_MAX_CTX": "1048576"})), 52)

    def test_lower_max_ctx_shrinks_headroom(self):
        got = int(bash("kimi_k3_headroom_gib", {"KIMI_K3_MAX_CTX": "32768"}))
        self.assertEqual(got, 26)   # 1 + 24 + 1


class LlamaFlagTest(unittest.TestCase):
    def test_required_flags_present(self):
        flags = bash("kimi_k3_llama_flags")
        for needed in ("-ngl 99", "--split-mode layer", "-b 2048", "-ub 512"):
            self.assertIn(needed, flags)
        # The no-mmap spelling is PROBED from llama-bench --help, not hardcoded:
        # upstream replaced --no-mmap with -lm/--load-mode and deprecated -mmp. Asserting
        # one literal spelling is what made this test pass while the real invocation was
        # failing arg-parse on the node. Assert the INTENT instead.
        self.assertTrue(
            any(f in flags for f in ("--load-mode none", "--no-mmap", "-mmp 0")),
            f"no recognised no-mmap flag in: {flags}")

    def test_server_flags_cover_the_hybrid_arch(self):
        # Without --no-context-shift a long eval dies mid-run on a recurrent arch;
        # without --no-jinja the raw-token-id gate gets a chat template prepended.
        flags = bash("kimi_k3_server_flags")
        self.assertIn("--no-context-shift", flags)
        self.assertIn("--no-jinja", flags)

    def test_overrides_take_effect(self):
        flags = bash("kimi_k3_llama_flags", {"KIMI_K3_UBATCH": "1024", "KIMI_K3_NGL": "93"})
        self.assertIn("-ub 1024", flags)
        self.assertIn("-ngl 93", flags)


class FitCheckTest(unittest.TestCase):
    """An 802 GiB model on a too-small node silently half-offloads and yields a
    non-reproducible baseline. The precheck must refuse, not warn."""

    def _with_fake_smi(self, total_mib_per_gpu: int, n_gpu: int, env=None):
        tmp = tempfile.mkdtemp()
        smi = Path(tmp) / "nvidia-smi"
        smi.write_text(textwrap.dedent(f"""\
            #!/usr/bin/env bash
            case "$*" in
              *memory.total*) for i in $(seq 1 {n_gpu}); do echo {total_mib_per_gpu}; done ;;
              *compute_cap*)  echo 9.0 ;;
              *--query-gpu=index*) for i in $(seq 1 {n_gpu}); do echo $((i-1)); done ;;
              *--query-gpu=name*) for i in $(seq 1 {n_gpu}); do echo "NVIDIA H200"; done ;;
              *) exit 0 ;;
            esac
        """))
        smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
        e = {"PATH": f"{tmp}:{os.environ['PATH']}", "CUDA_VISIBLE_DEVICES": ""}
        if env:
            e.update(env)
        return e

    # Target is 802 GiB + 52 GiB headroom at 1M ctx = 854 GiB needed.
    H200 = 144384   # MiB per GPU, 141 GB
    B200 = 184320   # 180 GB
    B300 = 294912   # 288 GB

    def test_m1_8x_h200_fits_target_at_1m(self):
        env = self._with_fake_smi(self.H200, 8, {"KIMI_K3_NODE": "h200x8"})   # 1128 GiB
        self.assertEqual(bash_rc("kimi_k3_check_fits", env), 0)
        self.assertEqual(bash("gpu_count", env), "8")

    def test_m2_8x_b200_fits_target_at_1m(self):
        env = self._with_fake_smi(self.B200, 8, {"KIMI_K3_NODE": "b200x8"})   # 1440 GiB
        self.assertEqual(bash_rc("kimi_k3_check_fits", env), 0)

    def test_m3_4x_b300_fits_target_at_1m(self):
        # The whole point of M3: FOUR GPUs hold it, where M1/M2 need eight.
        env = self._with_fake_smi(self.B300, 4, {"KIMI_K3_NODE": "b300x4"})   # 1152 GiB
        self.assertEqual(bash_rc("kimi_k3_check_fits", env), 0)

    def test_4x_h200_rejects_target(self):
        env = self._with_fake_smi(self.H200, 4)      # ~564 GiB < 854
        self.assertEqual(bash_rc("kimi_k3_check_fits", env), 1)

    def test_2x_b300_rejects_target(self):
        env = self._with_fake_smi(self.B300, 2)      # ~576 GiB < 854
        self.assertEqual(bash_rc("kimi_k3_check_fits", env), 1)

    def test_8x_h200_rejects_q8_at_1m(self):
        env = self._with_fake_smi(self.H200, 8, {"PRIMARY_QUANT": "UD-Q8_K_XL"})
        self.assertEqual(bash_rc("kimi_k3_check_fits", env), 1)

    def test_8x_b300_accepts_q8_at_1m(self):
        # 2304 GiB — the M3 stretch goal: a lossless in-repo reference.
        env = self._with_fake_smi(self.B300, 8, {"PRIMARY_QUANT": "UD-Q8_K_XL"})
        self.assertEqual(bash_rc("kimi_k3_check_fits", env), 0)

    def test_8x_b200_rejects_q4_at_1m_but_accepts_at_32k(self):
        # 1407 + 52 = 1459 > 1440 at 1M; 1407 + 26 = 1433 < 1440 at 32k. The context-aware
        # headroom is what makes this distinction possible at all.
        base = {"PRIMARY_QUANT": "UD-Q4_K_XL", "KIMI_K3_NODE": "b200x8",
                "KIMI_K3_MAX_CTX": "1048576"}   # explicit: the default is now 128k
        self.assertEqual(bash_rc("kimi_k3_check_fits", self._with_fake_smi(self.B200, 8, base)), 1)
        at32k = {**base, "KIMI_K3_MAX_CTX": "32768"}
        self.assertEqual(bash_rc("kimi_k3_check_fits", self._with_fake_smi(self.B200, 8, at32k)), 0)

    def test_smaller_quant_rescues_an_undersized_node(self):
        # 5x H200 (705 GiB) can't hold the 802 GiB TARGET, but holds IQ1_S (553 + 26 = 579).
        # Both quants are named explicitly: the harness default moved to UD-IQ1_S, so
        # relying on it to supply the "too big" side silently inverted this test's premise.
        small = {"PRIMARY_QUANT": "UD-IQ1_S", "KIMI_K3_MAX_CTX": "32768"}
        big   = {"PRIMARY_QUANT": "UD-Q2_K_XL", "KIMI_K3_MAX_CTX": "32768"}
        self.assertEqual(bash_rc("kimi_k3_check_fits", self._with_fake_smi(self.H200, 5, small)), 0)
        self.assertEqual(bash_rc("kimi_k3_check_fits", self._with_fake_smi(self.H200, 5, big)), 1)

    def test_4x_h200_cannot_hold_even_iq1s(self):
        # 564 GiB vs 553 GiB weights leaves 11 GiB — less than the compute buffers alone.
        # Worth asserting: "553 < 564 so it fits" is the exact mistake the context-aware
        # headroom exists to prevent.
        env = self._with_fake_smi(self.H200, 4, {"PRIMARY_QUANT": "UD-IQ1_S",
                                                 "KIMI_K3_MAX_CTX": "32768"})
        self.assertEqual(bash_rc("kimi_k3_check_fits", env), 1)

    def test_cuda_visible_devices_narrows_gpu_count(self):
        env = self._with_fake_smi(self.H200, 8)
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        self.assertEqual(bash("gpu_count", env), "3")

    def test_detects_sm90(self):
        env = self._with_fake_smi(self.H200, 8)
        self.assertEqual(bash("detect_arch", env), "90")


class BaselineDryRunTest(unittest.TestCase):
    def _dry(self, *args):
        out = subprocess.run(
            ["bash", str(SCRIPTS / "kimi_k3_baseline.sh"), "--dry-run", *args],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        return out.stdout

    def test_dry_run_needs_no_gpu_or_weights(self):
        out = self._dry()
        self.assertIn("Kimi-K3-UD-IQ1_S-00001-of-00014.gguf", out)
        self.assertIn("--split-mode layer", out)
        self.assertIn("553 GiB + 29 GiB headroom", out)
        self.assertIn(PINNED_COMMIT, out)

    def test_longctx_probe_is_opt_in(self):
        # A 1M-token prefill is minutes per rep, so it must not be in the default sweep —
        # but every hardware milestone requires it, so it has to be one flag away.
        self.assertNotIn("1048576  (reps=1", self._dry())
        self.assertIn("1048576", self._dry("--longctx"))

    def test_node_flag_overrides_profile(self):
        out = self._dry("--node", "b300x4")
        self.assertIn("4x+ B300", out)
        self.assertIn(">=4 GPU(s)", out)

    def test_rejects_unknown_args(self):
        out = subprocess.run(
            ["bash", str(SCRIPTS / "kimi_k3_baseline.sh"), "--nope"],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(out.returncode, 2)


class GgufTokenizerTest(unittest.TestCase):
    """K3 ships a tiktoken vocab, not tokenizer.json, so prompt ids come from the GGUF."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fake_gguf = Path(self.tmp) / "model-00001-of-00014.gguf"
        self.fake_gguf.write_text("not a real gguf")
        self.fake_bin = Path(self.tmp) / "llama-tokenize"
        # Mimic llama-tokenize: logs on stderr, one bracketed id list on stdout, and
        # requires --stdin/--ids/--no-bos to be passed.
        self.fake_bin.write_text(textwrap.dedent("""\
            #!/usr/bin/env python3
            import sys
            a = sys.argv[1:]
            for need in ("--stdin", "--ids", "--no-bos"):
                if need not in a:
                    print("missing " + need, file=sys.stderr); sys.exit(3)
            text = sys.stdin.read()
            print("load: vocab_only", file=sys.stderr)
            ids = [100 + (len(w) % 7) for w in text.split()]
            print("[" + ", ".join(map(str, ids)) + "]")
        """))
        self.fake_bin.chmod(self.fake_bin.stat().st_mode | stat.S_IEXEC)
        self.corpus = Path(self.tmp) / "corpus.txt"
        self.corpus.write_text("alpha beta gamma delta\n\nepsilon zeta eta theta iota kappa\n\n"
                               "lambda mu nu xi omicron pi rho sigma tau\n")

    def _run(self, *extra):
        cmd = [sys.executable, str(SCRIPTS / "gen_eval_prompt.py"), "seed-1", "-",
               str(self.corpus), "--gguf", str(self.fake_gguf),
               "--llama-tokenize", str(self.fake_bin), *extra]
        out = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        return [int(t) for t in out.stdout.split()]

    def test_short_prompt_ids(self):
        ids = self._run()
        self.assertTrue(ids)
        self.assertTrue(all(100 <= i <= 106 for i in ids), ids[:10])

    def test_long_prompt_hits_exact_length(self):
        ids = self._run("--len", "500")
        self.assertEqual(len(ids), 500)

    def test_deterministic_for_a_seed(self):
        self.assertEqual(self._run("--len", "120"), self._run("--len", "120"))

    def test_missing_binary_is_a_clear_error(self):
        out = subprocess.run(
            [sys.executable, str(SCRIPTS / "gen_eval_prompt.py"), "s", "-", str(self.corpus),
             "--gguf", str(self.fake_gguf), "--llama-tokenize", "/nonexistent/llama-tokenize"],
            capture_output=True, text=True, env={**os.environ, "PATH": ""})
        self.assertNotEqual(out.returncode, 0)

    def test_missing_gguf_is_a_clear_error(self):
        out = subprocess.run(
            [sys.executable, str(SCRIPTS / "gen_eval_prompt.py"), "s", "-", str(self.corpus),
             "--gguf", "/nonexistent/model.gguf", "--llama-tokenize", str(self.fake_bin)],
            capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("no such GGUF", out.stderr + out.stdout)


class SplitDownloadTest(unittest.TestCase):
    def test_incomplete_shard_set_is_rejected(self):
        tmp = tempfile.mkdtemp()
        d = Path(tmp) / "UD-IQ1_S"
        d.mkdir(parents=True)
        # 2 of 14 present, and a stub `hf` that downloads nothing.
        for i in (1, 2):
            (d / f"Kimi-K3-UD-IQ1_S-{i:05d}-of-00014.gguf").write_text("x")
        binp = Path(tmp) / "bin"
        binp.mkdir()
        hf = binp / "hf"
        hf.write_text("#!/usr/bin/env bash\nexit 0\n")
        hf.chmod(hf.stat().st_mode | stat.S_IEXEC)
        env = {"PATH": f"{binp}:{os.environ['PATH']}"}
        rc = bash_rc(
            f'ensure_model_split repo "{tmp}" "*UD-IQ1_S*" '
            f'"UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf" 14', env)
        self.assertEqual(rc, 1)

    def test_complete_shard_set_passes(self):
        tmp = tempfile.mkdtemp()
        d = Path(tmp) / "UD-IQ1_S"
        d.mkdir(parents=True)
        for i in range(1, 15):
            (d / f"Kimi-K3-UD-IQ1_S-{i:05d}-of-00014.gguf").write_text("x")
        rc = bash_rc(
            f'ensure_model_split repo "{tmp}" "*UD-IQ1_S*" '
            f'"UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf" 14')
        self.assertEqual(rc, 0)

    def test_manifest_mismatch_is_fatal(self):
        tmp = tempfile.mkdtemp()
        d = Path(tmp) / "UD-IQ1_S"
        d.mkdir(parents=True)
        (d / "Kimi-K3-UD-IQ1_S-00001-of-00014.gguf").write_text("x")
        man = Path(tmp) / "m.sha256"
        man.write_text("deadbeef  Kimi-K3-UD-IQ1_S-00001-of-00014.gguf\n")
        rc = bash_rc(f'verify_model_split "{tmp}" '
                     f'"UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf" "{man}"')
        self.assertEqual(rc, 1)

    def test_absent_manifest_is_warn_only(self):
        tmp = tempfile.mkdtemp()
        d = Path(tmp) / "UD-IQ1_S"
        d.mkdir(parents=True)
        (d / "Kimi-K3-UD-IQ1_S-00001-of-00014.gguf").write_text("x")
        rc = bash_rc(f'verify_model_split "{tmp}" '
                     f'"UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf" "{tmp}/absent.sha256"')
        self.assertEqual(rc, 0)


class ArchConfigTest(unittest.TestCase):
    """Guard the numbers that are silently catastrophic if wrong. Sources:
    moonshotai/Kimi-K3 config.json and the fork's conversion/kimi_k3.py."""

    def setUp(self):
        self.text = (ROOT / "bench" / "configs" / "models" / "kimi_k3.yaml").read_text()

    def test_expert_count_exceeds_upstream_cap(self):
        # This is the reason the baseline must be the fork at all.
        self.assertIn("num_experts: 896", self.text)

    def test_hybrid_layer_split_sums_to_93(self):
        self.assertIn("num_hidden_layers: 93", self.text)
        self.assertIn("full_attention_layers: 24", self.text)
        self.assertIn("kda_layers: 69", self.text)

    def test_full_attn_layer_list_is_flagged_one_based(self):
        # conversion/kimi_k3.py uses `(il + 1) in full_attn_layers`; an off-by-one
        # here is silent garbage, so the key name must carry the convention.
        self.assertIn("full_attention_layers_1based:", self.text)

    def test_mla_kv_bytes_per_token(self):
        # 24 MLA layers * (kv_lora_rank 512 + qk_rope 64) * 2 B, K only (no V cache).
        self.assertIn("mla_kv_bytes_per_token: 27648", self.text)
        self.assertEqual(24 * (512 + 64) * 2, 27648)

    def test_kda_state_bytes(self):
        # n_embd_s = 128*128*96 ; n_embd_r = 3*(4-1)*96*128 ; f32
        self.assertEqual((128 * 128 * 96 + 3 * 3 * 96 * 128) * 4, 6733824)
        self.assertIn("kda_state_bytes_per_seq_per_layer: 6733824", self.text)

    def test_required_kv_keys_documented(self):
        for key in ("expert_latent_length: 3584", "block_size: 12",
                    "situ_beta: 4.0", "situ_linear_beta: 25.0"):
            self.assertIn(key, self.text)

    def test_targets_declare_no_native_runtime(self):
        tdir = ROOT / "bench" / "configs" / "targets"
        for node in ("h200x8", "b200x8", "b300x4"):
            target = (tdir / f"kimi_k3_ud_q2kxl_{node}.yaml").read_text()
            self.assertIn("sparkinfer_native_decode: false", target, node)
            # Every sparkinfer slot must be null, not a guess.
            self.assertNotIn("bs1_ctx128: 0", target, node)

    def test_hardware_profiles_exist_for_every_node(self):
        cdir = ROOT / "bench" / "configs"
        for f in ("h200_sxm.yaml", "b200_sxm.yaml", "b300.yaml"):
            self.assertTrue((cdir / f).exists(), f)
        # B300's arch is a guess and must say so, or a mislabelled result looks authoritative.
        self.assertIn("arch_verified: false", (cdir / "b300.yaml").read_text())


class ReferenceLockProvenanceTest(unittest.TestCase):
    """check_reference_lock.py is the CI job that makes the repo's central claim
    enforceable: a pinned baseline must trace to a recorded measurement, because
    downstream a hand-filled number is indistinguishable from a measured one."""

    CHECKER = SCRIPTS / "check_reference_lock.py"

    def _run(self, lock_text=None, results=None):
        """Run the checker against a temp lock + results dir. Returns (rc, stdout)."""
        tmp = Path(tempfile.mkdtemp())
        # The checker reads the real reference.lock, so exercise it by running in a copy
        # of the tree layout it expects.
        scripts = tmp / "bench" / "scripts"
        scripts.mkdir(parents=True)
        (tmp / "bench" / "results").mkdir(parents=True)
        (scripts / "reference.lock").write_text(
            lock_text if lock_text is not None
            else (SCRIPTS / "reference.lock").read_text())
        shutil.copy(self.CHECKER, scripts / self.CHECKER.name)
        rdir = tmp / "bench" / "results"
        # Seed with the REAL committed baseline JSONs unless the caller supplies its own
        # set. The shipped lock now pins a measured value (the first K3 reference), so a
        # provenance run against an empty dir would fail for the right reason at the wrong
        # time — it would be asserting "nothing is measured yet", which is no longer true.
        real = ROOT / "bench" / "results"
        if real.is_dir():
            for f in real.glob("kimi_k3_*_baseline_*.json"):
                shutil.copy(f, rdir / f.name)
        for name, payload in (results or {}).items():
            (rdir / name).write_text(json.dumps(payload))
        out = subprocess.run(
            [sys.executable, str(scripts / self.CHECKER.name), "--results-dir", str(rdir)],
            capture_output=True, text=True)
        return out.returncode, out.stdout + out.stderr

    def test_shipped_lock_is_clean(self):
        """Every pinned slot in the shipped lock traces to a committed measurement.

        This used to assert "0 pinned" — correct while no K3 baseline existed. The first
        real reference (llama.cpp on 8x H200, UD-IQ1_S) is now pinned, so the invariant
        that matters is PROVENANCE, not emptiness: rc == 0 means every pinned value is
        backed by a results JSON in the tree."""
        rc, out = self._run()
        self.assertEqual(rc, 0, out)
        self.assertIn("K3 baseline slots", out)

    def test_hand_filled_baseline_is_rejected(self):
        lock = (SCRIPTS / "reference.lock").read_text().replace(
            'KIMI_K3_H200X8_LLAMA_4K="${KIMI_K3_H200X8_LLAMA_4K:-0}"',
            'KIMI_K3_H200X8_LLAMA_4K="${KIMI_K3_H200X8_LLAMA_4K:-311.20}"')
        rc, out = self._run(lock)
        self.assertEqual(rc, 1, out)
        self.assertIn("NO results file records", out)

    def test_backed_baseline_is_accepted(self):
        lock = (SCRIPTS / "reference.lock").read_text().replace(
            'KIMI_K3_H200X8_LLAMA_4K="${KIMI_K3_H200X8_LLAMA_4K:-0}"',
            'KIMI_K3_H200X8_LLAMA_4K="${KIMI_K3_H200X8_LLAMA_4K:-311.20}"')
        rc, out = self._run(lock, {
            "kimi_k3_UD-Q2_K_XL_h200x8_baseline_20260730T000000Z.json": {
                "node_profile": "h200x8",
                "contexts": {"4096": {"decode_tps": 311.2013, "prefill_pp": 8000.0}},
            }})
        self.assertEqual(rc, 0, out)

    def test_mismatched_value_is_rejected(self):
        lock = (SCRIPTS / "reference.lock").read_text().replace(
            'KIMI_K3_H200X8_LLAMA_4K="${KIMI_K3_H200X8_LLAMA_4K:-0}"',
            'KIMI_K3_H200X8_LLAMA_4K="${KIMI_K3_H200X8_LLAMA_4K:-311.20}"')
        rc, out = self._run(lock, {
            "kimi_k3_UD-Q2_K_XL_h200x8_baseline_20260730T000000Z.json": {
                "node_profile": "h200x8",
                "contexts": {"4096": {"decode_tps": 222.0}},
            }})
        self.assertEqual(rc, 1, out)
        self.assertIn("does not match any measurement", out)

    def test_wrong_node_does_not_satisfy(self):
        # A B200 measurement must not back an H200 pin — that is the whole point of the
        # per-node prefixes.
        lock = (SCRIPTS / "reference.lock").read_text().replace(
            'KIMI_K3_H200X8_LLAMA_4K="${KIMI_K3_H200X8_LLAMA_4K:-0}"',
            'KIMI_K3_H200X8_LLAMA_4K="${KIMI_K3_H200X8_LLAMA_4K:-311.20}"')
        rc, out = self._run(lock, {
            "kimi_k3_UD-Q2_K_XL_b200x8_baseline_20260730T000000Z.json": {
                "node_profile": "b200x8",
                "contexts": {"4096": {"decode_tps": 311.2013}},
            }})
        self.assertEqual(rc, 1, out)

    def test_prefill_and_decode_are_not_interchangeable(self):
        # A _PP pin needs a prefill_pp measurement, not a decode one at the same context.
        lock = (SCRIPTS / "reference.lock").read_text().replace(
            'KIMI_K3_H200X8_LLAMA_4K_PP="${KIMI_K3_H200X8_LLAMA_4K_PP:-0}"',
            'KIMI_K3_H200X8_LLAMA_4K_PP="${KIMI_K3_H200X8_LLAMA_4K_PP:-8000.00}"')
        rc, _ = self._run(lock, {
            "kimi_k3_UD-Q2_K_XL_h200x8_baseline_x.json": {
                "node_profile": "h200x8",
                "contexts": {"4096": {"decode_tps": 8000.0}},
            }})
        self.assertEqual(rc, 1)
        rc, _ = self._run(lock, {
            "kimi_k3_UD-Q2_K_XL_h200x8_baseline_x.json": {
                "node_profile": "h200x8",
                "contexts": {"4096": {"prefill_pp": 8000.0}},
            }})
        self.assertEqual(rc, 0)


class PinAuditTest(unittest.TestCase):
    """audit_baseline_pins.py watches the two external things the baseline depends on and
    cannot control. Only its parsers are tested here — the live checks need network."""

    AUDIT = SCRIPTS / "audit_baseline_pins.py"

    def test_offline_mode_makes_no_requests(self):
        out = subprocess.run([sys.executable, str(self.AUDIT), "--offline"],
                             capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("refs/pull/48/head", out.stdout)
        self.assertIn(PINNED_COMMIT, out.stdout)
        for q in ("UD-Q2_K_XL", "UD-IQ1_S", "UD-Q8_K_XL"):
            self.assertIn(q, out.stdout)

    def test_shard_pins_parse_from_the_shell_source(self):
        # The audit reads kimi_k3_n_shards()'s case arms; if that function is restructured
        # the audit must not silently start comparing an empty set.
        sys.path.insert(0, str(SCRIPTS))
        try:
            import audit_baseline_pins as audit
        finally:
            sys.path.pop(0)
        pins = audit.pinned_shards((SCRIPTS / "_kimi_k3.sh").read_text())
        self.assertEqual(pins, {"UD-IQ1_S": 14, "UD-IQ1_M": 15, "UD-IQ2_XXS": 16,
                                "UD-Q2_K_XL": 19, "UD-Q4_K_XL": 32, "UD-Q8_K_XL": 34})

    def test_lock_vars_parse(self):
        sys.path.insert(0, str(SCRIPTS))
        try:
            import audit_baseline_pins as audit
        finally:
            sys.path.pop(0)
        text = (SCRIPTS / "reference.lock").read_text()
        self.assertEqual(audit.lock_var("KIMI_K3_LLAMACPP_COMMIT", text), PINNED_COMMIT)
        self.assertEqual(audit.lock_var("KIMI_K3_LLAMACPP_REF", text), PINNED_REF)
        self.assertEqual(audit.lock_var("KIMI_K3_LLAMACPP_REPO", text), PINNED_REPO)


class CiWorkflowTest(unittest.TestCase):
    """The inherited CI linted four hand-listed files and never touched the K3 harness.
    These assert the replacement actually covers this repo."""

    WF = ROOT / ".github" / "workflows"

    def test_superseded_workflows_are_gone(self):
        for gone in ("eval-policy.yml", "rtx5090-required.yml", "build-attested-binaries.yml"):
            self.assertFalse((self.WF / gone).exists(), gone)

    def test_replacements_exist(self):
        for f in ("ci.yml", "build-gate.yml", "node-attestation.yml", "pin-audit.yml"):
            self.assertTrue((self.WF / f).exists(), f)

    def test_ci_has_the_five_jobs(self):
        import yaml
        d = yaml.safe_load((self.WF / "ci.yml").read_text())
        self.assertEqual(sorted(d["jobs"]), ["configs", "lock", "plans", "python", "shell"])

    def test_ci_lints_every_script_not_a_hand_list(self):
        # The specific failure mode being prevented: a hand-maintained file list meant
        # server/scripts/bench_api_vs_native.sh was broken by CRLF for its whole life.
        text = (self.WF / "ci.yml").read_text()
        self.assertIn("git ls-files '*.sh'", text)
        self.assertIn("No CRLF in source", text)

    def test_no_workflow_targets_the_upstream_repo(self):
        # close-stale-prs.yml used to hardcode REPO=gittensor-ai-lab/sparkinfer, so a
        # scheduled job here would have reached into another repository.
        for f in sorted(self.WF.glob("*.yml")):
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                self.assertNotIn("gittensor-ai-lab/sparkinfer\n", line + "\n",
                                 f"{f.name}:{i} targets the upstream repo")

    def test_build_gate_targets_milestone_arches_not_sm120(self):
        """sm_120 is Blackwell consumer/edge and nothing here runs on it — that was the
        inherited workflow compile-checking an irrelevant architecture.

        sm_100 (M2 · B200) was removed by request: this repo is Hopper-only in practice and
        there is no Blackwell hardware, so the job asserted a scope the project does not
        have. It is a real loss of coverage — the link chain reaches si_k3, so it did compile
        the K3 kernels for Blackwell — and the workflow says so where the entry used to be,
        so restoring it for M2 is a one-line change rather than an archaeology exercise."""
        text = (self.WF / "build-gate.yml").read_text()
        self.assertIn('arch: "90"', text)
        self.assertNotIn('arch: "120"', text)
        active = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
        self.assertNotIn('arch: "100"', active, "sm_100 was removed from the CI gate")
        self.assertIn("sm_100", text, "removing it silently loses why it went")

    def test_node_attestation_never_closes(self):
        # Deliberate severity change from the inherited rtx5090-required gate.
        text = (self.WF / "node-attestation.yml").read_text()
        self.assertIn("needs-node-run", text)
        self.assertNotIn("state: 'closed'", text)

    @unittest.skipUnless(shutil.which("node"), "node not available")
    def test_node_attestation_scan_is_scoped_to_the_node_run_section(self):
        """A body-wide ticked-checkbox scan matched the PR-template Checklist boilerplate
        '- [x] `...kimi_k3_baseline.sh --node h200x8 --dry-run` resolves' -- ticked on
        nearly every PR and containing 'h200' -- so it read as 'attested on 8x H200' while
        the real Node run box sat unticked. #74 cleared its needs-node-run label this way
        with no node run behind it (#78 fixed it).

        Tested by running the REAL functions extracted from the workflow's github-script
        block, against a body shaped like the PR template -- not a paraphrase that can
        drift. The fixture keeps the exact colliding Checklist line."""
        import yaml as _y, tempfile, textwrap
        wf = _y.safe_load((self.WF / "node-attestation.yml").read_text())
        script = wf["jobs"]["gate"]["steps"][0]["with"]["script"]
        self.assertIn("function nodeRunSection", script,
                      "the section scoper is gone — the body-wide scan is back")
        fns = script[script.index("const NODES"):script.index("const { owner, repo }")]

        fixture = textwrap.dedent("""\
            ## Summary
            x
            ## Node run
            - [ ] Tested on **8× H200** (`sm_90`)
            ## Checklist
            - [x] `bench/scripts/kimi_k3_baseline.sh --node h200x8 --dry-run` resolves
        """)
        driver = fns + textwrap.dedent("""
            const cases = [
              tickedNodes(process.argv[2]),                                   // boilerplate only
              tickedNodes(process.argv[2].replace('- [ ] Tested', '- [x] Tested')),  // real tick
              tickedNodes(process.argv[2].replace('## Node run', '## Runs')), // heading gone
            ];
            console.log(JSON.stringify(cases));
        """)
        f = tempfile.NamedTemporaryFile("w", suffix=".js", delete=False)
        f.write(driver); f.close()
        out = subprocess.run(["node", f.name, fixture], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        boilerplate_only, real_tick, no_heading = json.loads(out.stdout)
        self.assertEqual(boilerplate_only, [],
                         "the ticked Checklist boilerplate attested a node again (#74)")
        self.assertEqual(real_tick, ["8x H200 (sm_90)"],
                         "a genuinely ticked Node run box must still attest")
        self.assertEqual(no_heading, [],
                         "no Node run section must fail closed, not fall back to the body")

    def test_sensitive_paths_no_longer_guards_the_whole_harness(self):
        # bench/scripts/ IS the deliverable here; guarding all of it would gate every
        # contribution the repo exists to receive.
        text = (self.WF / "sensitive-paths-guard.yml").read_text()
        sensitive = [l for l in text.splitlines() if l.strip().startswith("SENSITIVE=")]
        self.assertEqual(len(sensitive), 1, sensitive)
        pattern = sensitive[0]
        # The narrow, verdict-determining paths must be there (the dot is regex-escaped).
        self.assertIn(r"reference\.lock", pattern)
        self.assertIn("bench/results/", pattern)
        # ...and the blanket bench/scripts/ alternative must NOT be.
        self.assertNotIn("bench/scripts/|", pattern)
        # Prove it behaves. The rule is NOT "narrow" for its own sake -- it is "everything
        # that determines a verdict, and nothing else". Anything kimi_k3_eval.sh invokes to
        # produce a number is part of the verdict: guarding eval.sh while leaving
        # baseline.sh (which emits the tok/s) or _kimi_k3.sh (which it sources) writable is
        # a bypass, not a narrower guard. An earlier revision of this test asserted those
        # two were open; that was wrong and is corrected here.
        import re as _re
        rx = _re.compile(pattern.split("'")[1])
        for guarded in ("bench/scripts/reference.lock", "bench/results/kimi_k3_x.json",
                        "bench/scripts/label.py", "bench/scripts/kimi_k3_eval.sh",
                        "bench/scripts/kimi_k3_baseline.sh", "bench/scripts/_kimi_k3.sh",
                        "bench/scripts/kimi_k3_attest.py"):
            self.assertTrue(rx.search(guarded), f"trust anchor left writable: {guarded}")
        # ...while the harness at large stays open to contributors.
        for open_path in ("bench/scripts/kimi_k3_run.sh", "bench/scripts/bench.sh",
                          "bench/scripts/probe_max_ctx.sh", "kernels/csrc/cuda/x.cu"):
            self.assertFalse(rx.search(open_path), f"needlessly guarded: {open_path}")

    def test_guard_covers_everything_the_eval_invokes(self):
        """Closes the class, not the instance. The guard is a hand-maintained list while
        the eval's call graph moves independently -- that is exactly how baseline.sh ended
        up writable. This walks what kimi_k3_eval.sh actually invokes and fails if any of
        it escapes the guard, so the next helper added to the chain cannot silently
        reopen the hole."""
        import re as _re
        wf = (ROOT / ".github/workflows/sensitive-paths-guard.yml").read_text()
        rx = _re.compile([l for l in wf.splitlines() if "SENSITIVE=" in l][0].split("'")[1])
        src = (ROOT / "bench/scripts/kimi_k3_eval.sh").read_text()
        invoked = set(_re.findall(r"bench/scripts/([A-Za-z0-9_.]+\.(?:sh|py))", src))
        invoked |= {m for m in _re.findall(r"\b(_[a-z0-9_]+\.sh)", src)}
        missing = [f for f in sorted(invoked)
                   if f != "kimi_k3_run.sh" and not rx.search(f"bench/scripts/{f}")]
        self.assertEqual(missing, [], f"kimi_k3_eval.sh invokes unguarded files: {missing}")

    def test_check_run_names_are_unique_across_workflows(self):
        """A branch-protection required context is matched by check-run NAME, and a check
        run is named by the job's `name:` if set, else by the job id. Two jobs in different
        workflows sharing a name make that context ambiguous: protection cannot tell which
        one satisfied it, so requiring the trust-bearing guard could be signed off by an
        unrelated advisory job. copycat-guard and sensitive-paths-guard both used the bare
        job id `guard` and both trigger on pull_request_target, so they genuinely coexisted
        on every PR commit."""
        import yaml as _y
        seen = {}
        for wf in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
            for jid, job in (_y.safe_load(wf.read_text()).get("jobs") or {}).items():
                seen.setdefault((job or {}).get("name") or jid, []).append(f"{wf.name}:{jid}")
        dupes = {n: w for n, w in seen.items() if len(w) > 1}
        self.assertEqual(dupes, {}, f"ambiguous required-context names: {dupes}")

    def test_required_checks_always_run(self):
        """The other half of the footgun. A required check that never produces a check run
        blocks merge forever, and a job-level `if:` is the usual cause. Every job named
        here is one we require in branch protection, so each must be unconditional: no
        job-level `if:`, and a trigger that fires on pull requests from forks. Adding an
        `if:` to one of these later would silently wedge the merge queue -- this fails
        instead."""
        import yaml as _y
        required = {
            "sensitive-paths-guard.yml": "guard",
            "ci.yml": ["shell", "python", "configs", "plans", "lock"],
        }
        for fname, jids in required.items():
            wf = ROOT / ".github" / "workflows" / fname
            doc = _y.safe_load(wf.read_text())
            on = doc.get(True) or doc.get("on")
            triggers = set(on if isinstance(on, (list, dict)) else [on])
            self.assertTrue(
                triggers & {"pull_request", "pull_request_target"},
                f"{fname} is required but does not run on pull requests: {triggers}")
            for jid in ([jids] if isinstance(jids, str) else jids):
                job = doc["jobs"][jid]
                self.assertNotIn(
                    "if", job,
                    f"{fname}:{jid} is a REQUIRED check but is conditional -- a skipped "
                    f"required check cannot be satisfied and blocks merge")

    def test_codeowners_matches_the_sensitive_guard(self):
        """CODEOWNERS says in its own header to keep its list in sync with SENSITIVE in
        sensitive-paths-guard.yml, and that instruction had already been violated: the
        guard was widened to cover kimi_k3_baseline.sh / _kimi_k3.sh / kimi_k3_attest.py
        and CODEOWNERS was not. Divergence is quietly bad in both directions -- a path in
        the guard but not CODEOWNERS gets blocked with no reviewer auto-requested, and a
        path in CODEOWNERS but not the guard looks protected while being freely editable.
        A comment cannot enforce itself, so this does."""
        import re as _re
        wf = (ROOT / ".github/workflows/sensitive-paths-guard.yml").read_text()
        pattern = [l for l in wf.splitlines() if "SENSITIVE=" in l][0].split("'")[1]
        # the explicit bench/scripts/(a|b|c)$ alternation -- the named scoring-path files
        group = _re.search(r"bench/scripts/\(([^)]*)\)", pattern)
        self.assertIsNotNone(group, "guard no longer names bench/scripts files explicitly")
        owners = (ROOT / ".github/CODEOWNERS").read_text()
        for alt in group.group(1).split("|"):
            name = alt.replace("\\", "").rstrip("$")
            if ".*" in name:                       # a glob like kimi_k3_.*\.sha256
                stem = name.split(".*")[0]
                self.assertIn(f"/bench/scripts/{stem}", owners,
                              f"guard covers {name} but CODEOWNERS has no matching entry")
            else:
                self.assertIn(f"/bench/scripts/{name}", owners,
                              f"guard covers bench/scripts/{name} but CODEOWNERS does not")
    def test_guard_covers_the_accuracy_answer_key(self):
        """bench/refdata/hello.spkl IS the accuracy gate. kimi_k3_eval.sh scores the build's
        logits against it, so a PR able to rewrite it can hand itself top1=1.0 / kl=0.0 and
        walk through the correctness gate that is supposed to make a speedup honest. It was
        unguarded. _common.sh (build flags) and _eval_speed.sh (parses the tok/s) are the
        same shape of hole on the speed side."""
        import re as _re
        wf = (ROOT / ".github/workflows/sensitive-paths-guard.yml").read_text()
        rx = _re.compile([l for l in wf.splitlines() if "SENSITIVE=" in l][0].split("'")[1])
        for p in ("bench/refdata/hello.spkl", "bench/refdata/hello.ids",
                  "bench/scripts/_common.sh", "bench/scripts/_eval_speed.sh"):
            self.assertTrue(rx.search(p), f"scoring input left writable by contributors: {p}")

    def test_provenance_cannot_overwrite_the_verdict(self):
        """label.py merges its optional 7th provenance arg into the result AFTER computing
        the verdict. Unfiltered, a provenance blob carrying {"label": "XL"} silently becomes
        the tier. The inline comment called it non-scoring; it was not."""
        import subprocess as sp, json as _j
        env = {**os.environ, "SPARKINFER_DIFFICULTY_REF": "18.321"}   # K3's pinned llama ref
        out = sp.run([sys.executable, str(ROOT / "bench/scripts/label.py"),
                      "4.0", "3.55", "0", "1.0", "0.001", "abc123",
                      _j.dumps({"label": "XL", "pass": True, "clock_mhz": 1980})],
                     capture_output=True, text=True, check=True, env=env).stdout
        res = _j.loads(out.split("RESULT_JSON ", 1)[1])
        self.assertNotEqual(res["label"], "XL", "provenance overwrote the computed tier")
        self.assertIn("label", res.get("provenance_rejected", []))
        self.assertEqual(res.get("clock_mhz"), 1980, "legitimate provenance was dropped")

    def test_verify_strict_actually_uses_its_validator(self):
        """verify_strict(receipt, validator) accepted a validator and never referenced it,
        so callers passing ReceiptValidator(r) got nothing: 'the receipt verifies' meant
        five fields were non-empty. Asserted structurally via AST so a future refactor that
        drops the call fails here rather than silently restoring a decorative parameter."""
        import ast as _ast
        src = (ROOT / "eval/polaris/verify.py").read_text()
        fn = [n for n in _ast.walk(_ast.parse(src))
              if isinstance(n, _ast.FunctionDef) and n.name == "verify_strict"][0]
        names = {x.id for x in _ast.walk(fn) if isinstance(x, _ast.Name)}
        self.assertIn("validator", names, "verify_strict ignores its validator parameter")
        calls = {_ast.unparse(c.func) for c in _ast.walk(fn) if isinstance(c, _ast.Call)}
        self.assertIn("validator.verify", calls, "validator.verify() is never invoked")

    def _verdict_script(self):
        """Extract the verdict heredoc out of eval-label.yml so the test exercises the code
        that actually runs in CI, not a copy of it that can drift."""
        import re as _re, tempfile
        txt = (ROOT / ".github/workflows/eval-label.yml").read_text().split("\n")
        i = [k for k, l in enumerate(txt) if 'python3 - "$PAYLOAD" "$PR_HEAD_SHA"' in l][0]
        j = [k for k in range(i + 1, len(txt)) if txt[k].strip() == "PY"][0]
        f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        f.write("\n".join(l[10:] for l in txt[i + 1:j])); f.close()
        return f.name

    def _run_verdict(self, payload, head):
        import subprocess as sp, json as _j
        r = sp.run([sys.executable, self._verdict_script(), _j.dumps(payload), head],
                   capture_output=True, text=True, cwd=str(ROOT))
        out = dict(l.split("=", 1) for l in r.stdout.strip().split("\n") if "=" in l)
        return r.returncode, out, r.stderr

    def test_tier_basis_comes_from_the_lock_not_the_payload(self):
        """frontier_tps and llama_ref are the denominator and the significance basis, so a
        payload that supplies them sets its own tier: understate the frontier and any speed
        looks like a large win. They must come from reference.lock in the trusted base
        checkout. Asserted by running the attack, not by reading the code."""
        head = "8757caf10683484582b346663ee8944ac66e2e06"
        base = {"tps": 4.2, "top1": 1.0, "kl": 0.001, "commit": head[:7]}
        rc_h, honest, _ = self._run_verdict(base, head)
        rc_a, attack, err = self._run_verdict({**base, "frontier_tps": 0.5}, head)
        self.assertEqual((rc_h, rc_a), (0, 0))
        self.assertEqual(honest["label"], attack["label"],
                         "understating frontier_tps changed the tier")
        self.assertEqual(honest["pct"], attack["pct"])
        self.assertIn("using the pinned value", err)

    def test_result_must_belong_to_this_pr(self):
        """Without binding, a RESULT_JSON and its receipt measured on any branch label any
        PR -- a valid receipt becomes a reusable token."""
        head = "8757caf10683484582b346663ee8944ac66e2e06"
        rc, _, err = self._run_verdict(
            {"tps": 4.2, "top1": 1.0, "kl": 0.001, "commit": "deadbeef"}, head)
        self.assertEqual(rc, 1, "a result from another commit was accepted")
        self.assertIn("is not this PR's head", err)

    def test_verdict_step_emits_only_key_value_pairs(self):
        """stdout of that heredoc IS $GITHUB_OUTPUT. A stray ::notice:: line corrupts the
        step outputs, which is how the annotations were first written."""
        head = "8757caf10683484582b346663ee8944ac66e2e06"
        import subprocess as sp, json as _j
        r = sp.run([sys.executable, self._verdict_script(),
                    _j.dumps({"tps": 4.2, "top1": 1.0, "kl": 0.001, "commit": head[:7]}), head],
                   capture_output=True, text=True, cwd=str(ROOT))
        for line in [l for l in r.stdout.strip().split("\n") if l]:
            self.assertNotIn("::", line, f"annotation leaked into GITHUB_OUTPUT: {line}")
            self.assertIn("=", line)

    def test_cap_rechecks_when_a_draft_becomes_ready(self):
        """Drafts do not count toward the open-PR cap, which is right -- parking an idea is
        not something to auto-close. But it makes the cap trivially evadable unless becoming
        ready re-checks: open the first PR as a DRAFT, open a second normally (the cap sees
        one non-draft and passes), then mark the draft ready. That fires ready_for_review,
        not opened, so nothing re-ran and the author holds two.

        widecloud did exactly this with #52 and #57. copycat-guard already listened for
        ready_for_review for the same reason -- the pattern was known and this workflow
        simply never got it."""
        import yaml as _y
        cap_yaml = _y.safe_load((ROOT / ".github/workflows/cap-open-prs.yml").read_text())
        types = cap_yaml[True]["pull_request_target"]["types"]
        self.assertIn("ready_for_review", types,
                      "a draft marked ready never re-checks the cap")
        for t in ("opened", "reopened"):
            self.assertIn(t, types)

    def test_cap_keeps_the_measured_pr_not_merely_the_oldest(self):
        """Node time is the scarce resource. A PR carrying an eval:* tier has had ~40 minutes
        of 8x H200 spent on it and holds a live number against the CURRENT frontier, so
        closing it in favour of an unmeasured sibling discards that and forces a re-run.

        The old rule kept the oldest unconditionally -- and its header comment claimed the
        opposite of what its code did, so neither reading was reliable. Age now decides only
        among equals. eval:none does not count as measured: it means the gain was inside the
        significance gate, so there is no tier to protect."""
        cap = (ROOT / ".github/workflows/cap-open-prs.yml").read_text()
        self.assertIn("scored(b) - scored(a)", cap, "measured PRs must sort first")
        self.assertIn("l.name !== 'eval:none'", cap,
                      "eval:none is not a tier worth protecting")
        self.assertNotIn("Keep the MAX oldest", cap,
                         "the stale comment contradicted the code")

    def test_destructive_workflows_skip_drafts(self):
        """A draft is work someone has explicitly marked unfinished. CI running on it is
        useful; closing it, denylisting its author, or attaching a payout tier to it is not.
        These three took destructive action on drafts: cap-open-prs auto-closed them,
        copycat-guard could permanently denylist the author of one, and eval-label would
        label one. The bot had a draft gate all along -- but the bot does not run, so it
        protected nothing."""
        import yaml as _y
        wf = ROOT / ".github" / "workflows"
        cap = (wf / "cap-open-prs.yml").read_text()
        self.assertIn("!p.draft", cap, "cap-open-prs counts drafts toward the cap")
        cop = _y.safe_load((wf / "copycat-guard.yml").read_text())
        self.assertIn("draft != true", str(cop["jobs"]["guard"].get("if", "")),
                      "copycat-guard can denylist the author of a draft")
        self.assertIn("ready_for_review", str(cop.get(True) or cop.get("on")),
                      "a PR opened as a draft would never be rescanned once ready")
        self.assertIn("IS_DRAFT", (wf / "eval-label.yml").read_text(),
                      "/eval would attach a payout tier to a draft")

    def test_qwen_bot_refuses_to_run_against_k3(self):
        """eval/pr_eval_bot.py is the inherited Qwen evaluator and contains zero K3 code. Its
        greenlight matches the literal substring "5090", which the K3 template cannot
        contain, and the unchecked-box gate then CLOSES the PR -- so pointing it at this repo
        auto-closes every K3 pull request. A commented-out crontab line is not a safe place
        to keep that. Asserted by running it, including the exit code, because main() was
        called bare and swallowed its own return value."""
        import subprocess as sp
        for bot in ("eval/pr_eval_bot.py", "eval/pr_dflash_bot.py"):
            path = ROOT / bot
            if not path.exists():
                continue
            r = sp.run([sys.executable, str(path), "--repo", "gittensor-ai-lab/sparkinfer-k3",
                        "--dry-run"], capture_output=True, text=True, cwd=str(ROOT))
            self.assertEqual(r.returncode, 2, f"{bot} did not refuse a -k3 repo (exit "
                                              f"{r.returncode}); output: {r.stdout[-300:]}")
            self.assertIn("refusing to run", r.stderr)

    def test_automerge_does_not_silently_bypass_protection(self):
        """Enabling auto-merge must not also mean 'and bypass branch protection'. The admin
        escalation defaulted to on, so a bot run under an admin token would have walked
        through the required-review rule."""
        for bot in ("eval/pr_eval_bot.py", "eval/pr_dflash_bot.py"):
            path = ROOT / bot
            if path.exists():
                self.assertNotIn('SPARKINFER_AUTOMERGE_ADMIN", "1"', path.read_text(),
                                 f"{bot} defaults to bypassing branch protection")

    def test_k3_bot_build_configuration(self):
        """Two bugs found by running the bot against the node, not by reading it.

        kimi_k3_tp_bench refuses to run sharded without a real collective, so a build
        configured without -DSPARKINFER_TP=ON fails at eval time with nothing pointing back
        at configure as the cause. And CMake emits to build/runtime/, so pointing
        SPARKINFER_BUILD at build/ makes the harness report "not built" immediately after a
        build that in fact succeeded."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn("-DSPARKINFER_TP=ON", src, "bot builds without NCCL collectives")
        self.assertIn("build/runtime", src, "bot points SPARKINFER_BUILD at the wrong dir")

    def test_k3_bot_does_not_post_a_verdict(self):
        """The bot runs the harness from the PR's OWN checkout, so any verdict field it
        reports was computed against whatever reference.lock that branch carries. On a
        branch predating the quant-qualified lookup that resolves to frontier=0 and labels
        everything BASELINE -- which then makes eval-label.yml hard-error 'reported label
        != re-derived' on a genuinely good run. Observed on PR #25: measured 9.56 tok/s,
        payload said BASELINE, main's lock derives XL.

        Measurements come from the node; the verdict is derived on the protected side.
        Posting a frontier at all would also reopen the substitution #35 closed."""
        import importlib.util as _u
        spec = _u.spec_from_file_location("k3bot", ROOT / "eval/k3_eval_bot.py")
        bot = _u.module_from_spec(spec)
        saved, sys.argv = sys.argv, ["k3bot"]
        try:
            spec.loader.exec_module(bot)
        finally:
            sys.argv = saved
        for field in ("label", "frontier_tps", "speed_label", "pass"):
            self.assertIn(field, bot.DERIVED_BY_CI, f"bot would post CI-derived {field}")
        import io, contextlib, json as _j
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bot.post("r", 1, {"commit": "abc1234", "tps": 9.56, "top1": 1.0, "kl": 0.004,
                              "label": "BASELINE", "frontier_tps": 0.0}, True)
        payload = _j.loads(buf.getvalue().split("RESULT_JSON ", 1)[1].split("\n")[0])
        self.assertNotIn("label", payload)
        self.assertNotIn("frontier_tps", payload)
        self.assertEqual(payload["tps"], 9.56, "the measurement itself must survive")

    def _bot(self):
        import importlib.util as _u
        spec = _u.spec_from_file_location("k3bot", ROOT / "eval/k3_eval_bot.py")
        m = _u.module_from_spec(spec)
        saved, sys.argv = sys.argv, ["k3bot"]
        try:
            spec.loader.exec_module(m)
        finally:
            sys.argv = saved
        return m

    def test_bot_comment_stays_machine_readable(self):
        """The comment got a human-readable table, which is exactly the change that can
        silently break the pipeline: eval-label.yml gates on
        startsWith(comment.body, '/eval') and then sed-scrapes the JSON off ONE line. Format
        it wrong and the workflow simply never fires -- no error, no label, no clue."""
        import io, contextlib, re as _re, json as _j
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._bot().post("r", 1, {"commit": "abc1234", "tps": 9.55, "top1": 1.0,
                                      "kl": 0.004, "label": "XL"}, True)
        body = buf.getvalue().split("---\n", 1)[1]
        self.assertTrue(body.startswith("/eval"), "workflow trigger requires /eval first")
        m = _re.match(r".*RESULT_JSON\s*(\{.*\}).*", body.split("\n")[0])
        self.assertIsNotNone(m, "the payload is no longer on one scrapeable line")
        self.assertNotIn("label", _j.loads(m.group(1)), "bot posted a CI-derived field")

    def test_merge_first_picks_the_largest_measured_gain(self):
        """Two PRs in one round share a frontier, so faster genuinely is the larger gain.
        Whichever merges moves the frontier, and the rest must be re-measured against it --
        which is why exactly one carries the label."""
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._bot().mark_merge_first(
                "r", [(20, {"tps": 8.17}), (25, {"tps": 9.55})], True)
        out = buf.getvalue()
        self.assertIn("--add-label merge-first on #25", out)
        self.assertIn("--remove-label merge-first on #20", out)

    def test_admin_merge_is_reachable_only_through_the_guards(self):
        """The bot may now bypass branch protection, by explicit operator choice. What must
        stay true is that the bypass is reachable from exactly ONE place -- merge_winner,
        which runs merge_blockers first -- so a future caller cannot acquire it by adding a
        gh merge call somewhere else."""
        import ast as _ast
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        tree = _ast.parse(src)
        admin_calls = []
        for fn in _ast.walk(tree):
            if not isinstance(fn, _ast.FunctionDef):
                continue
            for node in _ast.walk(fn):
                if isinstance(node, _ast.Call) and getattr(node.func, "id", "") == "gh":
                    args = [a.value for a in _ast.walk(node)
                            if isinstance(a, _ast.Constant) and isinstance(a.value, str)]
                    if "--admin" in args:
                        admin_calls.append(fn.name)
        self.assertEqual(admin_calls, ["merge_winner"],
                         f"--admin reachable from {admin_calls}, not just merge_winner")
        body = src[src.index("def merge_winner"):src.index("def sync_rebase_labels")]
        self.assertLess(body.index("merge_blockers"), body.index("--admin"),
                        "merge_winner must check the guards before merging")

    def test_stale_and_unknown_merge_states_block(self):
        """BEHIND reads as harmless -- the branch merely trails main -- but it trails main
        because something merged, which is exactly when the frontier moved, so its tier was
        computed against a baseline that no longer exists.

        UNKNOWN is not "fine" either, it is "GitHub has not finished computing
        mergeability". #20 reported UNKNOWN seconds after #25 merged while actually being
        DIRTY, and an earlier denylist treated that as clear.

        BLOCKED is the one that is different in kind. It is a PERMISSION gate -- the required
        approving review has not happened -- and bypassing that is exactly what --admin is
        for. Refusing on it made --merge-admin unable to merge anything at all, because a
        repo that requires review leaves every PR sitting at BLOCKED forever. So admin may
        pass BLOCKED and must still refuse BEHIND, DIRTY and UNKNOWN: admin rights bypass
        permission, not content."""
        bot = self._bot()
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn("MERGE_STATES", src,
                      "merge states must be allowlisted, not denylisted")
        self.assertEqual(bot.MERGE_STATES["strict"], ("CLEAN", "UNSTABLE"))
        for st in ("BEHIND", "UNKNOWN", "DIRTY", "BLOCKED"):
            self.assertIn(f'"{st}"', src, f"{st} must be handled explicitly")
            self.assertNotIn(st, bot.MERGE_STATES["strict"],
                             f"{st} must not be mergeable without a human")
        # admin bypasses the permission gate only
        self.assertIn("BLOCKED", bot.MERGE_STATES["admin"],
                      "--merge-admin exists to bypass the review requirement; refusing on "
                      "BLOCKED makes it a no-op on every repo that requires review")
        for st in ("BEHIND", "DIRTY", "UNKNOWN"):
            self.assertNotIn(st, bot.MERGE_STATES["admin"],
                             f"admin rights do not make {st} safe — that is content, not "
                             f"permission")

    def test_bot_grades_with_the_protected_harness(self):
        """CONTRIBUTING states the evaluator "grades with the harness pinned to the
        protected branch, so editing it in a PR never affects that PR's own score." That was
        not true of this bot -- it ran whatever bench/scripts the PR happened to carry.

        The trust argument is the obvious one, but version skew is what actually broke it:
        #25's head predates --seal, so sealing died with "unknown arg --seal" on a PR that
        was otherwise fine, and the same skew made its reference.lock resolve frontier=0 and
        label a real speedup BASELINE.

        refdata/ is NO LONGER shipped to the box, and that is the point. Restoring the answer
        key into the same working tree the PR's binary runs in -- fixed relative path, cwd at
        the repo root -- meant a bench could open bench/refdata/hello.spkl and write it back
        out as its own --logits dump: top1 1.0, KL 0.0, no work done. Nothing rejected a
        suspiciously exact result either, since label.py only bounds top1 >= 0.90 and
        KL <= 0.05 while honest main measures 0.004."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn("git checkout -q origin/main -- bench/scripts", src,
                      "the bot grades with the PR's own harness")
        self.assertNotIn("origin/main -- bench/scripts bench/refdata", src,
                         "the answer key must not be shipped to the box")
        self.assertIn('"rm -rf bench/refdata"', src,
                      "a PR could otherwise carry its own copy of the answer key")
        self.assertLess(src.index("origin/main -- bench/scripts"),
                        src.index("kimi_k3_eval.sh --node"),
                        "the harness must be restored BEFORE the eval runs")

    def test_accuracy_is_graded_off_the_box(self):
        """The correctness gate is all that stands between a fast-but-wrong kernel and a
        payout, so it must not be decided by the machine running the code under test.

        The ids go out, the logits come back, and compare_logits.py runs on the controller
        against the controller's reference. The box cannot fake a match to a file it does not
        have. Both main and the PRs are graded this way -- the frontier and the things
        compared to it have to be produced by an identical procedure."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn("def measure_accuracy(", src)
        self.assertIn("compare_logits.py", src, "the comparison must run here")
        self.assertEqual(src.count("measure_accuracy("), 3,
                         "definition + main + each PR: the frontier must be graded like the "
                         "PRs it is compared against")
        # the harness must accept the controller's numbers, and say where they came from
        ev = (ROOT / "bench/scripts/kimi_k3_eval.sh").read_text()
        self.assertIn("--top1)", ev)
        self.assertIn("--kl)", ev)
        self.assertIn("accuracy_source", ev,
                      "a verifier cannot otherwise tell which path graded the run")

    def test_bot_resets_box_state_between_prs(self):
        """Each PR must start from a clean box, and both ways it fails blame the wrong PR.

        An orphaned bench holds GPU memory: killing the controller does not kill what it
        started over ssh, so the next weight load dies in cudaMalloc partway through layer
        92 -- indistinguishable from an OOM bug in the PR under test. And restoring the
        protected harness stages bench/scripts, so the NEXT checkout aborts with "local
        changes would be overwritten" and that PR is reported as a build failure it had
        nothing to do with. Both were observed in one run.

        The pkill must match the built path, not the bare target name, or the build's own
        `--target kimi_k3_tp_bench` command line is caught by its own cleanup."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn("git reset -q --hard", src, "a dirty tree blocks the next checkout")
        self.assertIn("git clean -qfd", src)
        # NOT pkill -f. Any pattern precise enough to match the bench also appears in the
        # command line doing the matching, so pkill kills the remote shell running it. The
        # chain then dies before cmake and the PR is reported as "build failed" with no
        # output whatsoever -- verified on the box: the shell does not survive its own
        # cleanup. Killing what holds GPU memory cannot match a shell.
        # Check EXECUTABLE lines only. Asserting against the whole file matches the comment
        # above explaining why pkill -f is wrong -- which is the third time in this work
        # that an assertion has matched its own rationale.
        code = [l for l in src.split("\n") if not l.strip().startswith("#")]
        self.assertFalse([l for l in code if "pkill -f" in l],
                         "pkill -f matches the shell running it")
        self.assertIn("--query-compute-apps=pid", src,
                      "cleanup must target GPU memory holders, not a command-line pattern")

    def test_strip_list_covers_everything_label_py_emits(self):
        """DERIVED_BY_CI was hand-maintained and label.py grew pct_of_ceiling and
        effective_pct after it was written, so the bot posted two derived numbers the
        protected side is supposed to own. Derive the expectation from label.py's ACTUAL
        output instead of trusting the list to have kept up."""
        import subprocess as sp, json as _j
        env = {**os.environ, "SPARKINFER_DIFFICULTY_REF": "18.3210"}
        out = sp.run([sys.executable, str(ROOT / "bench/scripts/label.py"),
                      "9.54", "3.55", "0", "1.0", "0.004", "abc1234"],
                     capture_output=True, text=True, check=True, env=env).stdout
        emitted = set(_j.loads(out.split("RESULT_JSON ", 1)[1]))
        measurements = {"tps", "top1", "kl", "commit", "node", "devices", "layers",
                        "quant", "ms_per_token", "engine", "llama_commit", "receipt_id"}
        leaked = emitted - measurements - set(self._bot().DERIVED_BY_CI)
        self.assertEqual(leaked, set(),
                         f"label.py emits {leaked} and the bot would post them")

    def test_sealing_is_only_claimed_when_the_log_says_so(self):
        """A receipt id proves a receipt was minted, not that anyone can find it. One run
        reported "2 sealed to the log" while sparkinfer-k3-log stayed at one entry: the
        harness printed RESULT_JSON, THEN failed to publish and exited 1, and the bot
        scraped the id off stdout without checking either the exit status or the log."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn("r.returncode != 0", src, "the harness's seal failure is ignored")
        self.assertIn("contents/runs/", src, "sealing is claimed without asking the log")
        self.assertIn('r.pop("receipt_id", None)', src,
                      "an unfindable receipt must not stay in the payload")

    def test_labels_avoid_the_deprecated_graphql_path(self):
        """gh pr edit --add-label queries projectCards, which GitHub deprecated, so it exits
        1 even on a repo with no projects. The bot printed ">> merge-first: #25" while the
        label never landed, because nothing checked the return code."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        code = [l for l in src.split("\n") if not l.strip().startswith("#")]
        self.assertFalse([l for l in code if '"pr", "edit"' in l and "label" in l],
                         "gh pr edit --add-label exits 1 on the projectCards deprecation")
        self.assertIn("issues/{num}/labels", src, "labels must go through the REST endpoint")

    def test_receipt_is_located_by_id_not_by_mtime(self):
        """Receipt filenames are timestamp-based and carry no receipt id, so a glob on the
        id never matches. Any fallback to "newest file" publishes whichever run finished
        last -- with two PRs evaluated back to back that is a coin flip on the second, and
        it would attach one PR's signed receipt to the other PR's numbers."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        code = [l for l in src.split("\n") if not l.strip().startswith("#")]
        self.assertFalse([l for l in code if "ls -t" in l and "receipt" in l],
                         "receipt selected by modification time")
        self.assertTrue([l for l in code if "grep -l" in l and "rid" in l],
                        "receipt must be matched by the id inside the file")

    def test_admin_merge_is_opt_in_and_guarded(self):
        """--merge-admin bypasses the approving-review requirement on main, so it lands an
        emissions-bearing change with nobody reading it. Two properties matter.

        It must be OPT-IN: nothing should acquire that power by default, least of all by
        someone copying an old command line.

        And the guards are all that remain in its place. Each one is a reason a human would
        have wanted to look, and none can be satisfied by the PR merely being fast: a hold
        or copycat label, no eval tier at all, a diff touching the paths that decide
        payouts, or a conflicted branch."""
        bot = self._bot()
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn('"--merge-admin", action="store_true"', src,
                      "admin merging must be an explicit flag, never a default")
        self.assertNotIn("--merge-admin\", action=\"store_true\", default=True", src)
        # the guards themselves
        self.assertTrue(bot.merge_blockers("x", 1, {"labels": []}),
                        "a PR with no eval tier must not merge")
        for blocking in ("hold", "copycat", "needs-rebase", "flagged:gaming"):
            b = bot.merge_blockers("x", 1, {"labels": [{"name": blocking},
                                                       {"name": "eval:xl"}]})
            self.assertTrue(b, f"{blocking} must block an unreviewed merge")
        for p in ("eval/", ".github/", "bench/scripts/", "bench/refdata/"):
            self.assertIn(p, bot.NEVER_MERGE_PATHS,
                          f"{p} decides payouts and must block an unreviewed merge")

    def test_scored_context_is_128k_everywhere(self):
        """The scored context is 131,072, and THREE places have to agree on the slot suffix:
        the harness that measures, the workflow that re-derives the tier, and the bot that
        writes the frontier back. A mismatch means the bot updates one slot while CI scores
        from another, and nothing fails loudly -- the tier is just computed over unrelated
        numbers.

        This is not hypothetical. The harness used to read _128 while kimi_k3_tp_bench
        hardcoded max_ctx=64, so the slot name, the documented context and the measured
        context were three different things, and 128k -- the configuration CONTRIBUTING says
        this repo runs -- was never scored at all."""
        bot = self._bot()
        self.assertEqual(bot.CTX_SUFFIX, "128K")
        self.assertTrue(bot._frontier_slot("h200x8", "UD-IQ1_S").endswith("_128K"))
        ev = (ROOT / "bench/scripts/kimi_k3_eval.sh").read_text()
        self.assertIn('SCORED_CTX="${KIMI_K3_SCORED_CTX:-131072}"', ev)
        self.assertIn('CTX_SUFFIX="${KIMI_K3_CTX_SUFFIX:-128K}"', ev)
        self.assertIn('--ctx "$SCORED_CTX" --seek', ev,
                      "the speed pass must run at the scored context")
        self.assertIn('SPARKINFER_SCORED_CONTEXT="$SCORED_CTX"', ev,
                      "the verdict must record which context earned it")
        wf = (ROOT / ".github/workflows/eval-label.yml").read_text()
        self.assertIn('{kind}_128K', wf,
                      "the trusted re-derivation must read the same slot the bot writes")
        # 128K has to resolve to a context, or a pinned llama reference at it is unverifiable
        import importlib.util as _u
        spec = _u.spec_from_file_location("crl", ROOT / "bench/scripts/check_reference_lock.py")
        crl = _u.module_from_spec(spec)
        spec.loader.exec_module(crl)
        self.assertEqual(crl.CTX_OF_SUFFIX.get("128K"), 131072)

    def test_accuracy_is_not_measured_at_the_scored_context(self):
        """--seek attends over a ZEROED cache. That is faithful for TIMING (the MLA reduction
        is dense and value-independent) and meaningless for CORRECTNESS -- the logits are
        garbage by construction.

        So the accuracy pass must run WITHOUT those flags, against the llama.cpp capture at
        the context that capture was taken at. Scoring speed at 128k and correctness at 4
        tokens is the intended split, not an oversight; using the seeked run for top-1 would
        hand every PR a perfect score on noise."""
        ev = (ROOT / "bench/scripts/kimi_k3_eval.sh").read_text()
        acc = ev[ev.index("# ---- 2. correctness"):ev.index("# ---- 3. label")]
        self.assertIn("--ids", acc)
        self.assertNotIn("--seek", acc,
                         "accuracy measured over a zeroed cache is not accuracy")
        self.assertNotIn("--ctx", acc)

    def test_speed_is_timed_externally_not_read_from_the_binary(self):
        """kimi_k3_tp_bench is built from the PR's OWN runtime/, so anything it PRINTS is
        under the PR's control.

        The tier used to be sed'd straight out of its stdout with `head -1`, while the honest
        line is printed near the end -- so one printf at startup set the score. An earlier
        attempt to guard this by requiring the bench to announce its seek was defeated the
        same way: by printing the announcement. A binary cannot be asked to certify itself.

        So the cost per token is derived from WALL CLOCK across two runs at different token
        counts; fixed costs (the ~150 s load) cancel in the difference. Elapsed time is the
        one thing a printf cannot forge."""
        ev = (ROOT / "bench/scripts/kimi_k3_eval.sh").read_text()
        self.assertNotIn("sed -n 's/.*ms\\/token[^(]*(\\([0-9.]*\\) tok\\/s).*/\\1/p' | head -1)\"\nMSTOK", ev)
        self.assertIn("TOKENS_HI=", ev, "the second run is what makes the difference a rate")
        self.assertIn("date +%s%N", ev, "the timing must come from the controller's clock")
        speed = ev[ev.index("# ---- 1. speed"):ev.index("# ---- 2. correctness")]
        self.assertIn("ns_hi - ns_lo", speed, "the bound must be a wall-clock differential")
        # a claim faster than elapsed time allows must REFUSE, not warn. The exact
        # comparison lives in test_wall_clock_is_a_bound_not_the_reading.
        self.assertIn("refusing to score", speed)
        self.assertIn("self_tps > ext_tps * tol", speed,
                      "a binary claiming to be faster than elapsed time must be rejected")

    def test_wall_clock_is_a_bound_not_the_reading(self):
        """Two numbers, two jobs, and conflating them broke a round on TRUSTED main.

        The self-report is timed inside the process around decode only, so it is precise --
        and precision is what a tier needs, since a noisy number moves PRs across band
        boundaries for nothing. The differential is external and therefore unforgeable, but
        it carries the run-to-run variance of the model load, so it is a BOUND.

        Using it as the reading failed: at 8 and 24 tokens the marginal signal was ~3.5 s
        against a ~29 s load that varies by 1-2 s, so jitter became a 32% error -- 4.55
        tok/s self-reported against 3.45 measured, and the harness refused itself. The
        margin is a fixed, much larger token count now so the signal dominates the jitter.

        Jitter is one-directional -- it only ever makes the differential read SLOWER than
        truth -- so the two sides of the bound are not symmetric. `self_tps > ext_tps * tol`
        catches a claim faster than elapsed time; `ext_tps > self_tps * work_tol` catches a
        differential too CHEAP to be 128 tokens of decode. See
        test_the_longer_run_must_do_the_extra_work for why the second one is not optional."""
        ev = (ROOT / "bench/scripts/kimi_k3_eval.sh").read_text()
        speed = ev[ev.index("# ---- 1. speed"):ev.index("# ---- 2. correctness")]
        self.assertIn("MARGIN_TOKENS=", speed,
                      "a multiple of TOKENS gives a signal smaller than the load jitter")
        self.assertNotIn("TOKENS * 3", speed)
        self.assertIn("self_tps > ext_tps * tol", speed,
                      "a claim faster than elapsed time must be rejected")
        self.assertIn('print(f"{self_tps:.2f} {self_ms:.2f}")', speed,
                      "the SELF-REPORT is what gets scored; the differential only bounds it")
        # a missing self-report is fatal -- there is nothing to score
        self.assertIn("reported no ms/token line", speed)

    # ---- the speed guard, exercised rather than read -----------------------------------
    def _speed_guard(self):
        """Extract the speed heredoc out of kimi_k3_eval.sh so the test runs the code that
        actually gates a PR, not a paraphrase of it that can drift."""
        import tempfile
        ev = (ROOT / "bench/scripts/kimi_k3_eval.sh").read_text()
        body = ev[ev.index("<<'PY'\nimport sys"):]
        body = body[body.index("\n") + 1:body.index("\nPY\n")]
        f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        f.write(body); f.close()
        return f.name

    def _run_guard(self, ns_lo, ns_hi, n_lo, n_hi, self_tps, self_ms,
                   tol="1.5", work_tol="2.0", llama_ref="16.70",
                   max_over_llama="3.0", jitter_s="2.0"):
        return subprocess.run(
            [sys.executable, self._speed_guard(), str(int(ns_lo)), str(int(ns_hi)),
             str(n_lo), str(n_hi), str(self_tps), str(self_ms), tol, work_tol,
             llama_ref, max_over_llama, jitter_s], capture_output=True, text=True)

    def test_the_longer_run_must_do_the_extra_work(self):
        """THE DIFFERENTIAL ONLY BOUNDS A CLAIM IF THE SECOND RUN DECODED THE EXTRA TOKENS.

        n_tokens is argv[4] of a binary built from the PR's own runtime/. Nothing compels it
        to be honoured, and a bench that ignores it does identical work in both runs: d_ns
        collapses to load jitter, ext_tps goes to (nearly) infinity, and `self_tps > ext_tps
        * tol` passes for ANY claim. The upper bound alone is neutered by doing LESS work,
        which is the one thing it cannot notice.

        Honest numbers below are main's 128k reality: ~1 tok/s, so 128 marginal tokens is
        ~128 s of wall clock on top of a ~150 s load."""
        LOAD, HONEST_MS = 150e9, 1000.0          # 150 s load, 1.00 s/token
        honest_hi = LOAD + 8e9 + 128e9           # the 128 marginal tokens actually happen

        rc = self._run_guard(LOAD + 8e9, honest_hi, 8, 136, 1.0, HONEST_MS)
        self.assertEqual(rc.returncode, 0, f"an honest run was refused:\n{rc.stderr}")
        self.assertEqual(rc.stdout.split()[0], "1.00", "the self-report must be what is scored")

        # THE ATTACK: both runs decode 8 tokens, so the difference is load jitter alone
        # (1.2 s). 128 "marginal" tokens over 1.2 s reads as 106 tok/s, so the ceiling the
        # one-sided bound computes is ~160 -- and the claim below is 40 tok/s, forty times
        # the truth and comfortably underneath it.
        attack = dict(ns_lo=LOAD + 8e9, ns_hi=LOAD + 8e9 + 1.2e9, n_lo=8, n_hi=136,
                      self_tps=40.0, self_ms=25.0)

        # First: show the one-sided bound really does wave it through (work_tol off).
        rc = self._run_guard(**attack, work_tol="1e9", max_over_llama="1e9")
        self.assertEqual(rc.returncode, 0, "the premise of this test no longer holds")
        self.assertEqual(rc.stdout.split()[0], "40.00", "the fabricated claim was scored")

        rc = self._run_guard(**attack)
        self.assertEqual(rc.returncode, 1,
                         "a run where the longer pass did no extra work was scored:\n"
                         f"stdout={rc.stdout!r}")
        self.assertIn("did not do the extra work", rc.stderr)

        # ... and the same collapse cannot be laundered by claiming a modest number either:
        # the floor scales with the claim, so 5 tok/s still needs ~12.8 s of marginal time.
        rc = self._run_guard(LOAD + 8e9, LOAD + 8e9 + 1.2e9, 8, 136, 5.0, 200.0)
        self.assertEqual(rc.returncode, 1, "a collapsed differential passed with a small claim")

    def test_the_two_guards_leave_no_band_uncovered(self):
        """A timing guard can only bound a claim in proportion to the time that claim
        implies: 128 tokens at 1 tok/s is 128 s of evidence, at 500 tok/s it is 0.26 s, which
        load jitter supplies for free. The work check therefore stops biting above
        MARGIN_TOKENS/(jitter*work_tol), and the band above THAT is closed by plausibility --
        a claim past a few multiples of llama.cpp is refused outright, because label.py
        saturates at XL long before it and nothing here can tell 50x from a printf.

        The defaults must overlap. If the ceiling sat above the crossover there would be a
        band where a fabricated claim is both timing-free and scorable, which is the original
        hole in a narrower window."""
        MARGIN, JITTER, WORK_TOL, LLAMA, MULT = 128, 2.0, 2.0, 16.70, 3.0
        crossover = MARGIN / (JITTER * WORK_TOL)      # 32 tok/s: timing stops being evidence
        ceiling = LLAMA * MULT                        # 50.1 tok/s: plausibility takes over
        self.assertLessEqual(
            ceiling, crossover * 2,
            f"defaults leave a {crossover:.0f}-{ceiling:.0f} tok/s band that is neither "
            "timed nor bounded by the reference")

        LOAD = 150e9
        # A claim inside the free-timing band is still refused, by the ceiling.
        rc = self._run_guard(LOAD + 8e9, LOAD + 8e9 + 1.2e9, 8, 136, 120.0, 8.33)
        self.assertEqual(rc.returncode, 1, "a 7x-llama claim rode the jitter into a tier")
        self.assertIn("llama.cpp reference", rc.stderr)

        # And an honest run that genuinely beats llama.cpp by a normal margin is untouched.
        rc = self._run_guard(LOAD + 8e9, LOAD + 8e9 + 6.4e9, 8, 136, 20.0, 50.0)
        self.assertEqual(rc.returncode, 0, f"a real 20 tok/s run was refused:\n{rc.stderr}")

    def test_jitter_never_trips_the_work_check(self):
        """Load jitter INFLATES d_ns (the differential reads slower than truth), so it can
        only make the work check pass more easily. The check must therefore never fire on an
        honest run, however unlucky the load -- a false refusal here costs a real PR its
        tier, which is exactly how the 8-and-24-token version broke on trusted main."""
        LOAD, MS = 150e9, 1000.0
        for jitter_s in (-2, -1, 0, 1, 2, 5):
            rc = self._run_guard(LOAD + 8e9, LOAD + 8e9 + 128e9 + jitter_s * 1e9,
                                 8, 136, 1.0, MS)
            self.assertEqual(rc.returncode, 0,
                             f"{jitter_s:+} s of load jitter refused an honest run:\n{rc.stderr}")

    def test_a_faster_claim_than_elapsed_time_is_still_refused(self):
        """The original one-sided check must survive the second side being added."""
        LOAD = 150e9
        rc = self._run_guard(LOAD + 8e9, LOAD + 8e9 + 128e9, 8, 136, 4.0, 250.0)
        self.assertEqual(rc.returncode, 1, "a 4x-over-wall-clock claim was scored")
        self.assertIn("wall clock allows at most", rc.stderr)

    def test_a_guard_refusal_actually_aborts_the_script(self):
        """THE GUARD'S EXIT CODE HAS TO SURVIVE THE SHELL AROUND IT.

        The wiring used to be `read -r TPS MSTOK <<<"$(python3 ...)" || exit 1`. The
        `|| exit 1` binds to `read`, bash discards the substitution's exit status, and read
        returns 0 on the empty string a refusing guard leaves behind -- so the refusal
        printed its message and the script walked on with TPS="" into label.py, which died
        on float('') with a traceback that buried the reason. #64 hit exactly this on the
        guard's first live firing.

        Tested against the REAL fragment cut from the script, not a paraphrase: refusal
        inputs must exit 1, must not print the empty ">> sparkinfer:" line, and honest
        inputs must still score."""
        import tempfile
        ev = (ROOT / "bench/scripts/kimi_k3_eval.sh").read_text()
        live = [l for l in ev.split("\n") if not l.lstrip().startswith("#")]
        self.assertFalse([l for l in live if 'read -r TPS MSTOK <<<"$(' in l],
                         "the swallowed-exit wiring is back")
        start = ev.index('SPEED_VERDICT="$(python3 -')
        end = ev.index('[self-timed, within the wall-clock bound]"', start)
        end = ev.index("\n", end) + 1
        fragment = ev[start:end]

        def run(ns_lo, ns_hi, self_tps, self_ms):
            script = (
                "set -euo pipefail\n"
                f"NS_LO={int(ns_lo)}\nNS_HI={int(ns_hi)}\n"
                "TOKENS_LO=8\nTOKENS_HI=136\n"
                f"SELF_TPS={self_tps}\nSELF_MS={self_ms}\nLLAMA_REF=16.70\n"
                + fragment
                + "\necho SCORED_TPS=$TPS\n")
            f = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False)
            f.write(script); f.close()
            return subprocess.run(["bash", f.name], capture_output=True, text=True)

        LOAD = 150e9
        # honest: guard passes, verdict parsed, scored
        r = run(LOAD + 8e9, LOAD + 8e9 + 128e9, 1.0, 1000.0)
        self.assertEqual(r.returncode, 0, f"honest run failed:\n{r.stderr}")
        self.assertIn("SCORED_TPS=1.00", r.stdout)

        # refusal: #64's live numbers — collapsed differential, modest claim
        r = run(LOAD + 8e9, LOAD + 8e9 + 5.2e9, 10.53, 94.98)
        self.assertEqual(r.returncode, 1,
                         "the guard refused but the SCRIPT did not abort:\n"
                         f"stdout={r.stdout!r}")
        self.assertIn("did not do the extra work", r.stderr)
        self.assertNotIn(">> sparkinfer:  tok/s", r.stdout,
                         "the empty echo means the script walked past the refusal")
        self.assertNotIn("SCORED_TPS", r.stdout)

    def test_a_missing_reference_says_so_out_loud(self):
        """LLAMA_REF is 0 on a node/quant/context that has not been referenced yet. The
        ceiling cannot be computed, so the residual is open -- and a guard that quietly
        degrades to nothing is worse than one that says it did."""
        LOAD = 150e9
        rc = self._run_guard(LOAD + 8e9, LOAD + 8e9 + 128e9, 8, 136, 1.0, 1000.0,
                             llama_ref="0")
        self.assertEqual(rc.returncode, 0)
        self.assertIn("no llama.cpp reference", rc.stderr)
        self.assertIn("unguarded", rc.stderr)

    def test_polaris_key_is_kept_out_of_the_benchs_environment(self):
        """`. /root/.polaris_env` exports POLARIS_API_KEY, and every child inherits it --
        including kimi_k3_tp_bench, which the PR wrote. One getenv() exfiltrates the key that
        signs the attestation ledger, which would let anyone forge a receipt for any run.

        The harness needs the key for the seal step, so it keeps it; the bench does not, so
        every invocation of the bench strips it."""
        ev = (ROOT / "bench/scripts/kimi_k3_eval.sh").read_text()
        self.assertIn("env -u POLARIS_API_KEY", ev)
        # EVERY invocation, not just the first
        import re as _re
        calls = _re.findall(r'^\s*(?:\w+="\$\()?.*"\$BENCH"', ev, _re.M)
        stripped = _re.findall(r'env -u POLARIS_API_KEY "\$BENCH"', ev)
        self.assertGreaterEqual(len(stripped), 2,
                                f"a bench call still gets the signing key: {calls}")

    def test_model_weights_are_fingerprinted_before_scoring(self):
        """The weights are shared state and the binary under test runs as root, so a
        malicious build can corrupt the baseline every later measurement depends on --
        exactly what reference.lock's C2 quarantine exists for, and the eval path never
        checked. A full sha256 is unaffordable (553 GiB per run), so the shard set is
        fingerprinted by size and mtime.

        This DETECTS tampering; preventing it means mounting the models dir read-only."""
        ev = (ROOT / "bench/scripts/kimi_k3_eval.sh").read_text()
        self.assertIn("FINGERPRINT=", ev)
        self.assertIn("model_fingerprint", ev, "the fingerprint must reach the receipt")
        fp = ev[ev.index("FINGERPRINT="):ev.index("# ---- 1. speed")]
        self.assertIn("exit 1", fp, "a mismatch against a pinned fingerprint must refuse")

    def test_frontier_is_measured_not_pinned(self):
        """main is benchmarked every round and THAT is what PRs are scored against.

        The pinned constant failed exactly the way constants do. #25 merged and took main
        from 3.55 to 9.54 tok/s; nothing in the merge path updated reference.lock; the next
        PR's 8% gain was priced against a frontier main had already beaten by 2.7x and came
        out XL when the honest tier was S. A measured frontier cannot drift from main because
        it IS main, measured minutes earlier on the same box."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn("def measure_frontier(", src, "the round must measure main")
        # the measurement has to reach the harness, or it is decoration: left to itself
        # kimi_k3_eval.sh reads the frontier back out of reference.lock.
        self.assertIn('front_flag = "" if frontier is None else f" --frontier {frontier}"',
                      src, "the measured frontier must be passed to the harness")
        main_fn = src[src.index("def main("):]
        self.assertLess(main_fn.index("measure_frontier("), main_fn.index("evaluate("),
                        "main must be measured before any PR is scored against it")
        # ...and before anything is posted: eval-label.yml reads reference.lock at comment
        # time, so a comment posted first gets a tier derived from the stale frontier.
        self.assertLess(main_fn.index("reconcile_lock("), main_fn.index("post("),
                        "the lock must be reconciled before a result is posted")

    def test_frontier_write_is_narrow_and_raise_only(self):
        """Writing reference.lock gives the bot the file that sets payout tiers, on a path
        CODEOWNERS and sensitive-paths-guard exist to protect. Three things keep it narrow.

        RAISE-ONLY is the load-bearing one. Lowering the frontier makes every subsequent PR's
        gain look bigger, so a thermally-throttled box or a bad main could mint tiers for
        everyone measured behind it. That direction has to stop the round, not re-baseline
        it."""
        bot = self._bot()
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn("FRONTIER_ALARM", src, "a slower main must stop the round")
        self.assertLess(bot.FRONTIER_ALARM, 1.0)
        self.assertGreater(bot.FRONTIER_TOL, 0,
                           "without a tolerance the lock is rewritten on pure noise, and "
                           "every commit to main costs every open PR a rebase")
        # only the one node+quant frontier slot is ever addressed
        self.assertEqual(bot._frontier_slot("h200x8", "UD-IQ1_S"),
                         "KIMI_K3_H200X8_IQ1S_SPARKINFER_128K")
        self.assertNotIn("LLAMA", bot._frontier_slot("h200x8", "UD-IQ1_S"),
                         "the llama reference is a real external constant — never rewritten")
        rec = src[src.index("def reconcile_lock("):src.index("def evaluate(")]
        self.assertIn("if seen != 1:", rec,
                      "a rewrite that matched zero or many lines must refuse, not guess")
        self.assertIn('"branch=main"', rec)

    def test_merge_decisions_wait_for_asynchronous_github_state(self):
        """Both merge-path failures were one bug: acting on GitHub state that is computed
        asynchronously, without waiting for it to settle.

        mergeStateStatus is UNKNOWN while GitHub recomputes, and returns to UNKNOWN whenever
        the base branch moves -- which this bot does to itself when it commits the frontier.
        #57 reported non-BEHIND before that finished, so its branch update was skipped, and
        minutes later the merge was refused for the BEHIND the update would have fixed.

        And the tier is applied by eval-label.yml in CI, after the comment is posted. Reading
        the label immediately always found nothing, so --merge-admin could never merge a PR
        it had just evaluated. Waiting is the fix; merging without a tier is not, because the
        label is the trusted side's verdict and the bot must not act on its own number."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn("def wait_mergeable_state(", src)
        self.assertIn("def wait_for_tier(", src)
        self.assertIn('!= "UNKNOWN"', src, "UNKNOWN is no answer, not a stale one")
        # the branch-update decision must read a settled state
        main_fn = src[src.index("def main("):]
        self.assertIn("wait_mergeable_state(args.repo, num).get(", main_fn,
                      "the update decision reads mid-recompute state")
        self.assertLess(main_fn.index("wait_for_tier("), main_fn.index("merge_winner("),
                        "the tier must land before the merge decision reads it")
        # ...and a timeout must NOT be treated as permission
        tier = src[src.index("def wait_for_tier("):src.index("def merge_blockers(")]
        self.assertIn("refusing to merge on the bot's own number", tier)
        self.assertIn("return []", tier, "no tier means no merge, not merge anyway")

    def test_auto_merge_still_respects_the_substantive_guards(self):
        """Queueing is not merging, but it is arming: once the required review lands, GitHub
        merges with nobody looking again. So `hold` / `copycat` / a maintainer-owned path
        must stop a queue too.

        The merge-STATE guards must NOT apply, though -- waiting for the review is the entire
        point of the flag, and refusing to queue because the review has not happened yet
        would make --auto-merge a no-op."""
        bot = self._bot()
        held = {"labels": [{"name": "hold"}, {"name": "eval:xl"}]}
        self.assertTrue(bot.merge_blockers("x", 1, held, waiting_is_blocking=False),
                        "a held PR must not be armed for auto-merge")
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        mmf = src[src.index("def mark_merge_first("):src.index("def main(")]
        self.assertIn("waiting_is_blocking=False", mmf,
                      "auto-merge must consult the substantive guards")
        self.assertLess(mmf.index("merge_blockers("), mmf.index('"--auto"'),
                        "the guards must be checked before the queue is armed")

    def test_bot_clears_the_staleness_it_caused(self):
        """Committing the measured frontier moves main, and under
        required_status_checks.strict that puts every open PR BEHIND immediately.

        Labelling them needs-rebase and waiting asks contributors to hand-fix bookkeeping the
        bot did to them -- and it does not merely inconvenience them, it DEADLOCKS the round:
        needs-rebase is in NEVER_MERGE_LABELS, so the frontier commit blocks the very PR it
        was measured for. #20 was labelled merge-first and refused in the same round.

        The update has to happen BEFORE the measurement, because merging main into the branch
        changes the head sha and eval-label.yml refuses a payload measured on another
        commit."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn("def update_pr_branch(", src)
        self.assertIn("update-branch", src)
        self.assertIn("expected_head_sha=", src,
                      "updating without pinning the expected head races the author's push")
        main_fn = src[src.index("def main("):]
        self.assertLess(main_fn.index("update_pr_branch("), main_fn.index("evaluate("),
                        "the branch must be current before the number is taken, or the "
                        "measured sha is not the sha that merges")
        # and the stale-tier labels must be reconciled before any merge decision reads them
        self.assertLess(main_fn.index("sync_rebase_labels("), main_fn.index("merge_winner("),
                        "a leftover needs-rebase blocks the merge it never meant to block")

    def test_round_log_is_published_with_the_receipt(self):
        """A receipt proves a number was signed. It says nothing about what the round did to
        get there -- what the frontier moved to, which build failed, why a merge was refused.

        That context lived only in the operator's terminal, which is the one place someone
        checking the number afterwards cannot look. Both streams are captured: the
        interesting parts of a bad round are on stderr, and a log that drops them reads like
        a clean run."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn("def publish_log(", src)
        self.assertIn("runs/{rid}/eval.log", src)
        main_fn = src[src.index("def main("):]
        self.assertIn("sys.stdout, sys.stderr = _Tee(", main_fn,
                      "stderr must be captured too, or a failed round publishes a clean log")
        # Published last, so the artefact covers the whole round rather than its opening.
        # The AUTOMATIC publish is publish_round_log(); publish_log() also appears earlier in
        # main() for the --publish-log backfill, which returns before any round runs.
        self.assertLess(main_fn.index("merge_winner("), main_fn.rindex("publish_round_log("),
                        "the log must include the merge decision it explains")

    def test_compare_logits_exit_code_is_a_verdict_not_an_error(self):
        """compare_logits returns 0 only for mean_kld < 1e-5 -- a SAME-IMPLEMENTATION bar,
        "two implementations of one arithmetic". K3's accepted parity is 4.05e-03, ~400x
        that, from a documented cause (f32 activations where ggml quantizes before a
        quantized mat-vec). So it reports FAIL on every healthy K3 run.

        The shell this replaced ran it with `|| true` and read the JSON. Porting it to
        Python without that turned every round into "compare_logits failed" with an empty
        stderr -- which is exactly how the first hardened round died. The gate that decides
        anything is label.py's (top1 >= 0.90, kl <= 0.05), applied downstream."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        acc = src[src.index("def measure_accuracy("):src.index("def _box_eval(")]
        self.assertNotIn("if p.returncode != 0 or not", acc,
                         "a non-zero exit here is the normal, healthy K3 result")
        self.assertIn("if not out:", acc,
                      "only a MISSING payload is a real failure — the tool did not run")
        # and the real bars still live in label.py
        lbl = (ROOT / "bench/scripts/label.py").read_text()
        self.assertIn('SPARKINFER_TOP1_BAR",  "0.90"', lbl)
        # The shared DEFAULT stays loose because label.py also scores the inherited Qwen
        # track, whose honest runs measure KL 0.0175-0.03. Tightening it here would REJECT
        # every one of them.
        self.assertIn('SPARKINFER_KL_BAR",    "0.20"', lbl)

    def test_k3_pins_its_own_kl_bar_in_both_places(self):
        """K3's accuracy policy is 0.05, and it must be pinned where K3 is scored rather than
        in the shared default -- label.py also grades the Qwen track at KL 0.0175-0.03.

        BOTH places, or neither counts. The harness computes the verdict and eval-label.yml
        RE-DERIVES it from the same numbers on the protected branch; if they disagree about
        KL_BAR they disagree about REJECT, which is the same class of bug as disagreeing
        about the frontier -- and it would surface as 'reported label != re-derived'.

        Why tighter than the default matters: K3's main measures 0.004046, so under 0.20 a PR
        could degrade parity forty-fold and still pass clean; 0.05 bounds that at ~12x.

        It does NOT catch #63 and this test does not claim it does -- 0.008075 is under both
        0.05 and the 0.02 soft flag, so that PR would pass clean and unflagged today. A floor
        bounds the worst case; only a ratchet against measured parity detects a regression.
        See the KL_BAR comment in label.py."""
        ev = (ROOT / "bench/scripts/kimi_k3_eval.sh").read_text()
        self.assertIn('KL_BAR="${KIMI_K3_KL_BAR:-0.05}"', ev)
        self.assertIn('KL_PREFER="${KIMI_K3_KL_PREFER:-0.02}"', ev)
        self.assertIn('SPARKINFER_KL_BAR="$KL_BAR"', ev,
                      "the bar label.py is given must be the one recorded in provenance")
        self.assertIn('SPARKINFER_KL_PREFER="$KL_PREFER"', ev)
        wf = (ROOT / ".github/workflows/eval-label.yml").read_text()
        self.assertIn('KL_BAR, KL_PREFER = "0.05", "0.02"', wf,
                      "the trusted re-derivation would use the Qwen default and disagree")

    def test_a_moved_kl_knob_names_itself(self):
        """The harness bar is env-overridable and the workflow's is pinned, so a box running
        with KIMI_K3_KL_BAR set produces a verdict the trusted side cannot reproduce. Left to
        the label comparison it surfaces as 'payload edited or harness version mismatch' --
        naming the two things that are not wrong. The harness records the bars it used and
        the workflow checks them first, so the message names the knob."""
        ev = (ROOT / "bench/scripts/kimi_k3_eval.sh").read_text()
        self.assertIn('"kl_bar": float(kl_bar)', ev, "the bars are not recorded in provenance")
        lbl = (ROOT / "bench/scripts/label.py").read_text()
        self.assertNotIn('"kl_bar"', lbl.split("_VERDICT_KEYS = ")[1].split("}")[0],
                         "kl_bar in the allowlist would let a payload set its own bar")
        head = "8757caf10683484582b346663ee8944ac66e2e06"
        base = {"tps": 4.2, "top1": 1.0, "kl": 0.001, "commit": head[:7]}
        rc, _, err = self._run_verdict({**base, "kl_bar": 0.05, "kl_prefer": 0.02}, head)
        self.assertEqual(rc, 0, f"a run on the pinned bars was refused:\n{err}")
        rc, _, err = self._run_verdict({**base, "kl_bar": 0.20, "kl_prefer": 0.15}, head)
        self.assertEqual(rc, 1, "a run scored under a looser bar was labelled anyway")
        self.assertIn("kl_bar=0.2", err)
        self.assertIn("KIMI_K3_KL_BAR", err, "the error must name the knob to unset")
        # A payload from before the pin is warned about, not refused -- it would block every
        # in-flight PR the moment this lands.
        rc, _, err = self._run_verdict(base, head)
        self.assertEqual(rc, 0, "an older payload without the bars was refused")
        self.assertIn("does not record kl_bar", err)

    def test_speed_source_is_a_record_not_a_caption(self):
        """The provenance used to hardcode speed_source="wall_clock_differential" while the
        scored value is the binary's SELF-REPORT (bounded by the differential), and nothing
        read the field -- so every payload asserted the trustworthy path, including any that
        did not take it, and a log reader could not tell them apart.

        Two halves make it a record: the harness writes what actually happened, and the
        workflow refuses a value its guarded path does not produce. The legacy name stays
        accepted -- it labels the same bounded path, misnamed -- so pre-rename payloads in
        the log remain scorable."""
        ev = (ROOT / "bench/scripts/kimi_k3_eval.sh").read_text()
        self.assertIn('"speed_source": "self_report_wall_clock_bounded"', ev,
                      "the harness must record the path that actually ran")
        self.assertNotIn('"speed_source": "wall_clock_differential"', ev,
                         "the misleading caption is back")
        head = "8757caf10683484582b346663ee8944ac66e2e06"
        base = {"tps": 4.2, "top1": 1.0, "kl": 0.001, "commit": head[:7]}
        for src in ("self_report_wall_clock_bounded", "wall_clock_differential"):
            rc, _, err = self._run_verdict({**base, "speed_source": src}, head)
            self.assertEqual(rc, 0, f"speed_source={src} was refused:\n{err}")
        rc, _, err = self._run_verdict({**base, "speed_source": "binary_self_report"}, head)
        self.assertEqual(rc, 1, "an unrecognised speed_source was labelled anyway")
        self.assertIn("speed_source", err)
        rc, _, err = self._run_verdict(base, head)   # pre-recording payloads: warn only
        self.assertEqual(rc, 0, "a payload without speed_source was refused")

    def test_receipt_verification_enforces_the_repo_key_and_the_clock_policy(self):
        """Three facts about the strict path, asserted where CI actually runs it:

        1. verify_strict unpacks (passed, results) -- `ok = validator.verify()` bound the
           whole 2-tuple, always truthy, so `if not ok` was dead and a forged signature
           passed strict verification (behavioural proof lives in eval/polaris/
           test_receipt.py::TestStrictVerification).
        2. The workflow passes the repo's pinned key. Without it the signature is checked
           against the receipt's own embedded key -- integrity, not origin -- and
           eval/polaris/sparkinfer_eval.pub was committed but never loaded.
        3. It relaxes exactly the clock check, because kimi_k3_attest.py honestly records
           clocks_pinned=False (the box cannot lock clocks in-container). Default-strict
           plus that sealer means REQUIRE_EVAL_RECEIPT=1 rejects every receipt the repo's
           own sealer produces -- the flag would ship dead."""
        src = (ROOT / "eval/polaris/verify.py").read_text()
        self.assertIn("ok, checks = validator.verify(", src,
                      "the (passed, results) tuple is bound to one name again")
        wf = (ROOT / ".github/workflows/eval-label.yml").read_text()
        self.assertIn('load_trusted_key("eval/polaris/sparkinfer_eval.pub")', wf,
                      "the pinned key exists but the workflow never loads it")
        self.assertIn("trusted_key_b64=key", wf)
        self.assertIn("require_pinned_clocks=False", wf,
                      "strict clock check + honest sealer = every K3 receipt rejected")
        at = (ROOT / "bench/scripts/kimi_k3_attest.py").read_text()
        self.assertIn("clocks_pinned=False", at,
                      "the sealer must keep recording the truth, not start pinning on paper")

    # ---- clearing the previous round's tier -------------------------------------------
    def _bot(self):
        """Import the bot module itself. The structural tests read its source; this one has
        to run the function, because the bug it guards is that a call SUCCEEDS on paper."""
        import importlib
        sys.path.insert(0, str(ROOT / "eval"))
        try:
            return importlib.import_module("k3_eval_bot")
        finally:
            sys.path.pop(0)

    def _with_fake_gh(self, bot, labels, delete_rc, after=None):
        """Patch gh/_pr_state; returns (calls, restore). `after` is the label list the second
        _pr_state read sees — i.e. whether the DELETE actually took effect."""
        calls, state = [], {"n": 0}
        seen = [labels] if after is None else [labels, after]

        def fake_gh(args, timeout=120):
            calls.append(args)
            return subprocess.CompletedProcess(
                args, delete_rc, stdout="",
                stderr="" if delete_rc == 0 else "gh: Resource not accessible (HTTP 403)")

        def fake_state(repo, num):
            i = min(state["n"], len(seen) - 1)
            state["n"] += 1
            return {} if seen[i] is None else {"labels": [{"name": n} for n in seen[i]]}

        old = (bot.gh, bot._pr_state)
        bot.gh, bot._pr_state = fake_gh, fake_state
        return calls, lambda: (setattr(bot, "gh", old[0]), setattr(bot, "_pr_state", old[1]))

    def test_clearing_a_stale_tier_is_verified_not_assumed(self):
        """THE RETURN VALUE IS THE WHOLE GUARANTEE. gh() hands back a CompletedProcess and
        does not raise, so an unchecked DELETE loop treats a 403 exactly like a success: it
        printed 'cleared stale tier' over a label that was still on the PR, wait_for_tier
        returned it on the first poll, and the merge decision was taken on the previous
        round's number -- #59, with a log line asserting it had been prevented.

        Three states, three answers: cleared (True), still there (False), unreadable
        (False). Only the first may let a PR into the merge decision."""
        bot = self._bot()

        # 1. the DELETE works and the re-read confirms it
        calls, restore = self._with_fake_gh(bot, ["eval:none", "perf"], 0, after=["perf"])
        try:
            self.assertIs(bot.clear_stale_tier("o/r", 59, dry_run=False), True)
            self.assertEqual(len(calls), 1, "one DELETE for one eval:* label")
            self.assertIn("eval%3Anone", calls[0][-1], "the colon must be percent-encoded")
        finally:
            restore()

        # 2. the DELETE is refused and the label survives -- the case that used to print
        #    success. Nothing about the exit code is trusted here: the re-read decides.
        calls, restore = self._with_fake_gh(bot, ["eval:xl"], 1, after=["eval:xl"])
        try:
            self.assertIs(bot.clear_stale_tier("o/r", 59, dry_run=False), False,
                          "a stale tier that survived the DELETE was reported as cleared")
        finally:
            restore()

        # 3. someone else removed it first: the DELETE 404s but the PR is clean, which is
        #    the outcome we wanted. Exit codes would call this a failure.
        calls, restore = self._with_fake_gh(bot, ["eval:xs"], 1, after=[])
        try:
            self.assertIs(bot.clear_stale_tier("o/r", 59, dry_run=False), True)
        finally:
            restore()

        # 4. the state cannot be read at all. "No labels" is not the same as "cannot tell".
        calls, restore = self._with_fake_gh(bot, None, 0)
        try:
            self.assertIs(bot.clear_stale_tier("o/r", 59, dry_run=False), False,
                          "an unreadable PR state was treated as having no stale tier")
            self.assertEqual(calls, [], "nothing should be deleted on an unreadable state")
        finally:
            restore()

        # 5. no stale tier, and a dry run: both clean, neither writes.
        calls, restore = self._with_fake_gh(bot, ["perf"], 0)
        try:
            self.assertIs(bot.clear_stale_tier("o/r", 59, dry_run=False), True)
            self.assertEqual(calls, [])
        finally:
            restore()
        calls, restore = self._with_fake_gh(bot, ["eval:l"], 0)
        try:
            self.assertIs(bot.clear_stale_tier("o/r", 59, dry_run=True), True)
            self.assertEqual(calls, [], "dry-run deleted a label")
        finally:
            restore()

    def test_an_uncleared_tier_keeps_the_pr_out_of_the_merge_decision(self):
        """merge_blockers only asks whether SOME eval:* label is present, so on a PR whose
        previous tier could not be cleared a stale one satisfies it. The round still
        evaluates, posts and seals that PR -- the measurement is fine -- but it must not be
        the winner. Asserted structurally: the merge path reads a filtered list."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn("tier_cleared = clear_stale_tier(", src,
                      "the return value is discarded, which is the bug it exists to report")
        self.assertIn("unsafe_tier.add(num)", src)
        self.assertIn("mergeable = [r for r in results if r[0] not in unsafe_tier]", src)
        self.assertIn("mark_merge_first(args.repo, mergeable", src,
                      "merge-first still ranks a PR carrying a stale tier")
        self.assertIn("max(mergeable, key=", src,
                      "the winner is still picked from unfiltered results")
        # and the round log / sealing still see everything
        self.assertIn("for num, r in results:", src,
                      "excluding from the merge must not exclude from the receipt check")

    def test_a_fresh_eval_clears_a_moot_needs_rebase(self):
        """needs-rebase demands a re-measurement; a fresh eval at the CURRENT head IS that
        re-measurement, so the label must come off — and only then. #59 carried a label
        left over from #63's merge, was re-evaluated at the same head against the ratcheted
        frontier, won the round — and needs-rebase ∈ NEVER_MERGE_LABELS would have made
        merge_blockers refuse the bot's own winner. A human checked and removed it by hand;
        clear_moot_rebase_label is that check, and this test is the hand-check's spec:

          moot    = same head as evaluated, no conflicts     -> removed
          not moot = head moved since the eval, or DIRTY     -> kept
        """
        bot = self._bot()
        HEAD = "99e4bad5aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

        def with_state(labels, head=HEAD, state="BEHIND", after_labels=("perf",)):
            calls, seen = [], {"n": 0}
            def fake_gh(args, timeout=120):
                calls.append(args)
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            def fake_state(repo, num):
                seen["n"] += 1
                use = labels if seen["n"] == 1 else after_labels
                return {"labels": [{"name": n} for n in use],
                        "headRefOid": head, "mergeStateStatus": state}
            old = (bot.gh, bot._pr_state)
            bot.gh, bot._pr_state = fake_gh, fake_state
            return calls, lambda: (setattr(bot, "gh", old[0]),
                                   setattr(bot, "_pr_state", old[1]))

        # 1. moot: label present, head == evaluated, BEHIND (by e.g. the ratchet) -> removed
        calls, restore = with_state(["needs-rebase", "eval:m"])
        try:
            self.assertIs(bot.clear_moot_rebase_label("o/r", 59, HEAD, False), True)
            self.assertTrue(any("needs-rebase" in a[-1] for a in calls if "DELETE" in a),
                            "moot label was not deleted")
        finally:
            restore()

        # 2. head moved since the eval: the tier belongs to an older commit — label stands
        calls, restore = with_state(["needs-rebase", "eval:m"], head="f" * 40)
        try:
            self.assertIs(bot.clear_moot_rebase_label("o/r", 59, HEAD, False), False)
            self.assertEqual([a for a in calls if "DELETE" in a], [],
                             "a genuinely stale label was deleted")
        finally:
            restore()

        # 3. conflicts are real staleness no re-measurement cures
        calls, restore = with_state(["needs-rebase", "eval:m"], state="DIRTY")
        try:
            self.assertIs(bot.clear_moot_rebase_label("o/r", 59, HEAD, False), False)
            self.assertEqual([a for a in calls if "DELETE" in a], [])
        finally:
            restore()

        # 4. the delete must be VERIFIED, same discipline as clear_stale_tier
        calls, restore = with_state(["needs-rebase", "eval:m"],
                                    after_labels=("needs-rebase",))
        try:
            self.assertIs(bot.clear_moot_rebase_label("o/r", 59, HEAD, False), False,
                          "a DELETE that did not take effect was reported as cleared")
        finally:
            restore()

        # 5. wiring: every eval clears it (all modes), and both merge paths re-check
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn('clear_moot_rebase_label(args.repo, num, pr.get("headRefOid"', src,
                      "a plain round leaves the moot label for the next merge to trip on")
        mw = src[src.index("def merge_winner("):src.index("def sync_rebase_labels(")]
        self.assertIn("clear_moot_rebase_label(", mw,
                      "merge_winner trusts that nothing re-labelled the winner since eval")
        mf = src[src.index("def mark_merge_first("):src.index("def main(")]
        self.assertIn("clear_moot_rebase_label(", mf,
                      "the auto-merge queue trusts that nothing re-labelled the winner")

    def test_the_post_merge_sweep_does_not_wait_for_github_to_notice(self):
        """The sweep at the tail of a merge runs SECONDS after main moved, while GitHub
        still reports every other PR as UNKNOWN — and the BEHIND/DIRTY test, right for a
        standalone sweep, labels nothing at the one moment staleness is a certainty. The
        merge is the evidence; force_stale makes the sweep act on it. The no-tier noise
        rule and the merged-PR skip must survive, and 'current again' must not fire off a
        stale UNKNOWN."""
        bot = self._bot()
        posted, deleted = [], []

        def fake_gh(args, timeout=120):
            if args[:2] == ["pr", "view"]:
                num = args[2]
                labels = {"5": [], "7": ["eval:xs"], "8": [],
                          "9": ["needs-rebase", "eval:none"]}[num]
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps(
                    {"state": "OPEN", "mergeStateStatus": "UNKNOWN",
                     "labels": [{"name": n} for n in labels]}), stderr="")
            if "-X" in args and "POST" in args:
                posted.append(args[args.index("-X") - 1] if False else args[3])
            if "-X" in args and "DELETE" in args:
                deleted.append(args[3])
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        old = bot.gh
        bot.gh = fake_gh
        try:
            prs = [{"number": 5}, {"number": 7}, {"number": 8}, {"number": 9}]
            # tail-of-merge sweep: #5 merged; #7 (tier, UNKNOWN) must be labelled anyway
            bot.sync_rebase_labels("o/r", prs, merged_num=5, dry_run=False, force_stale=True)
            self.assertTrue(any("/issues/7/labels" in p for p in posted),
                            "an UNKNOWN tier-carrying PR was not labelled after the merge")
            self.assertFalse(any("/issues/5/" in p for p in posted),
                             "the merged PR itself was labelled")
            self.assertFalse(any("/issues/8/labels" in p for p in posted),
                             "a no-tier PR was labelled — the noise rule regressed")
            self.assertFalse(any("/issues/9/" in d for d in deleted),
                             "force_stale cleared a label off a stale UNKNOWN state")
            # standalone sweep unchanged: UNKNOWN neither sets nor clears
            posted.clear(); deleted.clear()
            bot.sync_rebase_labels("o/r", prs, merged_num=-1, dry_run=False)
            self.assertEqual(posted, [], "a standalone sweep labelled on UNKNOWN")
            self.assertEqual(deleted, [], "a standalone sweep cleared on UNKNOWN")
            # and the merge path passes force_stale
            src = (ROOT / "eval/k3_eval_bot.py").read_text()
            self.assertIn("sync_rebase_labels(args.repo, prs, winner, args.dry_run, "
                          "force_stale=True)", src,
                          "the tail-of-merge sweep still waits for BEHIND")
        finally:
            bot.gh = old

    def test_push_triggered_label_clear_is_safe_and_narrow(self):
        """rebase-label-sync.yml clears needs-rebase the moment a push answers it, so the
        'clears by itself' sentence in the bot's comment is true between sweeps.

        Its safety rests on ONE invariant: pull_request_target runs with base-repo WRITE
        access, which is only acceptable because the workflow never checks out or executes
        PR code — event payload and API calls, nothing else. A checkout of the head in this
        file hands a hostile fork write access to the repo, so that is the first thing this
        test refuses.

        Narrowness is the other half: it only REMOVES the label (adding is the sweep's job,
        tied to tier presence and frontier movement a push knows nothing about), decides
        from compare/behind_by (immediate and exact) rather than the UNKNOWN-settling
        mergeStateStatus, and verifies the delete rather than trusting the call."""
        import yaml as _y
        p = ROOT / ".github/workflows/rebase-label-sync.yml"
        self.assertTrue(p.exists(), "the push-triggered clear does not exist")
        txt = p.read_text()
        wf = _y.safe_load(txt)
        # THE invariant: write-context workflow, zero PR code. Asserted on the parsed
        # steps, not the text — the file's own warning comment names the footgun.
        steps = [s for job in wf["jobs"].values() for s in job.get("steps", [])]
        self.assertFalse([s for s in steps if "checkout" in str(s.get("uses", ""))],
                         "pull_request_target + checkout of PR code = write access for a "
                         "hostile fork; this workflow must stay API-only")
        self.assertFalse([s for s in steps if "github.event.pull_request.head.sha" in
                          str(s.get("run", "")) and "fetch" in str(s.get("run", ""))],
                         "fetching the head by sha is checkout by another name")
        trig = wf.get(True) or wf.get("on")
        self.assertIn("pull_request_target", trig,
                      "fork PRs get a read-only token under plain pull_request — the "
                      "decision would be right and the delete would fail")
        self.assertEqual(trig["pull_request_target"]["types"], ["synchronize"],
                         "only a push answers the label; other events are the sweep's job")
        # Narrow: removes only, decides by behind_by, fork-safe compare form, verified delete.
        self.assertIn("behind_by", txt, "mergeStateStatus reads UNKNOWN right after a push")
        self.assertIn("head.label", txt, "owner:branch is the fork-safe compare form")
        self.assertNotIn("labels[]=needs-rebase", txt, "this workflow must never ADD the label")
        self.assertIn('-X DELETE', txt)
        self.assertIn('index("needs-rebase")', txt, "an issued DELETE is not a removed label")
        # Rapid force-pushes race; only the newest head may decide.
        self.assertEqual(wf["concurrency"]["cancel-in-progress"], True)
        # No label, no work — and if the gate is dropped, every push on every PR runs this.
        self.assertIn("contains(github.event.pull_request.labels.*.name, 'needs-rebase')",
                      wf["jobs"]["sync"]["if"])

    def test_a_failed_round_still_has_a_log_path(self):
        """rounds/<main_sha>/ exists for the round that seals nothing -- and the first one
        that needed it published nothing anyway.

        _bail read main_sha out of locals(), which is unbound when the FRONTIER measurement
        is what failed, so the fallback reported "main is unknown" and dropped the log. That
        is precisely the round worth reading. main is therefore resolved before anything can
        fail, so the path exists before it is needed."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        main_fn = src[src.index("def main("):]
        self.assertNotIn('locals().get("main_sha"', main_fn,
                         "an unbound local is not a log path")
        self.assertLess(main_fn.index("main_sha = (_r.stdout"), main_fn.index("def _bail("),
                        "main must be resolved BEFORE the thing that can fail")
        self.assertIn("sha or main_sha", main_fn,
                      "_bail must fall back to the up-front sha")

    def test_every_round_publishes_a_log_even_when_nothing_seals(self):
        """The rule is ALWAYS, and receipt-indexing alone cannot satisfy it.

        publish_log used to be driven off `results`, so only a round that SEALED left a
        record. That lost exactly the rounds worth reading: the one that died on GPU
        contention mid-load minted no receipt and vanished, and a correctness fix scores
        `none`, is never sealed, and so was permanently unauditable.

        So a round that seals nothing falls back to rounds/<main_sha>/eval.log, and the
        early returns -- a frontier that would not measure, a lock that would not reconcile
        -- route through the same publish instead of returning past it. Those are the FAILED
        rounds; they must not be the ones that silently never land."""
        bot = self._bot()
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn("def publish_round_log(", src)
        self.assertIn('f"rounds/{main_sha[:12]}/eval.log"', src,
                      "a round that sealed nothing still needs a path")
        prl = src[src.index("def publish_round_log("):src.index("def merge_blockers(")]
        # sealed runs still get the log beside their receipt, and that is preferred
        self.assertLess(prl.index('f"runs/{rid}/eval.log"'), prl.index('f"rounds/'),
                        "a sealed run's log belongs with its receipt, not in the fallback")
        self.assertIn("if n:", prl, "the fallback must not double-publish a sealed round")
        main_fn = src[src.index("def main("):]
        self.assertIn("def _bail(", main_fn)
        for code in ("_bail(1", "_bail(3", "_bail(0"):
            self.assertIn(code, main_fn, f"an early return bypasses the log publish ({code})")
        # and the operator is told when it did not land, rather than it passing silently
        self.assertIn("this round is unauditable outside this", main_fn)

    def test_backfilled_log_needs_a_real_receipt(self):
        """--publish-log files a log somebody hands it rather than one the round produced,
        so it must refuse an id with no sealed run behind it.

        A log filed against a receipt that does not exist is worse than no log: it looks like
        evidence, and a reader has no measurement to check it against."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn('"--publish-log"', src)
        block = src[src.index("    if args.publish_log:"):src.index("    prs = list_prs(")]
        self.assertIn(f'contents/runs/{{rid}}', block,
                      "the receipt must be confirmed to exist before a log is filed on it")
        self.assertLess(block.index("contents/runs/"), block.index("publish_log("),
                        "existence has to be checked BEFORE the log is written")
        self.assertIn("--log-file", block)

    def test_rebase_label_clears_itself(self):
        """The frontier moves when something merges, so every other PR's tier goes stale by
        definition rather than by suspicion -- the denominator changed. The label says so.

        It must also clear itself once the branch is current: a label only a maintainer can
        remove turns a mechanical state into a queue someone has to babysit, and by then the
        miner has already done the work it was asking for.

        BLOCKED IS NOT STALE. GitHub reports BLOCKED for "waiting on the required approving
        review", which is an open PR's normal resting state here -- it says nothing about
        whether the branch trails main. Counting it as stale deadlocked #20: the label could
        never clear, and because needs-rebase is in NEVER_MERGE_LABELS it also blocked the
        merge, so the label survived on the strength of the review it was helping to prevent
        while the author had already rebased."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn('"-X", "DELETE", f"repos/{repo}/issues/{num}/labels/{REBASE_LABEL}"',
                      src, "needs-rebase is never removed once the branch is current")
        self.assertIn("elif has and current:", src)
        sync = src[src.index("def sync_rebase_labels("):src.index("def mark_merge_first(")]
        self.assertIn('stale = state in ("BEHIND", "DIRTY") or force_stale', sync,
                      "only a branch that trails main has a stale tier — except at the tail "
                      "of a merge, where the merge itself is the evidence")
        # BLOCKED means waiting on review, not behind — the label must clear. The
        # force_stale guard only suppresses clearing on the tail-of-merge sweep, where the
        # not-yet-recomputed state cannot say "current again" about anything.
        self.assertIn('current = (not force_stale) and state in ("CLEAN", "UNSTABLE", "BLOCKED")',
                      sync, "BLOCKED must clear the label in a standalone sweep")
        # UNKNOWN is 'GitHub has not finished computing mergeability': not evidence either
        # way, so it must neither set nor clear the label.
        self.assertNotIn('"UNKNOWN"', sync,
                         "acting on UNKNOWN labels people over an API race")
        # and the sweep must not run when the merge was blocked
        self.assertIn("if merge_winner(", src,
                      "rebase labelling must be conditional on something actually merging")

    def test_box_publish_failure_does_not_discard_the_receipt(self):
        """Signing and publishing are separate, and only signing has to happen on the box.

        kimi_k3_attest.py --publish git-pushes to the log, which needs write credentials
        wherever it runs. The box has none and should not be given any: a rented shared
        machine that can rewrite the ledger it is judged against is a worse trade than one
        that can only sign. So a failed box-side publish is expected, not fatal.

        Treating the harness's exit code as "not sealed" threw the receipt away and made
        the controller-side publish unreachable -- every run reported 0 sealed while three
        signed receipts sat on the box unpublished."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        seal = src[src.index("    if seal:"):src.index("    # Bind the measurement")]
        # the id must be taken whenever one was minted, not only on a clean exit
        self.assertIn('if rid:\n', seal,
                      "receipt id is only kept when the box-side publish succeeded")
        self.assertLess(seal.index('res["receipt_id"] = rid.group(1)'),
                        seal.index("rc != 0"),
                        "the receipt must be kept before the exit code is considered")
        # and sealed is still only claimed once the log confirms it
        self.assertIn("contents/runs/", src)

    def test_rebase_sweep_is_cheap_and_targeted(self):
        """The label is meant to clear itself the moment a miner rebases, which only works
        if checking costs seconds: requiring a 40-minute node eval to notice someone already
        did what they were asked would leave them under a stale label they cannot remove.

        It must also only touch PRs that HAVE a tier. needs-rebase means "your measured tier
        is stale because the frontier moved"; a PR with no eval:* tier has no stale tier, and
        noise on a label people are asked to act on is how labels stop being read."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn('"--rebase-sweep", action="store_true"', src)
        sweep = src[src.index("    if args.rebase_sweep:"):src.index("    evaluated = 0")]
        self.assertNotIn("evaluate(", sweep, "the sweep must not run a node eval")
        self.assertIn('if behind and not any(l.startswith("eval:") for l in labels)',
                      src, "the sweep would label PRs that were never scored")

    def test_rebase_comment_never_shows_the_sentinel(self):
        """merged_num is -1 for a standalone sweep, and interpolating it put a literal
        '#-1 merged' in front of a contributor. A sentinel that reaches a human is a bug
        regardless of what the code does with it."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn("if merged_num > 0", src,
                      "the sentinel is interpolated straight into the comment body")
        body = src[src.index("cause = (f\"#{merged_num} merged"):]
        self.assertIn('else "The frontier"', body[:400],
                      "no wording for the sweep case, so the sentinel leaks")

    def test_baseline_results_are_committable(self):
        """The `lock` job requires a committed bench/results JSON to back any pinned
        baseline. bench/.gitignore ignores results/ wholesale, so without an explicit
        negation that provenance rule is unsatisfiable in normal use and the pins can
        never be filled legitimately -- which would quietly turn the whole mechanism
        into decoration. Asserted with git itself rather than by reading the file."""
        import subprocess as sp
        probe = ROOT / "bench" / "results" / "kimi_k3_UD-Q2_K_XL_h200x8_baseline_probe.json"
        stray = ROOT / "bench" / "results" / "_ignore_probe.log"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("{}")
        stray.write_text("x")
        try:
            out = sp.run(["git", "status", "--porcelain", "--ignored",
                          "bench/results/"], cwd=ROOT, capture_output=True, text=True).stdout
            states = {}
            for line in out.splitlines():
                if not line.strip():
                    continue
                code, _, path = line.partition(" ")
                states[path.strip()] = code.strip()
            self.assertEqual(states.get(str(probe.relative_to(ROOT))), "??",
                             "a kimi_k3 baseline JSON must be committable, got "
                             f"{states.get(str(probe.relative_to(ROOT)))!r}")
            # And the negation must not be so broad that scratch output gets committed.
            self.assertEqual(states.get(str(stray.relative_to(ROOT))), "!!",
                             "stray run output must stay ignored, got "
                             f"{states.get(str(stray.relative_to(ROOT)))!r}")
        finally:
            probe.unlink(missing_ok=True)
            stray.unlink(missing_ok=True)

    def test_gitattributes_pins_shell_to_lf(self):
        text = (ROOT / ".gitattributes").read_text()
        self.assertIn("*.sh            text eol=lf", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
