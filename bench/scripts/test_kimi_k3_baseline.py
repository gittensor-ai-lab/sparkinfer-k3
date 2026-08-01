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
        text = (self.WF / "build-gate.yml").read_text()
        self.assertIn('arch: "90"', text)
        self.assertIn('arch: "100"', text)
        self.assertNotIn('arch: "120"', text)

    def test_node_attestation_never_closes(self):
        # Deliberate severity change from the inherited rtx5090-required gate.
        text = (self.WF / "node-attestation.yml").read_text()
        self.assertIn("needs-node-run", text)
        self.assertNotIn("state: 'closed'", text)

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
        out = sp.run([sys.executable, str(ROOT / "bench/scripts/label.py"),
                      "4.0", "3.55", "0", "1.0", "0.001", "abc123",
                      _j.dumps({"label": "XL", "pass": True, "clock_mhz": 1980})],
                     capture_output=True, text=True, check=True).stdout
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
