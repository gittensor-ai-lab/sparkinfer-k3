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

    def test_target_quant_is_q2kxl(self):
        # UD-Q2_K_XL is the accuracy knee (90.4% top-1). UD-IQ1_S is 249 GiB smaller but
        # 78.9% top-1 — too lossy to judge an optimization frontier on.
        self.assertEqual(bash("kimi_k3_quant"), "UD-Q2_K_XL")
        self.assertEqual(bash("kimi_k3_first_shard"),
                         "UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf")
        self.assertEqual(bash("kimi_k3_n_shards"), "19")
        self.assertEqual(bash("kimi_k3_include_glob"), "*UD-Q2_K_XL*")
        self.assertEqual(bash("kimi_k3_size_gib"), "802")

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
        self.assertTrue(bash("kimi_k3_manifest").endswith("kimi_k3_ud_q2_k_xl.sha256"))

    def test_gguf_path_joins_models_dir(self):
        got = bash("kimi_k3_gguf", {"KIMI_K3_MODELS_DIR": "/mnt/w"})
        self.assertEqual(got, "/mnt/w/UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf")


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
            # every hardware milestone is a 1M-context milestone
            self.assertIn("max_context: 1048576", text)
            self.assertIn("kv_at_1m: 27.00", text)

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

    def test_default_max_ctx_is_1m(self):
        self.assertEqual(bash('echo "$KIMI_K3_MAX_CTX"'), "1048576")

    def test_headroom_includes_kv_at_1m(self):
        # 27 GiB KV + 24 GiB compute buffers + 1 GiB state
        self.assertEqual(int(bash("kimi_k3_headroom_gib")), 52)

    def test_lower_max_ctx_shrinks_headroom(self):
        got = int(bash("kimi_k3_headroom_gib", {"KIMI_K3_MAX_CTX": "32768"}))
        self.assertEqual(got, 26)   # 1 + 24 + 1


class LlamaFlagTest(unittest.TestCase):
    def test_required_flags_present(self):
        flags = bash("kimi_k3_llama_flags")
        for needed in ("-ngl 99", "--split-mode layer", "--no-mmap", "-b 2048", "-ub 512"):
            self.assertIn(needed, flags)

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
        base = {"PRIMARY_QUANT": "UD-Q4_K_XL", "KIMI_K3_NODE": "b200x8"}
        self.assertEqual(bash_rc("kimi_k3_check_fits", self._with_fake_smi(self.B200, 8, base)), 1)
        at32k = {**base, "KIMI_K3_MAX_CTX": "32768"}
        self.assertEqual(bash_rc("kimi_k3_check_fits", self._with_fake_smi(self.B200, 8, at32k)), 0)

    def test_smaller_quant_rescues_an_undersized_node(self):
        # 5x H200 (705 GiB) can't hold the 802 GiB target, but holds IQ1_S (553 + 26 = 579).
        small = {"PRIMARY_QUANT": "UD-IQ1_S", "KIMI_K3_MAX_CTX": "32768"}
        self.assertEqual(bash_rc("kimi_k3_check_fits", self._with_fake_smi(self.H200, 5, small)), 0)
        self.assertEqual(bash_rc("kimi_k3_check_fits", self._with_fake_smi(self.H200, 5)), 1)

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
        self.assertIn("Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf", out)
        self.assertIn("--split-mode layer", out)
        self.assertIn("802 GiB + 52 GiB headroom", out)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
