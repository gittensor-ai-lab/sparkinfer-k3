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
        label a real speedup BASELINE. refdata/ is pinned too because it IS the accuracy
        answer key."""
        src = (ROOT / "eval/k3_eval_bot.py").read_text()
        self.assertIn("git checkout -q origin/main -- bench/scripts bench/refdata", src,
                      "the bot grades with the PR's own harness")
        self.assertLess(src.index("origin/main -- bench/scripts"),
                        src.index("kimi_k3_eval.sh --node"),
                        "the harness must be restored BEFORE the eval runs")

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

    def test_bench_must_prove_it_seeked(self):
        """kimi_k3_tp_bench is built from the PR's OWN runtime/ -- only bench/scripts and
        bench/refdata are restored from the protected branch. So the flags that select the
        scored context live in code the PR controls, and the arg loop IGNORES unknown flags
        rather than rejecting them, which makes a silent stub the easiest possible edit.

        A --seek that does nothing measures ctx 64 and reports it as 128k: 10.34 vs 1.00
        tok/s measured, so roughly 10x the score. The harness therefore requires the bench to
        announce the seek AND to name the context it was asked for."""
        ev = (ROOT / "bench/scripts/kimi_k3_eval.sh").read_text()
        self.assertIn('grep -q "seek: position -> .*(ctx alloc $SCORED_CTX)"', ev,
                      "a stubbed --seek would score short-context decode as 128k")
        guard = ev[ev.index("seek: position -> .*(ctx alloc"):]
        self.assertIn("exit 1", guard[:600], "the guard must refuse, not warn")
        # and the bench must actually emit that line
        cpp = (ROOT / "runtime/examples/kimi_k3_tp_bench.cpp").read_text()
        self.assertIn('"seek: position -> %d (ctx alloc %d)\\n"', cpp)

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
        self.assertIn('stale = state in ("BEHIND", "DIRTY")', sync,
                      "only a branch that trails main has a stale tier")
        self.assertIn('current = state in ("CLEAN", "UNSTABLE", "BLOCKED")', sync,
                      "BLOCKED means waiting on review, not behind — the label must clear")
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
