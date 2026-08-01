#!/usr/bin/env python3
"""Unit tests for the deterministic eval label policy.

Run from the repo root:
  python3 bench/scripts/test_label.py
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABEL = ROOT / "bench" / "scripts" / "label.py"


def score(tps, frontier=100.0, ceiling=200.0, top1=0.97, kl=0.02, commit="deadbee",
          prov=None, env=None, guards=None):
    cmd = [sys.executable, str(LABEL), str(tps), str(frontier), str(ceiling),
           str(top1), str(kl), commit]
    if prov is not None:
        cmd.append(json.dumps(prov, separators=(",", ":")))
    run_env = os.environ.copy()
    # Guards are opt-in per case. Popping first keeps a value in the developer's own shell from
    # leaking in and silently gating every unrelated case.
    run_env.pop("SPARKINFER_GUARDS", None)
    if guards is not None:
        run_env["SPARKINFER_GUARDS"] = json.dumps(guards, separators=(",", ":"))
    # label.py deliberately has NO default llama anchor: it used to default to 365.85, a Qwen
    # constant, so anything that forgot to pass one was silently scored on Qwen's maturity
    # curve. These tests use synthetic frontier=100 numbers rather than K3's, so they pin a
    # synthetic anchor too; a test that cares about the basis overrides it via env=.
    run_env.setdefault("SPARKINFER_DIFFICULTY_REF", "400")
    if env:
        run_env.update(env)
    out = subprocess.check_output(cmd, text=True, env=run_env).strip()
    assert out.startswith("RESULT_JSON "), out
    return json.loads(out[len("RESULT_JSON "):])


class LabelPolicyTest(unittest.TestCase):
    # A synthetic llama anchor. These cases use frontier=100 rather than K3 numbers because
    # they exercise the correctness GATE, not the tier; 400 is chosen so the incidental speed
    # tier lands in M. Previously this test passed no anchor at all and silently inherited
    # label.py's old 365.85 default -- a Qwen constant -- so it was asserting Qwen behaviour
    # in a K3 repo. label.py now refuses an unset anchor, which is what surfaced it.
    SYNTH_REF = {"SPARKINFER_DIFFICULTY_REF": "400"}

    def test_correctness_gate_rejects_bad_accuracy(self):
        # +30 over frontier=100 clears significance; anchored tier is M (30/400 = 7.5%).
        low_top1 = score(130, top1=0.89, kl=0.02)
        self.assertEqual(low_top1["label"], "REJECT")
        self.assertFalse(low_top1["pass"])
        # Speed tier still reported so prefill/decode progress can be annotated.
        self.assertEqual(low_top1["speed_label"], "M")
        self.assertEqual(low_top1["pct_over_frontier"], 30.0)

        high_kl = score(130, top1=0.97, kl=0.21)
        self.assertEqual(high_kl["label"], "REJECT")
        self.assertFalse(high_kl["pass"])
        self.assertEqual(high_kl["speed_label"], "M")

    def test_significance_gate_is_strictly_above_two_percent(self):
        exact_floor = score(102.0)
        self.assertEqual(exact_floor["label"], "none")
        self.assertTrue(exact_floor["pass"])
        self.assertEqual(exact_floor["pct_over_frontier"], 2.0)

        just_above = score(102.1)
        self.assertEqual(just_above["label"], "XS")
        self.assertTrue(just_above["pass"])

    def test_llama_anchored_buckets_without_difficulty_boost(self):
        # The label tier is sized against the llama.cpp reference (DIFF_REF), not the frontier — so
        # equal tok/s of work earns the same tier at any maturity. With DIFF_REF=365.85 and frontier=100
        # (below the reference -> no boost), the S/M/L/XL thresholds (3.5/6/10/18% of llama) fall at
        # delta ~= 12.8/22/36.6/65.9 tok/s. A verified-but-smaller gain floors at XS.
        env = {"SPARKINFER_DIFFICULTY_BOOST": "0", "SPARKINFER_DIFFICULTY_REF": "365.85"}
        self.assertEqual(score(105.0, env=env)["label"], "XS")   # +5: verified over frontier, < S vs llama
        self.assertEqual(score(113.0, env=env)["label"], "S")    # +13 = 3.55% of llama
        self.assertEqual(score(122.0, env=env)["label"], "M")    # +22 = 6.01%
        self.assertEqual(score(137.0, env=env)["label"], "L")    # +37 = 10.11%
        self.assertEqual(score(166.0, env=env)["label"], "XL")   # +66 = 18.04%

    def test_unoptimized_model_does_not_mint_xl_from_low_hanging_fruit(self):
        # The unfairness this policy fixes: over an un-optimized frontier a small absolute gain used to
        # explode to XL. Sized against llama.cpp it earns a fair tier. (frontier 23, llama 190.)
        env = {"SPARKINFER_DIFFICULTY_BOOST": "1", "SPARKINFER_DIFFICULTY_REF": "190"}
        self.assertEqual(score(28.0, frontier=23.0, env=env)["label"], "XS")   # +5 (+22% over frontier) but 2.6% of llama
        self.assertEqual(score(171.0, frontier=23.0, env=env)["label"], "XL")  # a real 7x IS an XL (78% of llama)

    def test_difficulty_boost_high_cap_still_capped_at_twice_raw_speedup(self):
        # Even with a generous DIFF_MAX, g_eff cannot exceed 2× pct_over_frontier.
        res = score(484.79, frontier=469.13, ceiling=366.0, top1=0.9612, kl=0.0175,
                    commit="c30bf58",
                    env={"SPARKINFER_DIFFICULTY_BOOST": "1",
                         "SPARKINFER_DIFFICULTY_REF": "365.85",
                         "SPARKINFER_DIFFICULTY_K": "8",
                         "SPARKINFER_DIFFICULTY_MAX": "4"})
        self.assertEqual(res["label"], "M")
        self.assertEqual(res["pct_over_frontier"], 3.3)
        self.assertGreaterEqual(res["effective_pct"], 6.0)
        self.assertLess(res["effective_pct"], 10.0)

    def test_difficulty_boost_default_cap_is_one_point_five_x(self):
        # Default SPARKINFER_DIFFICULTY_MAX=1.5 with strict 2× raw cap.
        res = score(484.79, frontier=469.13, ceiling=366.0, top1=0.9612, kl=0.0175,
                    commit="c30bf58",
                    env={"SPARKINFER_DIFFICULTY_BOOST": "1",
                         "SPARKINFER_DIFFICULTY_REF": "365.85",
                         "SPARKINFER_DIFFICULTY_K": "8"})
        self.assertEqual(res["label"], "M")
        self.assertEqual(res["difficulty_mult"], 1.5)
        self.assertEqual(res["pct_over_frontier"], 3.3)
        self.assertGreaterEqual(res["effective_pct"], 6.0)
        self.assertLess(res["effective_pct"], 10.0)

    def test_strict_cap_limits_per_context_llama_ref_inflation(self):
        # Qwen3.6 @ 128: a low per-context llama ref must not push a ~3% raw gain to L.
        res = score(479.96, frontier=465.41, ceiling=366.0, top1=0.967, kl=0.0281,
                    commit="d12766f",
                    env={"SPARKINFER_DIFFICULTY_BOOST": "1",
                         "SPARKINFER_DIFFICULTY_REF": "275.81",
                         "SPARKINFER_DIFFICULTY_K": "8"})
        self.assertEqual(res["label"], "M")
        self.assertEqual(res["pct_over_frontier"], 3.1)
        self.assertGreaterEqual(res["effective_pct"], 6.0)
        self.assertLess(res["effective_pct"], 10.0)

    def test_long_context_metadata_is_preserved_in_verdict(self):
        # Anchor pinned explicitly: this case asserts XL, which needs 70/ref >= 0.18, so the
        # helper's default 400 would land it in L. Stating it here keeps the tier the test
        # is asserting a property of the input rather than of a shared constant.
        prov = {
            "eval_mode": "longctx",
            "score_context": 16384,
            "best_context_label": "16k-context",
            "context_gains_pct": {"128-context": -2.04, "512-context": 1.25, "4k-context": 0.5, "16k-context": 20.0, "32k-context": 12.5},
            "ctx_128_tps": 480.0,
            "ctx_512_tps": 405.0,
            "ctx_4096_tps": 198.0,
            "ctx_16384_tps": 120.0,
            "ctx_32768_tps": 80.0,
            "guard_128_baseline": 490.0,
            "guard_128_ratio": 0.9796,
            "guard_128_pass": True,
            "guard_512_baseline": 400.0,
            "guard_512_ratio": 1.0125,
            "guard_512_pass": True,
            "guard_4k_baseline": 197.0,
            "guard_4k_ratio": 1.0051,
            "guard_4k_pass": True,
            "guard_16k_baseline": 100.0,
            "guard_16k_ratio": 1.2,
            "guard_16k_pass": True,
            "guard_32k_baseline": 71.11,
            "guard_32k_ratio": 1.125,
            "guard_32k_pass": True,
        }
        res = score(170.0, frontier=100.0, prov=prov, env={"SPARKINFER_DIFFICULTY_REF": "350"})   # +70 = 19.1% of llama -> XL
        self.assertEqual(res["label"], "XL")
        self.assertEqual(res["eval_mode"], "longctx")
        self.assertEqual(res["score_context"], 16384)
        self.assertEqual(res["best_context_label"], "16k-context")
        self.assertEqual(res["context_gains_pct"]["16k-context"], 20.0)
        self.assertEqual(res["context_gains_pct"]["32k-context"], 12.5)
        self.assertEqual(res["ctx_128_tps"], 480.0)
        self.assertEqual(res["ctx_512_tps"], 405.0)
        self.assertEqual(res["ctx_4096_tps"], 198.0)
        self.assertEqual(res["ctx_32768_tps"], 80.0)
        self.assertTrue(res["guard_128_pass"])
        self.assertTrue(res["guard_512_pass"])
        self.assertTrue(res["guard_4k_pass"])
        self.assertTrue(res["guard_16k_pass"])
        self.assertTrue(res["guard_32k_pass"])

    def test_baseline_label_when_no_frontier_exists(self):
        res = score(120.0, frontier=0.0)
        self.assertEqual(res["label"], "BASELINE")
        self.assertTrue(res["pass"])

    def test_prefill_frontier_scale_scores_L_at_eleven_pct(self):
        # PR #387-shaped: ~11% sequential prefill pp over same-box main; llama-batched ref is wrong scale.
        res = score(320.28, frontier=288.21, top1=0.94, kl=0.03,
                    env={"SPARKINFER_DIFFICULTY_REF": "0", "SPARKINFER_DIFFICULTY_BOOST": "1"})
        self.assertEqual(res["label"], "L")
        self.assertGreater(res["pct_over_frontier"], 10.0)

    def test_batched_prefill_frontier_scale_scores_XL_at_forty_six_pct(self):
        # PR #422-shaped: batched sparkinfer pp ~6k vs same-box main ~4151; llama ~11k understates tier.
        res = score(6062.59, frontier=4151.38, top1=0.94, kl=0.03,
                    env={"SPARKINFER_DIFFICULTY_REF": "0", "SPARKINFER_DIFFICULTY_BOOST": "1"})
        self.assertEqual(res["label"], "XL")
        self.assertGreater(res["pct_over_frontier"], 40.0)


class NoRegressionGuardTest(unittest.TestCase):
    """The tier is earned at one context; the others may not be paid out to buy it.

    Shape: score 128k, guard 128 / 4k / 32k. `base` is the RATCHETED per-context frontier that the
    merge path maintains with max(old, measured) — see the SPARKINFER_GUARDS comment in label.py for
    why a plain assignment there turns GUARD_TOL into a slow leak.
    """

    def guards(self, c128=3.61, c4k=3.40, c32k=3.10):
        return {"128": {"tps": c128, "base": 3.55},
                "4096": {"tps": c4k, "base": 3.40},
                "32768": {"tps": c32k, "base": 3.10}}

    def test_absent_guards_are_inert(self):
        # The whole block must stay dormant until the bench can actually measure the contexts.
        res = score(130.0)
        self.assertEqual(res["label"], "M")
        self.assertNotIn("guards", res)
        self.assertNotIn("guards_pass", res)

    def test_all_guards_holding_leaves_the_tier_alone(self):
        res = score(130.0, guards=self.guards())
        self.assertEqual(res["label"], "M")
        self.assertTrue(res["pass"])
        self.assertTrue(res["guards_pass"])
        self.assertEqual([g["ctx"] for g in res["guards"]], [128, 4096, 32768])
        self.assertTrue(all(g["pass"] for g in res["guards"]))

    def test_regression_at_one_context_rejects_despite_a_real_gain(self):
        # 32k drops 12.9% while the scored context gains 30%. This is a trade, not a speedup.
        res = score(130.0, guards=self.guards(c32k=2.70))
        self.assertEqual(res["label"], "REJECT")
        self.assertFalse(res["pass"])
        self.assertFalse(res["guards_pass"])
        self.assertIn("32768", res["reason"])
        # The gain is still reported — a rejected PR that really was faster should not have to
        # argue about whether the number was real.
        self.assertEqual(res["speed_label"], "M")
        self.assertEqual(res["pct_over_frontier"], 30.0)

    def test_tolerance_absorbs_noise_but_not_decay(self):
        # Default TOL 0.98: a 1% wobble is measurement, a 2.5% drop is not.
        self.assertTrue(score(130.0, guards=self.guards(c4k=3.366))["guards_pass"])   # -1.0%
        self.assertEqual(score(130.0, guards=self.guards(c4k=3.315))["label"], "REJECT")  # -2.5%

    def test_unset_baseline_cannot_regress(self):
        # A context measured for the first time has nothing to fall below; it must not block a PR.
        res = score(130.0, guards={"131072": {"tps": 1.2, "base": 0}})
        self.assertTrue(res["guards_pass"])
        self.assertIsNone(res["guards"][0]["pct"])

    def test_correctness_outranks_regression_in_the_reason(self):
        # Both gates fail. A wrong answer is disqualifying on its own, so it owns the reason.
        res = score(130.0, top1=0.80, guards=self.guards(c32k=2.70))
        self.assertEqual(res["label"], "REJECT")
        self.assertIn("correctness gate", res["reason"])
        self.assertFalse(res["guards_pass"])

    def test_latency_guards_respect_direction(self):
        # LOWER_IS_BETTER: a guard passes when the measured TTFT is no HIGHER than its baseline.
        env = {"SPARKINFER_LABEL_LOWER_IS_BETTER": "1", "SPARKINFER_DIFFICULTY_REF": "0"}
        faster = score(0.80, frontier=1.00, env=env, guards={"4096": {"tps": 1.90, "base": 2.00}})
        self.assertTrue(faster["guards_pass"])
        slower = score(0.80, frontier=1.00, env=env, guards={"4096": {"tps": 2.40, "base": 2.00}})
        self.assertEqual(slower["label"], "REJECT")

    def test_provenance_cannot_forge_a_passing_guard(self):
        # The forgery this exists to stop: a receipt that says the guards held when they did not.
        res = score(130.0, guards=self.guards(c32k=2.70),
                    prov={"guards_pass": True, "guards": [], "reason": "all good",
                          "node": "h200x8"})
        self.assertEqual(res["label"], "REJECT")
        self.assertFalse(res["guards_pass"])
        self.assertIn("32768", res["reason"])
        self.assertEqual(res["provenance_rejected"], ["guards", "guards_pass", "reason"])
        self.assertEqual(res["node"], "h200x8")          # non-verdict provenance still passes through

    def test_qwen_provenance_guard_fields_are_still_writable(self):
        # The Qwen long-context path ships flat guard_<ctx>_* through provenance. The K3 block is
        # namespaced away from those on purpose, so adding it must not break that path.
        res = score(130.0, prov={"score_context": 16384, "guard_32k_pass": True,
                                 "guard_32k_baseline": 71.11})
        self.assertEqual(res["score_context"], 16384)
        self.assertTrue(res["guard_32k_pass"])
        self.assertNotIn("provenance_rejected", res)

    def test_scored_context_is_recorded(self):
        res = score(130.0, env={"SPARKINFER_SCORED_CONTEXT": "131072"})
        self.assertEqual(res["scored_context"], 131072)

    def test_unparseable_guards_refuse_to_score(self):
        # A malformed guard set is silently NO guard set — the one failure mode that must be loud.
        env = os.environ.copy()
        env["SPARKINFER_DIFFICULTY_REF"] = "400"
        env["SPARKINFER_GUARDS"] = "{not json"
        p = subprocess.run([sys.executable, str(LABEL), "130", "100", "200", "0.97", "0.02", "abc"],
                           env=env, capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("not valid JSON", p.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
