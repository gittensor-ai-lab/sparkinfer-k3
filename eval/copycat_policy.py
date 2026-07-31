"""Shared copycat detection thresholds — used by copycat_guard, pr_eval_bot, copycat_sweep."""

# PR-level containment (shared file + added-line overlap required).
#
# THREE TIERS, and the gap between them is deliberate — the cost of being wrong is not
# symmetric. A missed copycat costs one PR's emission. A false block costs a real
# contributor their access permanently, and there is no appeal path in an automated
# system, so the block bar is set where near-verbatim reproduction is the only
# remaining explanation.
#
#   >= 90%   BLOCK   denylist + close. Effectively permanent.
#   80-90%   CLOSE   comment + auto-close. Recoverable: they can open a real PR next.
#   70-80%   JUDGE   llm-judge label, semantic review, label removed if cleared.
#            < 70%   ignore.
COPYCAT_BLOCK = 0.90   # >=90% → denylist + close (permanent)
COPYCAT_WARN = 0.80    # 80-90% → comment + auto-close
COPYCAT_JUDGE = 0.70   # 70-80% → llm-judge label + semantic review
MAX_WARNINGS = 1       # a 70-90% hit closes immediately; no strike accumulation

# Small PR guard — avoids ratio explosions on tiny diffs
MIN_ADDED_LINES = 15
LITERAL_BLOCK = 0.98   # still block tiny PRs if ≥98% literal

# Per-function dilution catch (near-verbatim kernel embedded in larger PR).
# WARN only — never escalates to block on its own (PR-level containment must be ≥85%).
FUNC_BLOCK_WARN = 0.92
# Ignore tiny device helpers / shared dequant that already live on main (PR #566 FP:
# pfm_scale_min_k4 matched 100% across MoE PRs but is identical on main).
FUNC_MIN_BODY_TOKENS = 80
# Skip func-layer warn when PR-level overlap is negligible — real dilution still
# clears this when a whole kernel is pasted into a larger diff.
FUNC_MIN_PR_LEVEL = 0.15
# Line-set overlap vs origin/main for the same file → treat as shared infrastructure.
FUNC_MAIN_SKIP = 0.90

# Structural (Levenshtein + bigram) layer disabled — too many FPs on independent
# contributors converging on the same optimization pattern.
STRUCTURAL_ENABLED = False

# LLM semantic judge: ON for the 50-70% band only.
#
# That band is exactly where containment stops being evidence. 70-80% overlap is the
# signature of BOTH a copycat that renamed things and an independent contributor who
# touched the same hot function — the two are indistinguishable by token overlap, which
# is why the structural layer below was disabled for false positives. A semantic judge
# is the only tool that can separate them, so it is scoped to the band where the
# arithmetic has genuinely run out rather than used as a general-purpose oracle.
#
# It can only ever ADD an llm-judge label for human review, and the label is removed
# when the judge clears the PR. It cannot close and it cannot block.
LLM_ENABLED = True

# Back-compat alias
COPYCAT_CONTAINMENT = COPYCAT_BLOCK

# Copycat reference pool: only still-open PRs (excludes closed and merged).
COPYCAT_REFERENCE_STATE = "open"


def skip_copycat_scoring(added_lines, containment):
    """Skip borderline scoring on tiny PRs unless near-literal copy."""
    n = len(added_lines)
    if n >= MIN_ADDED_LINES:
        return False
    if n >= 3 and containment >= LITERAL_BLOCK:
        return False
    return True
