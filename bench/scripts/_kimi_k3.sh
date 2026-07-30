#!/usr/bin/env bash
# Kimi K3 baseline definitions — sourced by kimi_k3_baseline.sh and
# kimi_k3_reference_server.sh. Requires _common.sh to be sourced first.
#
# HF weights: https://huggingface.co/unsloth/Kimi-K3-GGUF
# Baseline engine: https://github.com/unslothai/llama.cpp @ kimi-k3-fullsize-vision (PR #48)
#
# WHY A FORK IS THE BASELINE: Kimi K3 has 896 routed experts. Upstream
# ggml-org/llama.cpp asserts n_expert <= LLAMA_MAX_EXPERTS (512), so it cannot load the
# model at all — there is no upstream number to compare against. The fork raises the cap
# to 1024, adds LLM_ARCH_KIMI_K3 (hybrid KDA + MLA, situ activation, latent MoE,
# cross-layer attention residual, MLA output gate), and raises the graph node budget to
# n_tokens*160 because the generic *40 budget is exhausted at ubatch 3840.

KIMI_K3_REPO="${KIMI_K3_REPO:-unsloth/Kimi-K3-GGUF}"
KIMI_K3_MODELS_DIR="${KIMI_K3_MODELS_DIR:-${MODELS_DIR:-/workspace/models}_k3}"

# Point the shared llama.cpp builder at the FORK pin from reference.lock. Doing it here
# rather than in reference.lock itself means sourcing _kimi_k3.sh is the single act that
# switches the baseline engine — the Qwen evals never source this file, so they keep the
# upstream ggml-org pin and no fork commit can leak into their verdicts.
export LLAMACPP_REPO="${KIMI_K3_LLAMACPP_REPO:-https://github.com/unslothai/llama.cpp}"
export LLAMACPP_REF="${KIMI_K3_LLAMACPP_REF:-refs/pull/48/head}"
export LLAMACPP_COMMIT="${KIMI_K3_LLAMACPP_COMMIT:-efc8bc38f0a9950cbb10ccef2cf48b951c39d3b2}"
# Separate checkout: sharing .llamacpp with the upstream pin makes the two evals fight over
# one tree and trigger a full rebuild on every alternation.
export LLAMACPP_DIR="${KIMI_K3_LLAMACPP_DIR:-${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/.llamacpp-k3}"

# PRIMARY_QUANT: UD-IQ1_S (default, 553 GiB) | UD-IQ1_M | UD-IQ2_XXS | UD-Q2_K_XL | UD-Q8_K_XL
# Sizes are unsloth's published GiB, used only for the fits-in-HBM precheck.
kimi_k3_quant() { echo "${PRIMARY_QUANT:-UD-IQ1_S}"; }

kimi_k3_n_shards() {
  case "$(kimi_k3_quant)" in
    UD-IQ1_S)   echo "${KIMI_K3_IQ1S_SHARDS:-14}" ;;
    *)          echo "${KIMI_K3_SHARDS:-0}" ;;   # 0 = don't enforce a count
  esac
}

kimi_k3_size_gib() {
  case "$(kimi_k3_quant)" in
    UD-IQ1_S)   echo 553 ;;
    UD-IQ1_M)   echo 604 ;;
    UD-IQ2_XXS) echo 662 ;;
    UD-Q2_K_XL) echo 802 ;;
    UD-Q4_K_XL) echo 1407 ;;
    UD-Q8_K_XL) echo 1453 ;;
    *)          echo 0 ;;
  esac
}

# Shards live in a per-quant subdirectory in the HF repo.
kimi_k3_first_shard() {
  local q; q="$(kimi_k3_quant)" ; local n; n="$(kimi_k3_n_shards)"
  if [ "$n" -gt 0 ]; then
    printf '%s/Kimi-K3-%s-00001-of-%05d.gguf' "$q" "$q" "$n"
    return
  fi
  # Unknown shard count: resolve from whatever is already on disk.
  local f
  f="$(find "$KIMI_K3_MODELS_DIR/$q" -name "Kimi-K3-${q}-00001-of-*.gguf" 2>/dev/null | sort | head -1)"
  if [ -n "$f" ]; then
    printf '%s' "${f#"$KIMI_K3_MODELS_DIR/"}"
  else
    printf '%s/Kimi-K3-%s-00001-of-00001.gguf' "$q" "$q"
  fi
}

kimi_k3_include_glob() { printf '*%s*' "$(kimi_k3_quant)"; }

kimi_k3_gguf() { printf '%s/%s' "$KIMI_K3_MODELS_DIR" "$(kimi_k3_first_shard)"; }
kimi_k3_mmproj() { printf '%s/%s' "$KIMI_K3_MODELS_DIR" "${KIMI_K3_MMPROJ:-mmproj-BF16.gguf}"; }

# Path of the pinned shard manifest for the selected quant (may not exist yet).
kimi_k3_manifest() {
  printf '%s/kimi_k3_%s.sha256' "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" \
         "$(kimi_k3_quant | tr 'A-Z-' 'a-z_')"
}

# ---- llama.cpp invocation ----
# Shared flags for every K3 llama.cpp call. Rationale per flag:
#   -ngl 99            93 layers, all offloaded
#   --split-mode layer llama.cpp sizes the per-GPU split from tensor bytes; the layers are
#                      not uniform (24 MLA vs 69 KDA), so do NOT hand-force -ts
#   --no-mmap          the weights must land in HBM, not the host page cache; on a 553 GiB
#                      read, mmap turns load into a random-read storm
#   -b / -ub           compute buffers are per-GPU and scale with ubatch; the fork's
#                      n_tokens*160 node budget is sized for large ubatch but VRAM is not
kimi_k3_llama_flags() {
  echo "-ngl ${KIMI_K3_NGL:-99}" \
       "--split-mode ${KIMI_K3_SPLIT_MODE:-layer}" \
       "--no-mmap" \
       "-b ${KIMI_K3_BATCH:-2048}" \
       "-ub ${KIMI_K3_UBATCH:-512}" \
       "${KIMI_K3_EXTRA_FLAGS:-}"
}

# Server-only flags. K3 is a hybrid recurrent arch: llama.cpp cannot context-shift or
# restore slots for it, so without --no-context-shift a long eval dies mid-run.
kimi_k3_server_flags() {
  echo "--no-context-shift" "--no-jinja"
}

# Refuse to start a run that provably cannot fit — a 553 GiB model on a node with less HBM
# silently falls back to host offload and produces a baseline nobody can reproduce.
kimi_k3_check_fits() {
  local need have n headroom
  need="$(kimi_k3_size_gib)"; have="$(gpu_vram_gib_total)"; n="$(gpu_count)"
  headroom="${KIMI_K3_HEADROOM_GIB:-40}"
  echo ">> quant $(kimi_k3_quant): weights ~${need} GiB, visible HBM ${have} GiB across ${n} GPU(s)" >&2
  [ "$need" -eq 0 ] && { echo ">> unknown quant size — skipping fit check" >&2; return 0; }
  [ "$have" -eq 0 ] && { echo ">> WARN: no nvidia-smi — cannot verify fit" >&2; return 0; }
  if [ "$have" -lt "$((need + headroom))" ]; then
    echo ">> FATAL: $need GiB weights + $headroom GiB headroom > $have GiB HBM." >&2
    echo ">>        Pick a smaller quant (PRIMARY_QUANT) or a bigger node. Partial-offload" >&2
    echo ">>        baselines are not comparable and are not accepted." >&2
    return 1
  fi
}
