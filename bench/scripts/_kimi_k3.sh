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
#
# TARGET QUANT: UD-Q2_K_XL (802 GiB, 90.4% top-1 vs lossless). This is the accuracy knee
# — UD-IQ1_S is 249 GiB smaller but drops to 78.9% top-1, which is too lossy to be the
# thing an optimization frontier is judged on. Q2_K_XL fits every milestone node below
# with room for the full 1M context.

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

# ---- quant selection ----
# PRIMARY_QUANT: UD-Q2_K_XL (default, target) | UD-IQ1_S | UD-IQ1_M | UD-IQ2_XXS | UD-Q8_K_XL
# Sizes are unsloth's published GiB, used for the fits-in-HBM precheck only.
kimi_k3_quant() { echo "${PRIMARY_QUANT:-UD-Q2_K_XL}"; }

# Shard counts as published in unsloth/Kimi-K3-GGUF. Pinned rather than discovered, so an
# interrupted download fails loudly instead of looking complete. 0 = don't enforce a count
# (an unknown quant, or KIMI_K3_SHARDS=0 to opt out).
kimi_k3_n_shards() {
  local n
  case "$(kimi_k3_quant)" in
    UD-IQ1_S)   n=14 ;;
    UD-IQ1_M)   n=15 ;;
    UD-IQ2_XXS) n=16 ;;
    UD-Q2_K_XL) n=19 ;;   # the target quant
    UD-Q4_K_XL) n=32 ;;
    UD-Q8_K_XL) n=34 ;;
    *)          n=0 ;;
  esac
  echo "${KIMI_K3_SHARDS:-$n}"
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

# ---- milestone node profiles ----
# KIMI_K3_NODE selects which milestone's hardware this run claims to be. It does NOT
# reconfigure anything by itself (detect_arch / gpu_count read the real box); it exists so a
# result carries the node it was measured on, and so a run on the wrong hardware is caught
# instead of silently landing in the wrong reference slot.
#
#   h200x8   M1  8x H200 SXM     141 GB  sm_90   Hopper
#   b200x8   M2  8x B200 SXM     180 GB  sm_100  Blackwell datacenter
#   b300x4   M3  4x+ B300        288 GB  sm_103* Blackwell Ultra
#
# *sm_103 for B300 is NOT verified on hardware here. detect_arch reads compute_cap at run
# time and wins; this value only seeds ARCH when nvidia-smi is unavailable. Confirm with
# `nvidia-smi --query-gpu=compute_cap --format=csv` on first contact, and note that the
# top-level CMakeLists arch list ("89;90;100;120;121") has no 103 — a native sm_103 build
# needs it added AND a CUDA toolkit that knows the target.
kimi_k3_node() { echo "${KIMI_K3_NODE:-h200x8}"; }

kimi_k3_node_arch() {
  case "$(kimi_k3_node)" in
    h200x8) echo 90 ;;
    b200x8) echo 100 ;;
    b300x4) echo "${KIMI_K3_B300_ARCH:-103}" ;;
    *)      echo "" ;;
  esac
}

kimi_k3_node_gpus() {
  case "$(kimi_k3_node)" in
    h200x8|b200x8) echo 8 ;;
    b300x4)        echo 4 ;;   # 4 is the floor; more is fine and is checked as >=
    *)             echo 0 ;;
  esac
}

# Total HBM the profile expects, GiB. Vendor "GB" for HBM is binary here: nvidia-smi
# reports ~144384 MiB for a 141 GB H200 (141 GiB), ~184320 for a 180 GB B200, ~294912 for
# a 288 GB B300. These are the documented expectations; gpu_vram_gib_total() is the truth.
kimi_k3_node_gib() {
  case "$(kimi_k3_node)" in
    h200x8) echo 1128 ;;   #  141 x 8
    b200x8) echo 1440 ;;   #  180 x 8
    b300x4) echo 1152 ;;   #  288 x 4
    *)      echo 0 ;;
  esac
}

kimi_k3_node_label() {
  case "$(kimi_k3_node)" in
    h200x8) echo "8x H200 SXM (sm_90, Hopper)" ;;
    b200x8) echo "8x B200 SXM (sm_100, Blackwell)" ;;
    b300x4) echo "4x+ B300 (sm_103, Blackwell Ultra)" ;;
    *)      echo "$(kimi_k3_node) (unknown profile)" ;;
  esac
}

# Warn — do not fail — when the box does not match the claimed profile. Failing here would
# block a legitimate run on a 6x B300 or a partially-allocated node; recording the mismatch
# is what actually matters, because a number filed under the wrong node is worse than none.
kimi_k3_check_node() {
  local want_arch want_gpus got_arch got_gpus
  want_arch="$(kimi_k3_node_arch)"; want_gpus="$(kimi_k3_node_gpus)"
  got_arch="$(detect_arch)"; got_gpus="$(gpu_count)"
  echo ">> node profile: $(kimi_k3_node_label)" >&2
  echo ">> detected    : sm_$got_arch, $got_gpus GPU(s), $(gpu_vram_gib_total) GiB HBM" >&2
  [ -n "$want_arch" ] || { echo ">> WARN: unknown KIMI_K3_NODE '$(kimi_k3_node)'" >&2; return 0; }
  [ "$got_arch" = "$want_arch" ] || \
    echo ">> WARN: arch mismatch — profile expects sm_$want_arch, box is sm_$got_arch" >&2
  [ "$got_gpus" -ge "$want_gpus" ] || \
    echo ">> WARN: GPU count below profile — expects >=$want_gpus, box has $got_gpus" >&2
}

# ---- context / memory sizing ----
# All three hardware milestones target the full 1M context, so the fit check has to price
# the KV cache in rather than assume a flat headroom.
KIMI_K3_MAX_CTX="${KIMI_K3_MAX_CTX:-1048576}"

# MLA stores K only (llama-kv-cache.cpp: is_mla() -> has_v = false):
#   24 full-attn layers * (kv_lora_rank 512 + qk_rope_head_dim 64) * 2 B = 27648 B/token
KIMI_K3_KV_BYTES_PER_TOKEN=27648

kimi_k3_kv_gib() {   # $1 = ctx (default KIMI_K3_MAX_CTX) — rounded up
  local ctx="${1:-$KIMI_K3_MAX_CTX}"
  awk -v c="$ctx" -v b="$KIMI_K3_KV_BYTES_PER_TOKEN" \
      'BEGIN{ printf "%d", int((c*b + 1073741823) / 1073741824) }'
}

# Headroom = KV at the target context + compute buffers + the KDA recurrent state.
# The compute-buffer term is per-GPU-scaled by llama.cpp and grows with ubatch; 24 GiB is
# a deliberately generous flat allowance, overridable once a node has been characterised.
kimi_k3_headroom_gib() {
  local kv; kv="$(kimi_k3_kv_gib)"
  echo $(( kv + ${KIMI_K3_COMPUTE_BUF_GIB:-24} + 1 ))
}

# ---- llama.cpp invocation ----
# Shared flags for every K3 llama.cpp call. Rationale per flag:
#   -ngl 99            93 layers, all offloaded
#   --split-mode layer llama.cpp sizes the per-GPU split from tensor bytes; the layers are
#                      not uniform (24 MLA vs 69 KDA), so do NOT hand-force -ts
#   --no-mmap          the weights must land in HBM, not the host page cache; on an 802 GiB
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

# Refuse to start a run that provably cannot fit — an 802 GiB model on a node with less HBM
# silently falls back to host offload and produces a baseline nobody can reproduce.
kimi_k3_check_fits() {
  local need have n headroom total
  need="$(kimi_k3_size_gib)"; have="$(gpu_vram_gib_total)"; n="$(gpu_count)"
  headroom="${KIMI_K3_HEADROOM_GIB:-$(kimi_k3_headroom_gib)}"
  total=$((need + headroom))
  echo ">> quant $(kimi_k3_quant): weights ~${need} GiB + ${headroom} GiB headroom" \
       "(KV $(kimi_k3_kv_gib) GiB @ ${KIMI_K3_MAX_CTX} ctx) = ${total} GiB needed" >&2
  echo ">> visible HBM ${have} GiB across ${n} GPU(s)" >&2
  [ "$need" -eq 0 ] && { echo ">> unknown quant size — skipping fit check" >&2; return 0; }
  [ "$have" -eq 0 ] && { echo ">> WARN: no nvidia-smi — cannot verify fit" >&2; return 0; }
  if [ "$have" -lt "$total" ]; then
    echo ">> FATAL: need ${total} GiB but only ${have} GiB is visible." >&2
    echo ">>        Options: a smaller PRIMARY_QUANT, a lower KIMI_K3_MAX_CTX, or a bigger" >&2
    echo ">>        node. Partial-offload baselines are not comparable and not accepted." >&2
    return 1
  fi
}
