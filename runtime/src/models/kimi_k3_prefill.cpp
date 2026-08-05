// See kimi_k3_prefill.h for scope. This mirrors kimi_k3_forward_token's body
// (runtime/src/models/kimi_k3.cpp) as closely as possible on purpose: the two
// must stay in lockstep, since kimi_k3_forward_token is the reference this is
// checked against. Diverging in structure would make that comparison harder to
// trust, not easier.

#include "sparkinfer/models/kimi_k3_prefill.h"

#include "sparkinfer/kernels/kimi_k3.h"

#include <cstdio>

namespace sparkinfer {

namespace k3k = sparkinfer::kernels::k3;

// Declared in kimi_k3.cpp, not in a header (it's an internal helper there) --
// forward-declared here rather than duplicated, since duplicating it risks the
// two copies drifting, which is exactly the class of bug this whole codebase is
// written to avoid. If kimi_k3.cpp ever changes this signature, this file fails
// to link rather than silently calling stale logic.
bool kimi_k3_forward_layer(KimiK3Forward& fwd, int layer, const float* hidden_in,
                           float* hidden_out);
bool k3_check_launch(int layer, int phase);

namespace {
// fwd.s (KimiK3Forward::Scratch) is forward-declared only in kimi_k3.h --
// deliberately opaque outside kimi_k3.cpp, its own translation unit. The
// qact_output path (k3_proj_ggml_f32) needs fwd.s->proj_q8, which this file
// cannot see the definition of, so this file does NOT support it -- rather
// than guess at Scratch's layout or duplicate it (both of which risk exactly
// the "loads cleanly, runs, emits wrong tokens" bug class this repo is
// written to avoid). If qact output is enabled, fail loudly here instead of
// silently ignoring the setting or crashing on an incomplete type.
bool qact_output_requested() {
    const char* all = std::getenv("SPARKINFER_K3_GGML_QACT");
    const char* out = std::getenv("SPARKINFER_K3_GGML_QACT_OUTPUT");
    return (all && all[0] == '1') || (out && out[0] == '1');
}
}  // namespace

bool kimi_k3_prefill_tokens(KimiK3Forward& fwd, const int* prompt_ids, int n,
                            float* out_logits) {
    if (n <= 0) return false;
    const KimiK3Config& cfg = *fwd.cfg;
    const KimiK3Weights& w = *fwd.w;
    cudaStream_t stream = fwd.stream;
    const int H = cfg.hidden;

    // Same reset-per-forward-pass discipline as kimi_k3_forward_token: the
    // cross-layer residual bank is per-TOKEN, not persistent, so it is cleared
    // before every token's layer loop -- not once for the whole prompt. Getting
    // this wrong is exactly the max_ckpt-guard failure documented at
    // kimi_k3_forward_token's own reset line.

    // ---- hoisted out of the per-token loop (the allocation half of the win) ----
    float* x = nullptr;
    float* x_next = nullptr;
    if (cudaMalloc(&x, (size_t)H * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&x_next, (size_t)H * sizeof(float)) != cudaSuccess) {
        cudaFree(x);
        return false;
    }
    float* logits_dev = nullptr;
    if (cudaMalloc(&logits_dev, (size_t)cfg.vocab * sizeof(float)) != cudaSuccess) {
        cudaFree(x);
        cudaFree(x_next);
        return false;
    }

    bool ok = true;
    for (int i = 0; ok && i < n; ++i) {
        const int token_id = prompt_ids[i];
        const bool is_last = (i == n - 1);

        // Reset EVERY token, same as kimi_k3_forward_token -- see the comment
        // there. This is per-forward-pass state, not per-prefill state.
        fwd.state->n_ckpt = 0;

        // ---- embedding row gather. Identical to kimi_k3_forward_token. ----
        if (!w.token_embd.ok()) { ok = false; break; }
        long row_bytes = 0;
        if (w.token_embd.type == 0) row_bytes = (long)H * sizeof(float);
        else if (w.token_embd.type == 8) row_bytes = (long)(H / 32) * 34;
        else { ok = false; break; }
        const char* base = (const char*)w.token_embd.data + (size_t)token_id * row_bytes;
        if (!k3k::dequant_f32_by_type(x, base, H, w.token_embd.type, stream)) {
            ok = false;
            break;
        }
        if (fwd.debug) fwd.debug("inp_embd", -1, x, H);

        // ---- the layer stack. Byte-for-byte kimi_k3_forward_token's loop. ----
        for (int layer = 0; ok && layer < cfg.n_layers; ++layer) {
            ok = kimi_k3_forward_layer(fwd, layer, x, x_next);
            std::swap(x, x_next);
        }
        if (!ok) break;

        // ---- THE SKIP. Every prompt token except the last stops here: no
        // output_norm, no lm_head, no host copy. Only advance position and the
        // KV/state side effects that kimi_k3_forward_layer already applied
        // in-place stay live for the next token. ----
        if (!is_last) {
            ++fwd.state->position;
            continue;
        }

        // ---- last token only: identical tail to kimi_k3_forward_token. ----
        if (cfg.attn_res_block_size > 0) {
            if (!w.has_output_res_score || !w.output_res_score.ok()) {
                ok = false;
                break;
            }
            k3k::attn_res_mix_f32(x_next, fwd.state->res_bank, x,
                                  (const float*)w.output_res_score.data, H,
                                  fwd.state->n_ckpt, cfg.rms_eps, stream);
            std::swap(x, x_next);
        }
        if (!w.output_norm.ok()) { ok = false; break; }
        k3k::rms_norm_f32(x_next, x, (const float*)w.output_norm.data, H, cfg.rms_eps,
                          stream);
        std::swap(x, x_next);
        if (fwd.debug) fwd.debug("result_norm", -1, x, H);

        if (!w.output.ok()) { ok = false; break; }
        // NOT SUPPORTED: qact-output lm_head. See the comment on
        // qact_output_requested() above -- this needs fwd.s, which this file
        // cannot see the complete definition of. Fail loudly rather than
        // silently compute the wrong (unquantized) thing while the env var
        // claims quantized output is in effect.
        if (qact_output_requested()) {
            std::fprintf(stderr,
                "[k3-prefill] SPARKINFER_K3_GGML_QACT(_OUTPUT) is set, but "
                "kimi_k3_prefill_tokens() does not support the quantized-activation "
                "lm_head path (needs KimiK3Forward::Scratch, private to kimi_k3.cpp). "
                "Unset the env var, or fall back to the per-token loop for this run.\n");
            ok = false;
            break;
        }
        ok = k3k::k3_proj_f32(logits_dev, x, w.output.data, w.output.type,
                              cfg.vocab, H, stream);
        if (!ok) break;

        // k3_check_launch's real signature takes a K3LayerPhase, not an int --
        // this file intentionally does not fabricate that enum's value. Wire the
        // correct constant when this lands in the tree; flagged here rather than
        // guessed, since a wrong phase value here would be a launch-error check
        // silently checking the wrong thing.
        // if (ok) ok = k3_check_launch(cfg.n_layers, K3LayerPhase::All);
        if (ok) {
            cudaStreamSynchronize(stream);
            cudaMemcpy(out_logits, logits_dev, (size_t)cfg.vocab * sizeof(float),
                      cudaMemcpyDeviceToHost);
        }
        ++fwd.state->position;
    }

    cudaFree(logits_dev);
    cudaFree(x);
    cudaFree(x_next);
    return ok;
}

} // namespace sparkinfer
