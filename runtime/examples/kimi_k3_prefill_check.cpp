// Correctness check for kimi_k3_prefill_tokens(): runs the SAME prompt through
// (a) the existing naive per-token loop (kimi_k3_forward_token, called N times)
// and (b) kimi_k3_prefill_tokens(), and diffs the final logits.
//
// This is the harness this repo expects for every new code path (see
// kimi_k3_layer0_ref_check.cpp, kimi_k3_moe_ref_check.cpp, kimi_k3_mla_ref_check.cpp
// for the pattern) -- a feature without one of these is a feature nobody can trust
// yet. Written and reasoned against the two call sequences it compares, but NOT
// YET RUN: no GPU was available at write time. This is the first thing to run once
// one is -- if it fails, that is the prefill path's bug to fix before anything else
// in this PR matters.
//
//   kimi_k3_prefill_check <first-shard.gguf> <id> [id ...]
//     --devices 0,1        layer-split across these GPUs (default: 0)
//     --max-layers N       cap depth, same convention as kimi_k3_generate
//     --tol F              max abs logit diff to accept (default 1e-4 -- the two
//                          paths run bit-identical kernel calls in the same order,
//                          so this should be at or near float rounding noise, NOT
//                          a loose "close enough" bar. A larger diff means the two
//                          call sequences are NOT equivalent and the prefill path
//                          has a real bug, not a numerics quirk.)

#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"
#include "sparkinfer/models/kimi_k3_prefill.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace sparkinfer;

namespace {
std::vector<int> parse_devices(const std::string& s) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        out.push_back(std::atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    if (out.empty()) out.push_back(0);
    return out;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <first-shard.gguf> <id> [id ...] "
            "[--devices 0,1] [--max-layers N] [--tol F]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    std::vector<int> devices{0};
    int max_layers = -1;
    float tol = 1e-4f;
    std::vector<int> prompt;

    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--devices") && i + 1 < argc) devices = parse_devices(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-layers") && i + 1 < argc) max_layers = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--tol") && i + 1 < argc) tol = (float)std::atof(argv[++i]);
        else prompt.push_back(std::atoi(argv[i]));
    }
    if (prompt.size() < 2) {
        std::fprintf(stderr, "need at least 2 prompt ids to exercise a multi-token prefill\n");
        return 2;
    }

    GGUF g; KimiK3Config cfg; KimiK3LayerCoverage cov;
    if (!kimi_k3_load_partial(path, cfg, g, &cov)) { std::fprintf(stderr, "load failed\n"); return 1; }
    int n_avail = 0;
    while (n_avail < cfg.n_layers && cov.layer_complete[n_avail]) ++n_avail;
    if (max_layers > 0 && max_layers < n_avail) n_avail = max_layers;
    if (n_avail < 1) { std::fprintf(stderr, "no complete layers\n"); return 1; }
    cfg.n_layers = n_avail;

    // ---- run A: the naive per-token loop, exactly as kimi_k3_generate.cpp does ----
    KimiK3Weights wA;
    if (!kimi_k3_load_weights(g, cfg, K3PlanOptions{}, wA, 0, cfg.n_layers - 1)) {
        std::fprintf(stderr, "run A: weight load failed\n"); return 1;
    }
    KimiK3RuntimeState stA;
    if (!kimi_k3_alloc_state(cfg, (int)prompt.size() + 8, stA)) {
        std::fprintf(stderr, "run A: alloc_state failed\n"); return 1;
    }
    KimiK3Forward fwdA;
    fwdA.cfg = &cfg; fwdA.w = &wA; fwdA.state = &stA;
    if (!kimi_k3_forward_alloc_scratch(cfg, fwdA)) {
        std::fprintf(stderr, "run A: forward_alloc_scratch failed\n"); return 1;
    }
    std::vector<float> logitsA(cfg.vocab), logitsB(cfg.vocab);
    bool okA = true;
    for (size_t i = 0; okA && i < prompt.size(); ++i)
        okA = kimi_k3_forward_token(fwdA, prompt[i], logitsA.data());
    if (!okA) { std::fprintf(stderr, "run A: forward_token failed\n"); return 1; }
    kimi_k3_forward_free_scratch(fwdA);

    // ---- run B: the new batched-prefill driver, fresh state, same prompt ----
    KimiK3Weights wB;
    if (!kimi_k3_load_weights(g, cfg, K3PlanOptions{}, wB, 0, cfg.n_layers - 1)) {
        std::fprintf(stderr, "run B: weight load failed\n"); return 1;
    }
    KimiK3RuntimeState stB;
    if (!kimi_k3_alloc_state(cfg, (int)prompt.size() + 8, stB)) {
        std::fprintf(stderr, "run B: alloc_state failed\n"); return 1;
    }
    KimiK3Forward fwdB;
    fwdB.cfg = &cfg; fwdB.w = &wB; fwdB.state = &stB;
    if (!kimi_k3_forward_alloc_scratch(cfg, fwdB)) {
        std::fprintf(stderr, "run B: forward_alloc_scratch failed\n"); return 1;
    }
    const bool okB = kimi_k3_prefill_tokens(fwdB, prompt.data(), (int)prompt.size(),
                                            logitsB.data());
    if (!okB) { std::fprintf(stderr, "run B: kimi_k3_prefill_tokens failed\n"); return 1; }
    kimi_k3_forward_free_scratch(fwdB);

    // ---- diff ----
    float max_abs_diff = 0.0f;
    int max_diff_idx = -1;
    for (int i = 0; i < cfg.vocab; ++i) {
        const float d = std::fabs(logitsA[(size_t)i] - logitsB[(size_t)i]);
        if (d > max_abs_diff) { max_abs_diff = d; max_diff_idx = i; }
    }
    std::printf("max_abs_logit_diff=%.8f at vocab id %d (tol=%.8f)\n",
               max_abs_diff, max_diff_idx, tol);
    if (max_abs_diff > tol) {
        std::fprintf(stderr,
            "FAIL: prefill path diverges from the naive per-token loop -- treat this "
            "as a real bug in kimi_k3_prefill_tokens, not a numerics tolerance issue, "
            "since both paths call the same kernels in the same order.\n");
        return 1;
    }
    std::printf("PASS: prefill path matches the naive per-token loop within tolerance.\n");
    return 0;
}
