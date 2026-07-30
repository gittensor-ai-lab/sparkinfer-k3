// Kimi K3 generation driver: pre-tokenized prompt ids in, greedy-decoded text out.
//
// Takes ids rather than a string on purpose — see kimi_k3_vocab.h on why encoding
// is out of scope (it needs a full tiktoken BPE pass that llama.cpp already has as
// `llama-tokenize`, and the acceptance test must feed BOTH implementations the same
// ids anyway). Same convention as runtime/examples/qwen3_gguf_generate.cpp.
//
//   kimi_k3_generate <first-shard.gguf> <n_predict> <id> [id ...]
//     --devices 0,1,2,3    layer-split across these GPUs (default: 0)
//     --max-layers N       cap depth (default: all available; a 96 GiB card holds
//                          about 13 MoE layers, so a single-GPU run needs this)
//     --logits FILE        dump the FINAL step's logits in the .spkl format
//                          bench/scripts/compare_logits.py reads, for the
//                          numerical comparison against llama.cpp
//
// Prompt ingestion is a plain decode loop (one forward per prompt token, keeping
// the sampled id only for the last one). That is not how a fast implementation
// would prefill — a batched prefill exists for Qwen in qwen35_prefill.cpp — but it
// is the same arithmetic, and correctness against llama.cpp is the open question
// here, not prompt throughput.

#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"
#include "sparkinfer/models/kimi_k3_vocab.h"

#include <chrono>
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

// .spkl: magic | u32 version | u32 n_tokens | u32 n_vocab | f32[n_tokens][n_vocab].
// Format defined by bench/scripts/compare_logits.py.
bool write_spkl(const char* path, const std::vector<float>& logits, int n_vocab) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const uint32_t ver = 1, ntok = 1, nv = (uint32_t)n_vocab;
    bool ok = std::fwrite("SPKL", 1, 4, f) == 4 &&
              std::fwrite(&ver, 4, 1, f) == 1 &&
              std::fwrite(&ntok, 4, 1, f) == 1 &&
              std::fwrite(&nv, 4, 1, f) == 1 &&
              std::fwrite(logits.data(), sizeof(float), logits.size(), f) == logits.size();
    std::fclose(f);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <first-shard.gguf> <n_predict> <id> [id ...] "
            "[--devices 0,1] [--max-layers N] [--logits FILE]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    const int n_predict = std::atoi(argv[2]);
    std::vector<int> devices{0};
    int max_layers = -1;
    const char* logits_path = nullptr;
    std::vector<int> prompt;

    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--devices") && i + 1 < argc) devices = parse_devices(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-layers") && i + 1 < argc) max_layers = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--logits") && i + 1 < argc) logits_path = argv[++i];
        else prompt.push_back(std::atoi(argv[i]));
    }
    if (prompt.empty()) { std::fprintf(stderr, "no prompt ids given\n"); return 2; }

    GGUF g; KimiK3Config cfg; KimiK3LayerCoverage cov;
    if (!kimi_k3_load_partial(path, cfg, g, &cov)) { std::fprintf(stderr, "load failed\n"); return 1; }

    int n_avail = 0;
    while (n_avail < cfg.n_layers && cov.layer_complete[n_avail]) ++n_avail;
    const int n_full = cfg.n_layers;
    if (max_layers > 0 && max_layers < n_avail) n_avail = max_layers;
    if (n_avail < 1) { std::fprintf(stderr, "no complete layers\n"); return 1; }
    cfg.n_layers = n_avail;

    KimiK3Vocab vocab;
    const bool have_vocab = kimi_k3_vocab_from_gguf(g, vocab);

    std::printf("model: %d/%d layers, vocab %d, %zu stage(s)%s\n",
                cfg.n_layers, n_full, cfg.vocab, devices.size(),
                cfg.n_layers < n_full ? "  [TRUNCATED DEPTH — output is not the real "
                                        "model's, only the pipeline's]" : "");
    if (!have_vocab)
        std::printf("note: no vocab in this GGUF — ids only, no detokenized text\n");

    K3PlanOptions opt;
    {
        const int probe = cfg.n_layers > 3 ? 3 : 0;
        const std::string p = "blk." + std::to_string(probe) + ".";
        opt.has_q_lora         = g.tensor((p + "attn_q_a.weight").c_str()) != nullptr;
        opt.has_attn_gate      = g.tensor((p + "attn_gate.weight").c_str()) != nullptr;
        opt.has_fused_kv_b     = false;
        opt.has_shared_experts = g.tensor((p + "ffn_gate_shexp.weight").c_str()) != nullptr;
        opt.has_routed_norm    = g.tensor((p + "ffn_routed_norm.weight").c_str()) != nullptr;
    }

    const int max_ctx = (int)prompt.size() + n_predict + 8;
    KimiK3Pipeline pipe;
    if (!kimi_k3_pipeline_init(g, cfg, opt, devices, max_ctx, pipe)) {
        std::fprintf(stderr, "pipeline init failed\n"); return 1;
    }

    std::vector<float> logits(cfg.vocab);
    std::vector<int> generated;

    // ---- prompt ingestion: one forward per token, keep only the last sample ----
    for (size_t i = 0; i < prompt.size(); ++i) {
        if (!kimi_k3_pipeline_forward_token(pipe, prompt[i], logits.data())) {
            std::fprintf(stderr, "forward failed on prompt token %zu\n", i);
            kimi_k3_pipeline_free(pipe); return 1;
        }
    }

    // ---- greedy decode (timed) ----
    using clk = std::chrono::steady_clock;
    auto t_dec0 = clk::now();
    int decoded = 0;
    for (int step = 0; step < n_predict; ++step) {
        int best = 0;
        for (int i = 1; i < cfg.vocab; ++i) if (logits[i] > logits[best]) best = i;
        generated.push_back(best);
        if (vocab.ok() && best == vocab.eos_id) {
            std::printf("[eos at step %d]\n", step);
            break;
        }
        if (step + 1 >= n_predict) break;
        if (!kimi_k3_pipeline_forward_token(pipe, best, logits.data())) {
            std::fprintf(stderr, "forward failed at step %d\n", step);
            kimi_k3_pipeline_free(pipe); return 1;
        }
        ++decoded;
    }
    const double dec_s = std::chrono::duration<double>(clk::now() - t_dec0).count();
    if (decoded > 0)
        std::printf("decode: %d forward passes in %.3f s = %.2f tok/s "
                    "(%.1f ms/token over %d layers => ~%.0f ms/token extrapolated to %d)\n",
                    decoded, dec_s, decoded / dec_s, 1000.0 * dec_s / decoded, cfg.n_layers,
                    1000.0 * dec_s / decoded * (double)n_full / cfg.n_layers, n_full);

    if (logits_path) {
        if (!write_spkl(logits_path, logits, cfg.vocab))
            std::fprintf(stderr, "failed to write %s\n", logits_path);
        else
            std::printf("wrote final-step logits to %s (.spkl, for compare_logits.py)\n",
                        logits_path);
    }

    std::printf("\nprompt ids:");
    for (int id : prompt) std::printf(" %d", id);
    std::printf("\ngenerated ids:");
    for (int id : generated) std::printf(" %d", id);
    std::printf("\n");
    if (have_vocab)
        std::printf("\ngenerated text: %s\n",
                    kimi_k3_detokenize(vocab, generated).c_str());

    kimi_k3_profile_report();
    kimi_k3_pipeline_free(pipe);
    return 0;
}
