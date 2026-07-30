// Capture llama.cpp's reference logits + tokenization for a K3 model, as the ground
// truth sparkinfer's implementation is validated against. Emits:
//   <out>.spkl   — the last prompt token's next-token logits (bench compare_logits.py
//                  format: SPKL | u32 ver | u32 n_tokens | u32 n_vocab | f32[][])
//   <out>.ids    — the prompt token ids (one per line), so OUR executor is fed the
//                  IDENTICAL tokenization (the acceptance test compares two runs on
//                  the same ids, so they must come from one tokenizer — llama.cpp's).
//   <out>.txt    — the prompt text and a human-readable head of the top logits.
//
// Built against the K3-supporting unslothai/llama.cpp branch (kimi-k3-fullsize-vision,
// efc8bc38 — the same commit sparkinfer's kernels transcribe). Runs on CPU with mmap
// so it fits a single box: the 594 GB weights page from disk on demand rather than
// needing 594 GB of RAM.
//
// Usage: dump_ref_logits <model.gguf> <out-prefix> "<prompt text>" [n_predict]

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool write_spkl(const std::string& path, const float* logits, int n_vocab) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t ver = 1, ntok = 1, nv = (uint32_t)n_vocab;
    const bool ok = std::fwrite("SPKL", 1, 4, f) == 4 &&
                    std::fwrite(&ver, 4, 1, f) == 1 &&
                    std::fwrite(&ntok, 4, 1, f) == 1 &&
                    std::fwrite(&nv, 4, 1, f) == 1 &&
                    std::fwrite(logits, sizeof(float), (size_t)n_vocab, f) == (size_t)n_vocab;
    std::fclose(f);
    return ok;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <out-prefix> \"<prompt>\" [n_predict]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string out_prefix = argv[2];
    const std::string prompt = argv[3];
    const int n_predict = argc > 4 ? std::atoi(argv[4]) : 0;

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;         // CPU + mmap: page the 594 GB from disk, no VRAM need
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Tokenize the prompt with the model's own tokenizer (add BOS/special as the
    // chat/base template would). These are the ids OUR executor must be fed.
    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                           toks.data(), (int)toks.size(),
                           /*add_special=*/true, /*parse_special=*/true);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                           toks.data(), (int)toks.size(), true, true); }
    if (n <= 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);
    std::printf("prompt %zu chars -> %d tokens, vocab %d\n", prompt.size(), n, n_vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)(n + n_predict + 16);
    cp.n_batch = (uint32_t)(n + 16);
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "failed to create context\n"); return 1; }

    // Decode the whole prompt in one batch; logits at the last position are the
    // next-token distribution — the thing to compare against.
    llama_batch batch = llama_batch_get_one(toks.data(), n);
    if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }

    const float* logits = llama_get_logits_ith(ctx, n - 1);
    if (!logits) { std::fprintf(stderr, "no logits\n"); return 1; }

    if (!write_spkl(out_prefix + ".spkl", logits, n_vocab)) {
        std::fprintf(stderr, "failed to write .spkl\n"); return 1;
    }

    // ids file
    {
        FILE* f = std::fopen((out_prefix + ".ids").c_str(), "w");
        for (int i = 0; i < n; ++i) std::fprintf(f, "%d\n", (int)toks[i]);
        std::fclose(f);
    }

    // argmax + a small human-readable summary
    int amax = 0;
    for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[amax]) amax = i;
    char piece[256];
    const int pl = llama_token_to_piece(vocab, amax, piece, sizeof(piece), 0, true);
    std::string tok_txt(piece, pl > 0 ? pl : 0);
    {
        FILE* f = std::fopen((out_prefix + ".txt").c_str(), "w");
        std::fprintf(f, "model: %s\nprompt: %s\ntokens: %d  vocab: %d\n",
                     model_path.c_str(), prompt.c_str(), n, n_vocab);
        std::fprintf(f, "argmax next-token id: %d  logit: %.6f  piece: '%s'\n",
                     amax, logits[amax], tok_txt.c_str());
        std::fclose(f);
    }
    std::printf("argmax next token: %d (logit %.4f) '%s'\n", amax, logits[amax], tok_txt.c_str());
    std::printf("wrote %s.spkl / .ids / .txt\n", out_prefix.c_str());

    // Optionally continue greedy for a few tokens, dumping each step's logits too,
    // so a multi-token comparison is possible later. Kept simple: append to the ids
    // file and write per-step .spkl as <prefix>.step<k>.spkl.
    llama_token cur = amax;
    int pos = n;
    for (int k = 0; k < n_predict; ++k) {
        llama_batch b1 = llama_batch_get_one(&cur, 1);
        if (llama_decode(ctx, b1) != 0) break;
        const float* lg = llama_get_logits_ith(ctx, 0);
        if (!lg) break;
        write_spkl(out_prefix + ".step" + std::to_string(k) + ".spkl", lg, n_vocab);
        int a = 0; for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[a]) a = i;
        FILE* f = std::fopen((out_prefix + ".ids").c_str(), "a");
        std::fprintf(f, "%d\n", (int)cur); std::fclose(f);
        cur = a; ++pos;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
