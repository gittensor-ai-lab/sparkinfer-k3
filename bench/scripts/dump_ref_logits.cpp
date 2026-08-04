// Capture llama.cpp's reference logits + tokenization for a K3 model, as the ground
// truth sparkinfer's implementation is validated against. Emits, per context depth:
//   <out>.ctx<L>.spkl  — the next-token logits after the first L prompt tokens
//                        (bench compare_logits.py format:
//                         SPKL | u32 ver | u32 n_tokens | u32 n_vocab | f32[][])
//   <out>.ctx<L>.ids   — those L token ids, one per line, so OUR executor is fed the
//                        IDENTICAL tokenization (the acceptance test compares two runs
//                        on the same ids, so they must come from one tokenizer —
//                        llama.cpp's).
//   <out>.depths.txt   — a human-readable summary: each depth, its argmax and top logit.
//                        NOT <out>.txt, which is typically the corpus this reads from.
//
// With no --prefixes the old single-shot behaviour is kept exactly: <out>.spkl/.ids/.txt
// for the whole prompt, plus <out>.step<k>.spkl for [n_predict] greedy continuations.
//
// WHY MANY DEPTHS. A single short probe cannot certify parity at the context the engine
// is actually scored on. Prefixes of ONE document at 4…32768 tokens exercise the KV
// cache, the attention paths and the routing at the depths that matter, and each depth
// is an independent comparison rather than a single number. See bench/refdata/README.md.
//
// WHY A FRESH CONTEXT PER DEPTH, rather than one pass with logits at many positions:
// causal masking means position L-1 of a long pass is mathematically the same
// distribution, but it is not the same CODE PATH — batch splitting, cache growth and
// graph selection all differ. A reference is only useful if it was produced the way the
// thing under test will be run, so each depth gets its own context. The model is loaded
// once; only the (cheap) context is rebuilt, so this costs prefill time, not load time.
//
// Built against the K3-supporting unslothai/llama.cpp branch (kimi-k3-fullsize-vision).
// Defaults to CPU + mmap so it fits a single box (the weights page from disk on demand);
// pass --ngl 999 to offload to GPU, which is what makes the deep prefixes tractable —
// a 32k CPU prefill re-reads the whole model per ubatch and is effectively unbounded.
//
// Usage: dump_ref_logits <model.gguf> <out-prefix> <"prompt"|@file> [n_predict]
//                        [--ngl N] [--ubatch N] [--prefixes L1,L2,...]

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

static bool write_ids(const std::string& path, const llama_token* toks, int n) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    for (int i = 0; i < n; ++i) std::fprintf(f, "%d\n", (int)toks[i]);
    std::fclose(f);
    return true;
}

// "@path" reads the prompt from a file: a 32k-token prompt is ~130 KB of text, which is
// at Linux's MAX_ARG_STRLEN (32 pages) and cannot be passed as one argv element.
static bool load_prompt(const std::string& spec, std::string& out) {
    if (spec.empty() || spec[0] != '@') { out = spec; return true; }
    FILE* f = std::fopen(spec.c_str() + 1, "rb");
    if (!f) { std::fprintf(stderr, "cannot open prompt file %s\n", spec.c_str() + 1); return false; }
    char buf[65536];
    size_t n;
    out.clear();
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    std::fclose(f);
    return true;
}

static std::vector<int> parse_csv_ints(const char* s) {
    std::vector<int> out;
    while (s && *s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        out.push_back(std::atoi(s));
        while (*s && *s != ',' && *s != ' ') ++s;
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <out-prefix> <\"prompt\"|@file> [n_predict]\n"
            "          [--ngl N] [--ubatch N] [--prefixes L1,L2,...]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string out_prefix = argv[2];
    std::string prompt;
    if (!load_prompt(argv[3], prompt)) return 1;

    int n_predict = 0;
    int n_gpu_layers = 0;      // CPU + mmap by default: page the weights from disk
    int n_ubatch = 0;          // 0 = leave llama.cpp's default
    std::vector<int> prefixes;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--ngl") && i + 1 < argc) n_gpu_layers = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--ubatch") && i + 1 < argc) n_ubatch = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--prefixes") && i + 1 < argc) prefixes = parse_csv_ints(argv[++i]);
        else if (argv[i][0] != '-') n_predict = std::atoi(argv[i]);
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
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
    std::printf("prompt %zu chars -> %d tokens, vocab %d, ngl %d\n",
                prompt.size(), n, n_vocab, n_gpu_layers);

    // No --prefixes: the whole prompt is the one depth, and the greedy continuation
    // still runs. This is the historical contract and hello.* was captured with it.
    const bool multi = !prefixes.empty();
    if (!multi) prefixes.push_back(n);

    for (int L : prefixes) {
        if (L <= 0 || L > n) {
            std::fprintf(stderr, "prefix %d out of range (prompt has %d tokens) — "
                                 "refusing to emit a reference that is not the depth "
                                 "it claims\n", L, n);
            return 1;
        }
    }

    // One line per depth, so the capture is auditable at a glance.
    //
    // In multi-depth mode this is <out>.depths.txt, NOT <out>.txt: the natural invocation
    // is `dump_ref_logits … refdata/longctx @refdata/longctx.txt`, where <out>.txt IS the
    // corpus. Writing the summary there silently truncates the prompt file to a few hundred
    // bytes -- the run in progress survives (the corpus is read into memory before any
    // output), so the damage only surfaces the NEXT time someone regenerates, against a
    // corpus that is now the previous run's summary.
    const std::string summary_path = out_prefix + (multi ? ".depths.txt" : ".txt");
    if (summary_path == std::string(argv[3]).substr(argv[3][0] == '@' ? 1 : 0)) {
        std::fprintf(stderr, "refusing to write the summary over the prompt file %s\n",
                     summary_path.c_str());
        return 1;
    }
    FILE* summary = std::fopen(summary_path.c_str(), "w");
    if (summary) {
        std::fprintf(summary, "model: %s\ntokens available: %d  vocab: %d  ngl: %d\n",
                     model_path.c_str(), n, n_vocab, n_gpu_layers);
        if (!multi) std::fprintf(summary, "prompt: %s\n", prompt.c_str());
    }

    for (size_t pi = 0; pi < prefixes.size(); ++pi) {
        const int L = prefixes[pi];
        const std::string tag = multi ? (".ctx" + std::to_string(L)) : std::string();

        // A fresh context per depth: same reason the engine under test gets a fresh run
        // per depth. Rebuilding the context does NOT reload the model.
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t)(L + n_predict + 16);
        cp.n_batch = (uint32_t)(L + 16);
        if (n_ubatch > 0) cp.n_ubatch = (uint32_t)n_ubatch;
        llama_context* ctx = llama_init_from_model(model, cp);
        if (!ctx) { std::fprintf(stderr, "failed to create context at L=%d\n", L); return 1; }

        // Decode the prefix in one batch; logits at the last position are the
        // next-token distribution — the thing to compare against.
        llama_batch batch = llama_batch_get_one(toks.data(), L);
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "decode failed at L=%d\n", L);
            return 1;
        }

        const float* logits = llama_get_logits_ith(ctx, L - 1);
        if (!logits) { std::fprintf(stderr, "no logits at L=%d\n", L); return 1; }

        if (!write_spkl(out_prefix + tag + ".spkl", logits, n_vocab)) {
            std::fprintf(stderr, "failed to write %s%s.spkl\n", out_prefix.c_str(), tag.c_str());
            return 1;
        }
        write_ids(out_prefix + tag + ".ids", toks.data(), L);

        int amax = 0;
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[amax]) amax = i;
        char piece[256];
        const int pl = llama_token_to_piece(vocab, amax, piece, sizeof(piece), 0, true);
        const std::string tok_txt(piece, pl > 0 ? pl : 0);
        if (summary) {
            std::fprintf(summary, "ctx %6d  argmax %6d  logit %10.6f  piece '%s'\n",
                         L, amax, logits[amax], tok_txt.c_str());
            std::fflush(summary);
        }
        std::printf("ctx %6d -> argmax %d (logit %.4f) '%s'  [%s%s.spkl]\n",
                    L, amax, logits[amax], tok_txt.c_str(), out_prefix.c_str(), tag.c_str());
        std::fflush(stdout);

        // Greedy continuation, single-depth mode only: append to the ids file and write
        // per-step logits as <prefix>.step<k>.spkl, so a multi-STEP comparison is
        // possible as well as a multi-DEPTH one.
        if (!multi) {
            llama_token cur = amax;
            for (int k = 0; k < n_predict; ++k) {
                llama_batch b1 = llama_batch_get_one(&cur, 1);
                if (llama_decode(ctx, b1) != 0) break;
                const float* lg = llama_get_logits_ith(ctx, 0);
                if (!lg) break;
                write_spkl(out_prefix + ".step" + std::to_string(k) + ".spkl", lg, n_vocab);
                int a = 0; for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[a]) a = i;
                FILE* f = std::fopen((out_prefix + ".ids").c_str(), "a");
                if (f) { std::fprintf(f, "%d\n", (int)cur); std::fclose(f); }
                cur = a;
            }
        }

        llama_free(ctx);
    }

    if (summary) std::fclose(summary);
    std::printf("wrote %zu reference depth(s) under %s (summary: %s)\n",
                prefixes.size(), out_prefix.c_str(), summary_path.c_str());

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
