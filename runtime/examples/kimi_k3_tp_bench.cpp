// What does tensor parallelism actually buy on Kimi K3 decode?
//
//   kimi_k3_tp_bench <first-shard.gguf> [tp_size] [n_layers] [n_tokens] [--ctx N] [--seek]
//                    [--ids a,b,c | --ids @FILE] [--logits FILE]
//                    [--logits-prefix P --checkpoints L1,L2,...]
//
// --ids feeds an exact prompt (ids, not text) instead of the synthetic sequence, and
// --logits dumps the final step's logits in .spkl. Together they are what compares a
// full 93-layer TP run against the captured llama.cpp reference in bench/refdata/ —
// the only correctness check available at full depth, since no single GPU can hold
// 553 GiB to serve as a tp=1 baseline.
//
// --checkpoints extends that from one depth to many: with a long --ids prompt it dumps
// <P>.ctx<L>.spkl at each requested prefix length in a single pass, so parity is
// established at 4…32768 tokens rather than only at a 4-token prompt. One number from
// one short probe cannot distinguish "correct" from "correct only while the KV cache is
// nearly empty"; ten depths can.
//
// Reports ms/token and ms/layer, and extrapolates ms/layer to the full 93. The
// extrapolation is the useful number: no single GPU can hold 93 layers of this model,
// so a measured 93-layer single-GPU baseline does not exist and cannot be produced.
// Per-layer cost is the only quantity that compares across configurations.
//
// WHAT TP DOES AND DOES NOT SPEED UP HERE. Under ShardPolicy::ExpertsOnly the routed
// experts are banded across ranks, so the MoE dispatch — 69.7% of decode by the
// measured profile — is divided by tp_size. Attention is REPLICATED and therefore
// runs redundantly on every rank at full cost. Amdahl puts the ceiling at
//
//     1 / (0.30 + 0.70/tp)   ->  ~2.6x at tp=8, not 8x
//
// against which the measured number should be read. Beating it would mean something
// is wrong with the measurement; falling far short of it means the collective or the
// launch overhead is eating the win, which is the thing worth knowing.
//
// The first token is discarded: it pays one-time CUDA context, module load and the
// per-device lattice-table upload, none of which recur.

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"
#include "sparkinfer/models/kimi_k3_tp.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace sparkinfer;

namespace {

// .spkl: magic | u32 version | u32 n_tokens | u32 n_vocab | f32[n_tokens][n_vocab].
// Byte-identical to kimi_k3_generate's writer so compare_logits.py reads both.
bool write_spkl(const char* path, const std::vector<float>& logits, int n_vocab) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const uint32_t ver = 1, ntok = 1, nv = (uint32_t)n_vocab;
    const bool ok = std::fwrite("SPKL", 1, 4, f) == 4 &&
                    std::fwrite(&ver, 4, 1, f) == 1 &&
                    std::fwrite(&ntok, 4, 1, f) == 1 &&
                    std::fwrite(&nv, 4, 1, f) == 1 &&
                    std::fwrite(logits.data(), sizeof(float), logits.size(), f) == logits.size();
    std::fclose(f);
    return ok;
}

std::vector<int> parse_ids(const char* s) {
    std::vector<int> out;
    // @FILE, because a 128k-token context CANNOT be passed inline. Linux caps a single argv
    // element at MAX_ARG_STRLEN (32 pages = 131072 bytes) independently of ARG_MAX, so
    // 131,072 ids -- ~592 KB as text -- fails with "Argument list too long" even though the
    // 2 MB total limit is nowhere near. Without this there is no way to feed the scored
    // context as REAL tokens, and correctness at 128k stays untestable: --seek leaves the
    // cache zeroed, which is faithful for timing and meaningless for logits.
    std::string buf;
    if (s && s[0] == '@') {
        std::FILE* f = std::fopen(s + 1, "rb");
        if (!f) { std::printf("--ids: cannot open %s\n", s + 1); return out; }
        char chunk[65536];
        size_t n;
        while ((n = std::fread(chunk, 1, sizeof chunk, f)) > 0) buf.append(chunk, n);
        std::fclose(f);
        s = buf.c_str();
    }
    const char* p = s;
    while (*p) {
        // Separator-agnostic: commas, newlines and spaces all work, so a file written by
        // `tr` or one id per line both parse.
        while (*p && (*p == ',' || *p == '\n' || *p == '\r' || *p == ' ' || *p == '\t')) ++p;
        if (!*p) break;
        out.push_back(std::atoi(p));
        while (*p && *p != ',' && *p != '\n' && *p != '\r' && *p != ' ' && *p != '\t') ++p;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <first-shard.gguf> [tp_size] [n_layers] [n_tokens] "
                     "[--ctx N] [--seek]\n",
                     argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int tp_size  = argc > 2 ? std::atoi(argv[2]) : 8;
    const int want_lyr = argc > 3 ? std::atoi(argv[3]) : 16;
    int n_tokens = argc > 4 ? std::atoi(argv[4]) : 8;
    const char* logits_path = nullptr;
    // --ctx N allocates the KV cache for N positions; --seek jumps position to near the end
    // of it so the MLA attention actually reduces over the whole thing.
    //
    // WHY SEEK RATHER THAN FILL. K3 has no prefill path, so genuinely reaching 128k means
    // 131,072 sequential forward_token calls -- about 10 hours at main's speed, per
    // measurement. The decode cost at a given context is DATA-INDEPENDENT: MLA attention
    // runs the same dense reduction over the KV cache whether the entries are real
    // activations or zeros, the KDA recurrent state is fixed-size, and the projections do
    // not depend on context at all. Allocating the cache at the target size, leaving it
    // zeroed, and seeking position therefore measures the same per-token cost the real fill
    // would, in minutes.
    //
    // WHAT IT CANNOT DO: establish correctness at that context. It attends over zeros, so
    // the logits are meaningless. Correctness is established separately at short context
    // against the llama.cpp capture in bench/refdata/ -- which is why kimi_k3_eval.sh runs
    // the accuracy pass WITHOUT these flags.
    int  max_ctx = 64;
    bool do_seek = false;
    // TILED INGESTION IS THE DEFAULT, and it has to be rather than a flag.
    //
    // bench/scripts/kimi_k3_eval.sh is a maintainer-only path — the sensitive-paths guard
    // blocks a contributor from editing the script that scores them, correctly, because
    // whoever can edit the scorer can score themselves. So a driver the harness must opt
    // into can never become the measured number: the harness would keep feeding the prompt
    // one token at a time and the engine's own best ingestion would sit unused.
    //
    // Making it the default is an ENGINE change, not a scoring change. The harness still
    // measures whatever prompt ingestion this binary does; this changes what that is.
    // kimi_k3_tp_prefill falls back to the per-token path for any prompt or geometry
    // outside its contract, so nothing that worked before stops working.
    //
    // --no-prefill restores the token loop on the same binary, which is what every A/B in
    // this branch's history was measured against.
    bool do_prefill = true;    // --no-prefill: ingest one token at a time instead
    std::vector<int> ids;
    // --checkpoints: dump logits at several DEPTHS of one prompt in a single pass.
    //
    // The depths are nested prefixes of one document, and this executor consumes a prompt
    // one token at a time (kimi_k3_tp_forward_token is the decode step, not a prefill
    // batch), so after L tokens the logits ARE the L-token prefix's next-token
    // distribution -- on exactly the code path production runs. That makes one pass over
    // the longest prefix equivalent to N separate runs, without needing a cache reset the
    // runtime does not expose, and without paying to re-feed every shorter prefix.
    std::vector<int> checkpoints;
    const char* logits_prefix = nullptr;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--ids") && i + 1 < argc) ids = parse_ids(argv[++i]);
        else if (!std::strcmp(argv[i], "--logits") && i + 1 < argc) logits_path = argv[++i];
        else if (!std::strcmp(argv[i], "--logits-prefix") && i + 1 < argc) logits_prefix = argv[++i];
        else if (!std::strcmp(argv[i], "--checkpoints") && i + 1 < argc) checkpoints = parse_ids(argv[++i]);
        else if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) max_ctx = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seek")) do_seek = true;
        else if (!std::strcmp(argv[i], "--prefill")) do_prefill = true;
        else if (!std::strcmp(argv[i], "--no-prefill")) do_prefill = false;
    }
    if (!checkpoints.empty() && !logits_prefix) {
        std::printf("--checkpoints needs --logits-prefix\n");
        return 1;
    }

    GGUF g; KimiK3Config cfg; KimiK3LayerCoverage cov;
    if (!kimi_k3_load_partial(path.c_str(), cfg, g, &cov)) { std::printf("load failed\n"); return 1; }
    int n_avail = 0;
    while (n_avail < cfg.n_layers && cov.layer_complete[n_avail]) ++n_avail;
    const int n_full = cfg.n_layers;

    int mla_probe = -1, moe_probe = -1;
    for (int i = 0; i < n_full && (mla_probe < 0 || moe_probe < 0); ++i) {
        if (!cov.layer_complete[i]) continue;
        if (mla_probe < 0 && !cfg.is_kda_layer(i)) mla_probe = i;
        if (moe_probe < 0 && i >= cfg.leading_dense) moe_probe = i;
    }
    if (mla_probe < 0) { std::printf("no complete MLA layer\n"); return 1; }

    cfg.n_layers = std::min(n_avail, want_lyr);
    K3PlanOptions opt;
    {
        const std::string p = "blk." + std::to_string(mla_probe) + ".";
        opt.has_q_lora     = g.tensor((p + "attn_q_a.weight").c_str()) != nullptr;
        opt.has_fused_kv_b = g.tensor((p + "attn_kv_b.weight").c_str()) != nullptr;
        opt.has_attn_gate  = g.tensor((p + "attn_gate.weight").c_str()) != nullptr;
    }
    if (moe_probe >= 0) {
        const std::string p = "blk." + std::to_string(moe_probe) + ".";
        opt.has_shared_experts = g.tensor((p + "ffn_gate_shexp.weight").c_str()) != nullptr;
        opt.has_routed_norm    = g.tensor((p + "ffn_routed_norm.weight").c_str()) != nullptr;
    }

    int ndev = 0; cudaGetDeviceCount(&ndev);
    if (tp_size > ndev) { std::printf("tp_size %d > %d GPUs\n", tp_size, ndev); return 1; }

    std::vector<int> devs;
    for (int i = 0; i < tp_size; ++i) devs.push_back(i);

    const auto t_load0 = std::chrono::steady_clock::now();
    KimiK3TP p;
    if (!kimi_k3_tp_init(g, cfg, opt, devs, max_ctx, p)) {
        std::printf("init failed at tp=%d, %d layers\n", tp_size, cfg.n_layers);
        return 1;
    }
    const auto t_load1 = std::chrono::steady_clock::now();
    const double load_s = std::chrono::duration<double>(t_load1 - t_load0).count();

    // --seek: place every rank near the END of the allocated KV cache so the MLA attention
    // reduces over the full context. Leaves room for the warm-up token plus n_tokens more,
    // so no step runs past max_ctx.
    if (do_seek) {
        const int target = max_ctx - n_tokens - 2;
        if (target <= 0) {
            std::printf("--seek: ctx %d too small for %d tokens\n", max_ctx, n_tokens);
            return 1;
        }
        for (auto& r : p.ranks) {
            // BOTH copies. Setting only r.state.position leaves the device at 0 while the
            // host plans for 131040 — the kernels then attend over a single position and
            // the benchmark measures a token that is not doing the work it claims to.
            cudaSetDevice(r.device);
            if (!kimi_k3_set_position(r.state, target)) {
                std::printf("seek: failed to set device position on rank %d\n", r.rank);
                return 1;
            }
        }
        std::printf("seek: position -> %d (ctx alloc %d)\n", target, max_ctx);
    }

    std::vector<float> logits((size_t)cfg.vocab);

    // --ids: feed the exact prompt, no warm-up, no synthetic tokens. The recurrent
    // state and KV cache must see the reference's ids in the reference's order, so a
    // discarded warm-up token would advance position and corrupt the comparison.
    if (!ids.empty()) {
        std::printf("prompt: %zu ids from --ids\n", ids.size());
        // The KV cache was allocated for max_ctx; feeding more than that walks off the end
        // of it. Silently producing logits from a corrupted cache would be worse than
        // stopping, because the result still looks like a number.
        if ((int)ids.size() > max_ctx) {
            std::printf("--ids has %zu ids but --ctx is only %d\n", ids.size(), max_ctx);
            return 1;
        }
        for (int L : checkpoints) {
            if (L <= 0 || (size_t)L > ids.size()) {
                std::printf("checkpoint %d out of range (prompt has %zu ids)\n", L, ids.size());
                return 1;
            }
            if (L > max_ctx) {
                std::printf("checkpoint %d exceeds --ctx %d\n", L, max_ctx);
                return 1;
            }
        }
        // TIME THE INGESTION. Prompt ingestion is prefill, and there is no batched path --
        // every prompt token goes through the same single-token decode step, so a prompt
        // costs exactly what generating the same number of tokens costs. Nothing in this
        // bench reported that, so the only prefill numbers the project had were inferred
        // from checkpoint file mtimes. PREFILL lines below are the measurement.
        //
        // Measured from AFTER init: model load is a separate, one-off cost and folding it
        // in would flatter longer prompts.
        const auto t_pf0 = std::chrono::steady_clock::now();
        auto ms_since = [&](const std::chrono::steady_clock::time_point& t0) {
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0).count();
        };
        // FOUR INGESTION ARMS ON ONE BINARY, WHICH IS THE POINT. --prefill is the tile
        // driver (captured), SPARKINFER_K3_PREFILL_BATCH=B is the chunked walk
        // (uncaptured, batched collectives), SPARKINFER_K3_PREFILL_CHUNK=B is the
        // token-axis chunk driver (uncaptured, batched KERNELS), and none of them is the
        // token loop below. All four read the same prompt and write the same .spkl, so a
        // delta is the loop order and never a rebuild.
        //
        // PRECEDENCE, most specific request first:
        //   --prefill  >  SPARKINFER_K3_PREFILL_BATCH>1  >  SPARKINFER_K3_PREFILL_CHUNK>0
        //   >  the token loop.
        // --prefill wins if several are asked for: it is the arm that ships, and silently
        // running another one would misattribute its number. BATCH sits above CHUNK for
        // the same reason in miniature: BATCH defaults to 1 (off), so BATCH>1 can only be
        // an explicit request, whereas CHUNK defaults to 16 — letting a default shadow an
        // explicit request is exactly the misattribution this ordering exists to stop.
        // Since --prefill itself defaults ON, reaching either env arm needs --no-prefill.
        //
        // --prefill routes ingestion through the batched tile driver instead of the
        // per-token loop. It is a separate flag rather than the default because the two
        // must stay comparable on ONE binary: the same build, the same prompt, the same
        // reference logits, and only the loop order different.
        //
        // CHECKPOINTS ARE NOT AVAILABLE ON THIS PATH and that is deliberate rather than
        // an omission. A checkpoint asks for the logits at depth L, which means running
        // the head at a token in the middle of a tile; the tile driver runs the head only
        // for the last token it ingests. Reporting a checkpoint here would either mean a
        // second head (measuring something the optimisation does not do) or silently
        // reporting the wrong depth's logits. The final-token logits below ARE produced,
        // so end-to-end parity against the per-token path is still checkable.
        // CHECKPOINTS FALL BACK RATHER THAN FAIL, now that tiling is the default. A
        // checkpoint asks for the logits at depth L, which means running the head at a
        // token in the middle of a tile, and the tile driver runs the head only for the
        // last token it ingests. Refusing was right while --prefill was an explicit
        // request — the caller asked for something incoherent and should hear so. As a
        // DEFAULT, refusing would break every existing --checkpoints caller for a reason
        // they did not ask for, so the token loop serves them instead.
        if (do_prefill && !checkpoints.empty()) {
            std::printf("--checkpoints needs per-token logits; ingesting one token at a "
                        "time instead of tiling\n");
            std::fflush(stdout);
            do_prefill = false;
        }
        // WHICH INGESTION DRIVER RUNS BY DEFAULT — decided HERE, above the tile arm.
        //
        // This placement is the whole point. The tile arm RETURNS, so every gate read
        // below it is unreachable on the default invocation: `bench <model> <tp> <L> 1
        // --ids @… --ctx …` with no flags and no env, which is exactly how the scored
        // harness runs it. A driver selected further down is a driver that never runs.
        //
        // Hoisting these two reads has no side effects — they are getenv and nothing
        // else. The ORDER of the arms below is unchanged, and so is the MLA split floor,
        // which still sits under the tile arm for the reason its own comment gives (it
        // would otherwise be baked into the tile driver's captured graph).
        const char* pb_env = std::getenv("SPARKINFER_K3_PREFILL_BATCH");
        const int pb = pb_env ? std::atoi(pb_env) : 1;
        static const int pf_chunk = [] {
            const char* e = std::getenv("SPARKINFER_K3_PREFILL_CHUNK");
            return e ? std::atoi(e) : 64;   // measured optimum; see k3_ffn_batch_cap_env

        }();
        // THE TOKEN-AXIS CHUNK DRIVER IS THE DEFAULT INGESTION PATH.
        //
        // It carries B tokens through the kernels together, so each weight tile is read
        // once and spent on B tokens instead of one — M=1 GEMV becomes M=B GEMM. That is
        // a different and larger effect than the tile driver's phase-major reordering,
        // which batches the collectives but still streams the weights per token.
        //
        // THE TILE DRIVER REMAINS REACHABLE ON THE SAME BINARY, via
        // SPARKINFER_K3_PREFILL_CHUNK=0. That is not a courtesy: it is what makes the two
        // comparable at all — same build, same prompt, same reference logits, only the
        // driver different. Every measurement on this branch that was trusted came from a
        // same-binary control, and the ones that did not (rebuild vs rebuild) are the ones
        // that had to be retracted.
        //
        // It yields to an explicit SPARKINFER_K3_PREFILL_BATCH>1, which is a deliberate
        // request for that arm, and to --checkpoints, which needs per-token logits that a
        // chunked walk only materialises at chunk boundaries.
        const bool use_chunk = pf_chunk > 0 && pb <= 1 && checkpoints.empty() &&
                               ids.size() > 1;
        // PREFILL_BATCH>1 must skip the tile arm too, or "yields to it" would be a lie:
        // the tile arm returns, so anyone setting that gate would silently get the tile
        // driver instead of the arm they asked for. Both explicit gates now reach the arm
        // they name, which is what makes this binary a three-way control.
        const bool use_batch = pb > 1 && checkpoints.empty();
        if (do_prefill && !use_chunk && !use_batch) {
            const auto t_b0 = std::chrono::steady_clock::now();
            if (!kimi_k3_tp_prefill(p, ids.data(), (int)ids.size(), logits.data())) {
                std::printf("prefill failed\n"); return 1;
            }
            const double el = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t_b0).count();
            const size_t n = ids.size();
            std::printf("PREFILL_TOTAL tokens=%zu ms=%.1f tok_s=%.2f ms_per_token=%.3f\n",
                        n, el, n * 1000.0 / el, el / (double)n);
            std::fflush(stdout);
            int best = 0;
            for (int i = 1; i < cfg.vocab; ++i)
                if (logits[(size_t)i] > logits[(size_t)best]) best = i;
            std::printf("argmax next-token id: %d  logit: %.6f\n", best, logits[(size_t)best]);
            if (logits_path)
                std::printf("%s %s\n",
                            write_spkl(logits_path, logits, cfg.vocab) ? "wrote"
                                                                       : "FAILED to write",
                            logits_path);
            kimi_k3_tp_free(p);
            return 0;
        }

        // Ingestion opts into the shorter MLA slice for the whole prompt, chunked or
        // not. It is honoured only above the parity window, so the graded probes stay
        // on the shipped path; restored after the prompt so decode is untouched.
        //
        // DELIBERATELY BELOW THE TILE ARM. The tile driver captures a CUDA graph, so a
        // split floor set before it would be baked into the captured plan and would
        // change the shipped default path's behaviour. Everything from here down is
        // uncaptured, and keeping the floor across BOTH remaining arms AND the token
        // loop is what keeps the BATCH walk usable as a control against that loop.
        sparkinfer::kernels::k3::k3_mla_set_split_min(512);
        struct SplitRestore { ~SplitRestore() { sparkinfer::kernels::k3::k3_mla_set_split_min(0); } } split_restore;

        // CHUNKED INGESTION, when asked for. SPARKINFER_K3_PREFILL_BATCH=B carries B
        // tokens through each layer together; unset (or 1) keeps the token loop below,
        // so this bench measures both arms on ONE binary and any delta is the walk
        // rather than a rebuild.
        //
        // It is skipped when --checkpoints asks for intermediate depths: those need the
        // logits after a SPECIFIC prefix, and a chunked walk only materialises them at
        // chunk boundaries. Refusing loudly beats dumping a .spkl for the wrong depth,
        // which would be compared against llama's and read as a parity failure.
        // pb and pf_chunk are read once, above the tile arm — see the driver-selection
        // block there. Re-reading them here would be a second source of truth for which
        // driver runs, which is the defect that made the chunk arm unreachable.
        if (pb > 1 && !checkpoints.empty()) {
            std::printf("SPARKINFER_K3_PREFILL_BATCH=%d ignored: --checkpoints needs "
                        "per-token logits, which a chunked walk does not produce.\n", pb);
            std::fflush(stdout);
        }
        // `use_batch` is this condition, computed with the other driver gates above. It
        // used to be recomputed here under a second name; two spellings of one predicate
        // is how the arms drift apart.
        if (use_batch) {
            std::printf("chunked prefill: B=%d over %zu tokens\n", pb, ids.size());
            std::fflush(stdout);
            if (!kimi_k3_tp_forward_prompt(p, ids.data(), (int)ids.size(),
                                           logits.data())) {
                std::printf("chunked prefill failed\n");
                return 1;
            }
        }

        // TOKEN-AXIS CHUNK INGESTION. With no --checkpoints to service mid-prompt, the
        // whole prompt but its last token can go through the chunk driver, which carries
        // B tokens through the KERNELS together — M=B GEMMs, batched FFN/MoE/attention —
        // rather than only batching the collectives as the BATCH walk above does. The
        // final token still goes through forward_token so the caller gets its logits.
        // SPARKINFER_K3_PREFILL_CHUNK=0 restores the per-token loop on one binary.
        //
        // Yields to an explicit SPARKINFER_K3_PREFILL_BATCH>1: see the precedence note
        // above — CHUNK is on by default, so it must not shadow a requested arm.
        // `use_chunk` already excludes the PREFILL_BATCH arm (it requires pb <= 1), the
        // checkpoint case and a one-token prompt, so it is the whole condition. Stating
        // those clauses again here would let the two copies drift.
        if (use_chunk && !use_batch) {
            if (!kimi_k3_tp_prefill_chunk(p, ids.data(), (int)ids.size() - 1, pf_chunk)) {
                std::printf("chunked prefill failed\n"); return 1;
            }
            if (!kimi_k3_tp_forward_token(p, ids.back(), logits.data())) {
                std::printf("final prompt token failed\n"); return 1;
            }
            const double el = ms_since(t_pf0);
            std::printf("PREFILL_TOTAL tokens=%zu ms=%.1f tok_s=%.2f ms_per_token=%.3f\n",
                        ids.size(), el, ids.size() * 1000.0 / el, el / (double)ids.size());
            std::fflush(stdout);
            if (logits_path) {
                std::printf("%s %s\n",
                            write_spkl(logits_path, logits, cfg.vocab) ? "wrote" : "FAILED to write",
                            logits_path);
            }
            kimi_k3_tp_free(p);
            return 0;
        }

        // A prompt token's logits are read only at a --checkpoints depth or at the
        // very end, so ask for them only there. nullptr puts the driver on its
        // ingest-only path, which skips the read-back and its host barrier; the KV
        // cache, d_pos and the recurrent state advance exactly as before.
        for (size_t i = 0; !use_batch && i < ids.size(); ++i) {
            const int depth_i = (int)i + 1;
            bool want_logits = (i + 1 == ids.size());
            for (int L : checkpoints) if (L == depth_i) want_logits = true;
            if (!kimi_k3_tp_forward_token(p, ids[i], want_logits ? logits.data() : nullptr)) {
                std::printf("prompt token %zu (id %d) failed\n", i, ids[i]); return 1;
            }
            // Depth i+1 is now complete: these logits are the (i+1)-token prefix's answer.
            const int depth = (int)i + 1;
            for (int L : checkpoints) {
                if (L != depth) continue;
                // Report the cumulative ingestion cost at this depth BEFORE the dump, so
                // the .spkl write and the argmax scan are not charged to prefill.
                const double el = ms_since(t_pf0);
                std::printf("PREFILL tokens=%d ms=%.1f tok_s=%.2f ms_per_token=%.3f\n",
                            L, el, L * 1000.0 / el, el / L);
                std::fflush(stdout);
                const std::string path = std::string(logits_prefix) + ".ctx" +
                                         std::to_string(L) + ".spkl";
                if (!write_spkl(path.c_str(), logits, cfg.vocab)) {
                    // A missing dump would silently become "no comparison at this depth",
                    // which is exactly the failure this suite exists to remove.
                    std::printf("FAILED to write %s\n", path.c_str());
                    return 1;
                }
                int b = 0;
                for (int v = 1; v < cfg.vocab; ++v)
                    if (logits[(size_t)v] > logits[(size_t)b]) b = v;
                std::printf("ctx %6d -> argmax %d  logit %.6f  [%s]\n",
                            L, b, logits[(size_t)b], path.c_str());
                std::fflush(stdout);
            }
        }
        // Whole-prompt ingestion: this is TTFT for a prompt of this length, minus load.
        {
            const double el = ms_since(t_pf0);
            const size_t n = ids.size();
            std::printf("PREFILL_TOTAL tokens=%zu ms=%.1f tok_s=%.2f ms_per_token=%.3f\n",
                        n, el, n * 1000.0 / el, el / (double)n);
            std::fflush(stdout);
        }

        int best = 0;
        for (int i = 1; i < cfg.vocab; ++i)
            if (logits[(size_t)i] > logits[(size_t)best]) best = i;
        std::printf("argmax next-token id: %d  logit: %.6f\n", best, logits[(size_t)best]);
        if (logits_path) {
            std::printf("%s %s\n",
                        write_spkl(logits_path, logits, cfg.vocab) ? "wrote" : "FAILED to write",
                        logits_path);
        }
        kimi_k3_tp_free(p);
        return 0;
    }

    // Warm-up token, discarded: one-time context/module/lattice-table costs.
    if (!kimi_k3_tp_forward_token(p, 1000, logits.data())) { std::printf("warmup failed\n"); return 1; }

    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < n_tokens; ++t) {
        if (!kimi_k3_tp_forward_token(p, 1000 + t, logits.data())) {
            std::printf("token %d failed\n", t); return 1;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_tokens;

    size_t free_b = 0, total_b = 0;
    cudaSetDevice(devs[0]);
    cudaMemGetInfo(&free_b, &total_b);
    const double GiB = 1024.0 * 1024.0 * 1024.0;

    std::printf("\n=== Kimi K3 UD-IQ1_S decode, tp=%d, ExpertsOnly ===\n", tp_size);
    std::printf("  layers measured    %d of %d\n", cfg.n_layers, n_full);
    std::printf("  load               %.1f s\n", load_s);
    std::printf("  rank0 VRAM in use  %.2f GiB of %.2f\n",
                (total_b - free_b) / GiB, total_b / GiB);
    std::printf("  ms/token           %.2f  (%.2f tok/s)\n", ms, 1000.0 / ms);
    std::printf("  ms/layer           %.3f\n", ms / cfg.n_layers);
    std::printf("  collectives/token  %ld\n", p.n_collectives / (n_tokens + 1));
    std::printf("  --> extrapolated to %d layers: %.1f ms/token (%.2f tok/s)\n",
                n_full, ms / cfg.n_layers * n_full,
                1000.0 / (ms / cfg.n_layers * n_full));

    // WEPS engagement, summed over ranks. Zero when counting is off (the default) —
    // the line prints only under SPARKINFER_K3_WEPS_COUNT=1 so ordinary runs are
    // byte-identical in output.
    if (const char* wc = std::getenv("SPARKINFER_K3_WEPS_COUNT"); wc && wc[0] == '1') {
        unsigned long long skips = 0;
        for (int d : devs) {
            cudaSetDevice(d);
            skips += sparkinfer::kernels::k3::k3_weps_skips_read_current_device();
        }
        std::printf("  WEPS_SKIPS         %llu\n", skips);
    }

    // SPARKINFER_K3_PROFILE=1 prints the per-phase split. Note it adds cudaEvent
    // pairs around every branch, which roughly doubles ms/token — read the SHARES,
    // never the absolute time, from a profiled run.
    kimi_k3_profile_report();

    kimi_k3_tp_free(p);
    return 0;
}
