// Validate the SHARDED weight loader against the real Kimi K3 GGUF.
//
//   kimi_k3_tp_load_check <first-shard.gguf> [tp_size]
//
// tp_weight_residency_cpu_test already checks the band arithmetic on synthetic
// shapes. This checks it on the shapes and quant types the actual file has, and then
// checks that the loader's COPY reproduces the planned bytes on a device. Those are
// different failures: the planner can be right about a [4096, 4096] Q4_K tensor and
// still be wrong about K3's [3072, 3584, 896] IQ1_S expert stack, and the copy can be
// wrong even when the plan is right (cudaMemcpy2D's pitch arguments are easy to
// transpose, and a transposed pitch still copies the right NUMBER of bytes).
//
// THE INVARIANT BEING CHECKED IS TILING. Every byte of a sharded tensor must be
// claimed by EXACTLY ONE rank. A gap means an expert's weights are on no GPU and its
// contribution silently vanishes from the sum; an overlap means two ranks hold the
// same rows and the all-reduce counts them twice. Both leave a model that loads, runs
// and emits fluent text. Neither is visible without this check.
//
// Runs against a PARTIAL download: tensors the shards do not cover are skipped and
// counted, so this is useful long before all 14 shards land.

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/kimi_k3.h"
#include "sparkinfer/models/kimi_k3_config.h"
#include "sparkinfer/models/kimi_k3_gguf_manifest.h"
#include "sparkinfer/tp/shard.h"
#include "sparkinfer/tp/weight_residency.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace sparkinfer;

namespace {

int g_fail = 0;
int g_pass = 0;

void ok(bool cond, const std::string& what) {
    if (cond) { ++g_pass; return; }
    ++g_fail;
    std::printf("  FAIL  %s\n", what.c_str());
}

// One rank's claim on the source tensor, as a set of [begin, end) byte intervals.
// Contiguous rules give one interval; ColShard gives one per row, but every row has
// the SAME offset pattern, so the row-0 pattern is what has to tile.
struct Claim {
    std::size_t off = 0;    // offset within the row (ColShard) or the tensor
    std::size_t len = 0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <first-shard.gguf> [tp_size]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int tp_size = argc > 2 ? std::atoi(argv[2]) : 8;

    GGUF g;
    KimiK3Config cfg;
    KimiK3LayerCoverage cov;
    if (!kimi_k3_load_partial(path.c_str(), cfg, g, &cov)) {
        std::printf("load failed\n");
        return 1;
    }
    std::printf("model: %d layers, hidden %d, %d experts, latent %d, tp_size %d\n",
                cfg.n_layers, cfg.hidden, cfg.n_experts, cfg.expert_latent, tp_size);

    // Shard dims, per rank. Everything the rules consult must be set: an unset
    // n_q_heads_total makes the attention rules band a zero-length axis, which
    // reports EmptyShard rather than the real answer.
    auto dims_for = [&](int rank) {
        tp::ShardDims d;
        d.tp_size = tp_size;
        d.rank = rank;
        d.hidden = cfg.hidden;
        d.n_q_heads_total = cfg.n_q_heads;
        d.head_dim = cfg.kda_head_dim;
        d.n_q_heads = cfg.n_q_heads / tp_size;
        d.q_rows = d.n_q_heads * d.head_dim;
        d.n_kv_heads_total = 1;             // K3 stores MLA as MQA
        d.n_kv_heads = 1;
        d.kv_replicated = true;             // 1 kv head cannot split across 8 ranks
        d.n_experts_total = cfg.n_experts;
        d.experts_sharded = (cfg.n_experts % tp_size) == 0;
        d.n_experts = d.experts_sharded ? cfg.n_experts / tp_size : cfg.n_experts;
        d.expert_band = d.experts_sharded ? tp::even_band(cfg.n_experts, tp_size, rank)
                                          : tp::Band{0, cfg.n_experts};
        d.moe_ffn_total = cfg.moe_ffn;
        d.moe_ffn = cfg.moe_ffn;
        return d;
    };

    // ---------------------------------------------------------------- part A
    // Every tensor in the file, every rank: does the plan tile the source exactly?
    std::printf("\n[A] tiling — every byte claimed by exactly one rank\n");
    long n_checked = 0, n_replicated = 0, n_sharded = 0, n_skipped = 0;
    std::size_t bytes_model = 0, bytes_rank0 = 0;

    for (const auto& name : g.tensor_names()) {
        const GGUFTensor* t = g.tensor(name);
        if (!t || !t->data) { ++n_skipped; continue; }

        std::vector<tp::TensorResidency> per_rank(tp_size);
        bool all_ok = true;
        for (int r = 0; r < tp_size; ++r) {
            per_rank[r] = tp::plan_tensor_residency(name, t->dims, t->n_dims,
                                                    t->ggml_type, dims_for(r));
            if (!per_rank[r].ok()) {
                ok(false, name + ": rank " + std::to_string(r) + " -> " +
                          tp::residency_error_name(per_rank[r].error) + " " + per_rank[r].note);
                all_ok = false;
            }
        }
        if (!all_ok) continue;
        ++n_checked;
        bytes_model += per_rank[0].full_bytes;
        bytes_rank0 += per_rank[0].rank_bytes;

        if (per_rank[0].is_replicated()) {
            ++n_replicated;
            // Replication is only correct if every rank really takes the WHOLE tensor.
            for (int r = 0; r < tp_size; ++r)
                ok(per_rank[r].rank_bytes == per_rank[r].full_bytes,
                   name + ": replicated rank " + std::to_string(r) + " holds the whole tensor");
            continue;
        }
        ++n_sharded;

        // Collect each rank's claim and check it tiles.
        std::vector<Claim> claims(tp_size);
        const bool strided = !per_rank[0].copy.contiguous();
        for (int r = 0; r < tp_size; ++r) {
            const tp::StridedCopy& c = per_rank[r].copy;
            if (strided) {
                // ColShard: same sub-range in every row. Offset within the row.
                claims[r].off = c.src_offset % (c.src_stride ? c.src_stride : 1);
                claims[r].len = c.row_bytes;
            } else {
                claims[r].off = c.src_offset;
                claims[r].len = c.total_bytes();
            }
        }
        std::sort(claims.begin(), claims.end(),
                  [](const Claim& a, const Claim& b) { return a.off < b.off; });

        const std::size_t span = strided
            ? per_rank[0].copy.src_stride            // one row
            : per_rank[0].full_bytes;                // whole tensor

        std::size_t cursor = 0;
        bool tiles = true;
        for (int r = 0; r < tp_size; ++r) {
            if (claims[r].off != cursor) { tiles = false; break; }
            cursor += claims[r].len;
        }
        if (cursor != span) tiles = false;
        ok(tiles, name + ": " + tp::rule_name(per_rank[0].rule) + " bands tile " +
                  std::to_string(span) + " bytes exactly across " +
                  std::to_string(tp_size) + " ranks");
    }

    const double GiB = 1024.0 * 1024.0 * 1024.0;
    std::printf("  %ld tensors checked (%ld sharded, %ld replicated, %ld skipped)\n",
                n_checked, n_sharded, n_replicated, n_skipped);
    std::printf("  present-shard bytes: model %.2f GiB, rank0 %.2f GiB (%.2fx over %d ranks)\n",
                bytes_model / GiB, bytes_rank0 / GiB,
                bytes_model ? (double)bytes_rank0 * tp_size / (double)bytes_model : 0.0, tp_size);

    // ---------------------------------------------------------------- part B
    // The COPY, on a device, against real bytes. Picks the first complete layer and
    // one tensor per rule so both the contiguous and the strided (cudaMemcpy2D) path
    // are exercised on real quantised data.
    std::printf("\n[B] copy — device bytes match the planned source slice\n");
    int probe_layer = -1;
    for (int i = 0; i < cfg.n_layers; ++i)
        if (cov.layer_complete[i]) { probe_layer = i; break; }
    if (probe_layer < 0) {
        std::printf("  SKIPPED: no complete layer in the downloaded shards yet\n");
    } else {
        const std::string p = "blk." + std::to_string(probe_layer) + ".";
        const char* probes[] = {
            "attn_output.weight",     // ColShard  -> strided, cudaMemcpy2D
            "ffn_gate_exps.weight",   // ExpertShard -> contiguous band of ne2
            "attn_norm.weight",       // Replicate -> whole tensor
        };
        for (const char* suffix : probes) {
            const std::string name = p + suffix;
            const GGUFTensor* t = g.tensor(name);
            if (!t || !t->data) { std::printf("  (absent: %s)\n", name.c_str()); continue; }

            for (int r = 0; r < tp_size; ++r) {
                const tp::TensorResidency res =
                    tp::plan_tensor_residency(name, t->dims, t->n_dims, t->ggml_type, dims_for(r));
                if (!res.ok()) { ok(false, name + " rank plan"); continue; }

                void* dev = nullptr;
                if (cudaMalloc(&dev, res.rank_bytes) != cudaSuccess) {
                    ok(false, name + ": cudaMalloc " + std::to_string(res.rank_bytes));
                    continue;
                }
                const char* src = static_cast<const char*>(t->data);
                const tp::StridedCopy& c = res.copy;
                cudaError_t e;
                if (c.contiguous()) {
                    e = cudaMemcpy((char*)dev + c.dst_offset, src + c.src_offset,
                                   c.total_bytes(), cudaMemcpyHostToDevice);
                } else {
                    e = cudaMemcpy2D((char*)dev + c.dst_offset, c.dst_stride,
                                     src + c.src_offset, c.src_stride,
                                     c.row_bytes, (size_t)c.n_rows, cudaMemcpyHostToDevice);
                }
                if (e != cudaSuccess) {
                    ok(false, name + ": upload " + cudaGetErrorString(e));
                    cudaFree(dev);
                    continue;
                }

                // Read back and compare against the source, row by row, independently
                // of the StridedCopy that produced it.
                std::vector<char> got(res.rank_bytes);
                if (cudaMemcpy(got.data(), dev, res.rank_bytes, cudaMemcpyDeviceToHost)
                        != cudaSuccess) {
                    ok(false, name + ": readback");
                    cudaFree(dev);
                    continue;
                }
                cudaFree(dev);

                bool same = true;
                const long rows = c.n_rows > 0 ? c.n_rows : 1;
                for (long j = 0; j < rows && same; ++j) {
                    const char* want = src + c.src_offset + (std::size_t)j * c.src_stride;
                    const char* have = got.data() + c.dst_offset + (std::size_t)j * c.dst_stride;
                    if (std::memcmp(want, have, c.row_bytes) != 0) same = false;
                }
                ok(same, name + " rank " + std::to_string(r) + ": " +
                         std::to_string(res.rank_bytes) + " bytes match source (" +
                         (c.contiguous() ? "contiguous" : "strided") + ")");
            }
        }
    }

    std::printf("\n%s: %d checks, %d failures\n",
                g_fail ? "FAIL" : "PASS", g_pass + g_fail, g_fail);
    return g_fail ? 1 : 0;
}
