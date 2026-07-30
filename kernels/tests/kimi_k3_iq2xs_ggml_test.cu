// IQ2_XS dequant: our CUDA kernel vs GGML'S OWN dequantize_row_iq2_xs,
// on REAL Kimi K3 expert weights.
//
// This is the strongest validation available for this kernel and it is worth
// saying why. For the other ten K3 kernels the reference was a float64 CPU
// implementation I wrote from the spec — independent of my CUDA, but not
// independent of my READING of the spec. IQ2_XS has a 512-entry lattice table
// where a single wrong constant corrupts only the codepoints that use it: the
// model still loads, still runs, and degrades invisibly. No spec-reading check
// catches that.
//
// So instead we link ggml's own compiled dequantiser and feed both it and our
// kernel the SAME BYTES OUT OF THE ACTUAL MODEL FILE. Agreement then means our
// kernel reproduces the implementation that produced the weights, on the data
// that will really be loaded — tables, scale nibbles, sign handling and all.

#include "sparkinfer/kernels/kimi_k3.h"
#include "sparkinfer/gguf.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" void dequantize_row_iq2_xs(const void* x, float* y, int64_t k);

using namespace sparkinfer;

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1]
        : "/workspace/models_k3/UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf";

    GGUF g;
    if (!g.open(path)) { std::printf("open failed\n"); return 1; }

    // Find a real IQ2_XS (ggml type 17) tensor — an expert block.
    const GGUFTensor* t = nullptr;
    std::string tname;
    for (const std::string& n : g.tensor_names()) {
        const GGUFTensor* c = g.tensor(n);
        if (c && c->ggml_type == 17 && c->data) { t = c; tname = n; break; }
    }
    if (!t) { std::printf("no IQ2_XS tensor found\n"); return 1; }
    std::printf("tensor : %s\n", tname.c_str());
    std::printf("type   : %d (IQ2_XS)   values %ld   bytes %ld   shard %d\n",
                t->ggml_type, t->n_values, t->n_bytes, t->shard);

    // Dequantise a chunk: enough blocks to hit many distinct grid codepoints.
    const int64_t QK_K = 256;
    const int64_t n_blocks = 4096;                 // 1M values
    const int64_t n = n_blocks * QK_K;
    if (t->n_values < n) { std::printf("tensor too small\n"); return 1; }
    const size_t block_bytes = 74;
    const size_t src_bytes = (size_t)n_blocks * block_bytes;

    std::printf("checking %lld values (%lld blocks) of REAL weight data\n\n",
                (long long)n, (long long)n_blocks);

    // --- ggml's own implementation, on the host ---
    std::vector<float> ref((size_t)n);
    dequantize_row_iq2_xs(t->data, ref.data(), n);

    // --- our CUDA kernel, same bytes ---
    void* d_src = nullptr; float* d_out = nullptr;
    cudaMalloc(&d_src, src_bytes);
    cudaMalloc(&d_out, (size_t)n * sizeof(float));
    cudaMemcpy(d_src, t->data, src_bytes, cudaMemcpyHostToDevice);
    sparkinfer::kernels::k3::dequant_iq2_xs_f32(d_out, d_src, n, 0);
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("CUDA: %s\n", cudaGetErrorString(e)); return 1; }

    std::vector<float> got((size_t)n);
    cudaMemcpy(got.data(), d_out, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);

    // --- compare. These must be BIT-IDENTICAL: same table, same arithmetic order,
    // no transcendentals involved. Anything else means a real divergence. ---
    int64_t mismatch = 0, nonzero = 0;
    double maxabs = 0.0;
    int64_t first_bad = -1;
    for (int64_t i = 0; i < n; ++i) {
        if (ref[i] != 0.0f) ++nonzero;
        if (got[i] != ref[i]) {
            if (first_bad < 0) first_bad = i;
            ++mismatch;
            maxabs = std::fmax(maxabs, std::fabs((double)got[i] - ref[i]));
        }
    }
    std::printf("non-zero reference values : %lld / %lld\n", (long long)nonzero, (long long)n);
    std::printf("bit-exact mismatches      : %lld\n", (long long)mismatch);
    if (mismatch) {
        std::printf("max |diff|                : %.9g\n", maxabs);
        for (int64_t i = first_bad; i < first_bad + 8 && i < n; ++i)
            std::printf("   [%lld] got %.9g  ggml %.9g\n", (long long)i, got[i], ref[i]);
    }
    // Sanity: a table or scale error often still produces *a* number, so also
    // confirm the output is not degenerate.
    double sum2 = 0.0;
    for (int64_t i = 0; i < n; ++i) sum2 += (double)ref[i] * ref[i];
    std::printf("reference RMS             : %.6g\n", std::sqrt(sum2 / n));

    cudaFree(d_src); cudaFree(d_out);
    const bool ok = (mismatch == 0) && (nonzero > n / 4);
    std::printf("\n%s\n", ok ? "PASS: bit-identical to ggml on real K3 expert weights"
                             : "FAIL");
    return ok ? 0 : 1;
}
