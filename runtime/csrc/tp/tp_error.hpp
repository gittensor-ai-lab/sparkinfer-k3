// CUDA error handling. Header-only; throws on any non-Success cudaError_t.
//
// Wrap every CUDA runtime call that can fail with SPARKINFER_TP_CUDA_CHECK(expr).
// Don't use this inside __global__ kernels — only host code. Inside a
// destructor that can't propagate, swallow the error or terminate; do
// NOT let CudaError escape a dtor.
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace sparkinfer::tp {

class CudaError : public std::runtime_error {
public:
    CudaError(const char* expr, cudaError_t code)
        : std::runtime_error(std::string("CUDA: ") + expr + ": "
                             + cudaGetErrorString(code)),
          code_(code) {}
    cudaError_t code() const noexcept { return code_; }

private:
    cudaError_t code_;
};

}  // namespace sparkinfer::tp

#define SPARKINFER_TP_CUDA_CHECK(expr)                                                \
    do {                                                                        \
        cudaError_t _sp_tp_cuda_err = (expr);                                 \
        if (_sp_tp_cuda_err != cudaSuccess) {                                 \
            throw ::sparkinfer::tp::CudaError(#expr, _sp_tp_cuda_err);         \
        }                                                                       \
    } while (0)
