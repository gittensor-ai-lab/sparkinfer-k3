# Kimi K3 specific kernels: situ, KDA decode step, KDA output gating, cross-layer
# attention residual mix, MLA output gate. See
# kernels/include/sparkinfer/kernels/kimi_k3.h for the reference semantics each one
# must match, and kernels/tests/kimi_k3_numeric_test.cu for the float64 check.
file(GLOB SRC "*.cu")
add_library(si_k3 STATIC ${SRC})
target_include_directories(si_k3 PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(si_k3 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-lineinfo;--ptxas-options=-v>)
set_target_properties(si_k3 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
