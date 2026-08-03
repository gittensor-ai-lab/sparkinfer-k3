# CPU reference correctness tests — pure C++, no CUDA needed. They validate the
# kernel algorithms against an independent double-precision ground truth.
add_executable(cpu_reference_test cpu_reference_test.cpp)
set_target_properties(cpu_reference_test PROPERTIES CXX_STANDARD 17)
add_test(NAME cpu_reference_test COMMAND cpu_reference_test)

# Kimi K3 kernel coherence test — GPU. Same principle as the CPU reference tests
# above (independent double-precision ground truth), but the ops it covers only exist
# on the device, so it needs a GPU. It skips cleanly with exit 0 when none is present,
# so ctest stays green on a CPU-only box.
#
# Each case additionally asserts that a WRONG-BUT-PLAUSIBLE variant of the same
# formula is measurably different — scalar decay instead of KDA's per-channel decay,
# sigmoid on both situ branches, weighting normalised instead of raw residuals. Without
# that, a passing tolerance would only prove the tolerance was generous.
add_executable(kimi_k3_numeric_test kimi_k3_numeric_test.cu)
target_link_libraries(kimi_k3_numeric_test PRIVATE si_k3)
target_include_directories(kimi_k3_numeric_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME kimi_k3_numeric_test COMMAND kimi_k3_numeric_test)

# k3_proj_f32: the generic f32-activation GEMV every non-expert K3 weight goes
# through (attn_q/k/v/output, ffn_gate/up/down, ffn_routed_down/up, token_embd,
# output.weight — everything the real file shows as Q8_0 or F32). Separate from the
# bf16 GEMV path the Qwen models use, since K3's kernels are f32 end to end.
add_executable(kimi_k3_proj_gemv_test kimi_k3_proj_gemv_test.cu)
target_link_libraries(kimi_k3_proj_gemv_test PRIVATE si_k3)
target_include_directories(kimi_k3_proj_gemv_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME kimi_k3_proj_gemv_test COMMAND kimi_k3_proj_gemv_test)

# IQ1_S device path, including llama.cpp-compatible Q8_K activation conversion.
add_executable(kimi_k3_iq1s_gpu_test kimi_k3_iq1s_gpu_test.cu)
target_link_libraries(kimi_k3_iq1s_gpu_test PRIVATE si_k3)
target_include_directories(kimi_k3_iq1s_gpu_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME kimi_k3_iq1s_gpu_test COMMAND kimi_k3_iq1s_gpu_test)

# Row dequant (F32 passthrough + Q8_0), extending dequant_f32_by_type: what
# token_embd.weight / output.weight lookups need (dequantise one contiguous row,
# not a reduction). Bit-exact vs a float64 reference for both types.
add_executable(kimi_k3_row_dequant_test kimi_k3_row_dequant_test.cu)
target_link_libraries(kimi_k3_row_dequant_test PRIVATE si_k3)
target_include_directories(kimi_k3_row_dequant_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME kimi_k3_row_dequant_test COMMAND kimi_k3_row_dequant_test)

# Plain full-width RMS norm — attn_norm, ffn_norm, q_a_norm, kv_a_norm,
# ffn_routed_norm all use this; every other norm in kimi_k3.h is a fused variant.
add_executable(kimi_k3_rms_norm_test kimi_k3_rms_norm_test.cu)
target_link_libraries(kimi_k3_rms_norm_test PRIVATE si_k3)
target_include_directories(kimi_k3_rms_norm_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME kimi_k3_rms_norm_test COMMAND kimi_k3_rms_norm_test)

# K3 decode fast paths — ONE TEST BINARY PER OPTIMISATION, mirroring one source file
# per optimisation (kernels/csrc/cuda/kimi_k3/k3_*.cu). A factor that has to be dropped
# after a re-measurement takes its own test with it and leaves the others building.
#
# The _cpu_ ones check the SCHEDULES against float64 with no device, which is the only
# coverage that runs on a fork's PR: the five ci.yml checks stay pending until a
# maintainer approves the workflow. The _gpu_ ones run the same paths at K3's REAL dims
# — the shape every fast path is gated on, and the one kimi_k3_numeric_test's shrunk
# dims (kv_lora 64, key_length 80) never reach. They exit 0 with no device.
add_executable(k3_kda_step_cpu_test k3_kda_step_cpu_test.cpp)
set_target_properties(k3_kda_step_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME k3_kda_step_cpu_test COMMAND k3_kda_step_cpu_test)

foreach(k3t k3_kda_step_gpu_test k3_mla_absorb_gpu_test
            k3_proj_1bar_gpu_test)
    add_executable(${k3t} ${k3t}.cu)
    target_link_libraries(${k3t} PRIVATE si_k3)
    target_include_directories(${k3t} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    add_test(NAME ${k3t} COMMAND ${k3t})
endforeach()
