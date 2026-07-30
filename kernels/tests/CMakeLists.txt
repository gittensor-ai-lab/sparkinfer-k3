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
