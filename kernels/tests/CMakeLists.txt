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

# The batched prefill projection against the single-token path it replaces, on device.
# The CPU order test proves the two loop nests are the same float program; this proves the
# two IMPLEMENTATIONS agree on hardware -- dp4a, the real block_for tier, the launch
# geometry and the epilogue, none of which a host simulation reaches.
add_executable(k3_prefill_proj_gpu_test k3_prefill_proj_gpu_test.cu)
target_link_libraries(k3_prefill_proj_gpu_test PRIVATE si_k3)
target_include_directories(k3_prefill_proj_gpu_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME k3_prefill_proj_gpu_test COMMAND k3_prefill_proj_gpu_test)

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

# Chunk-parallel KDA prefill (k3_kda_chunk_prefill.cu). Grades the chunk form
# against the SAME token-by-token recurrence k3_kda_step_cpu_test already pins,
# in float64 — multi-chunk (the cross-chunk state carry a single chunk cannot
# exercise), a ragged tail, T=1, the full folded contract from raw inputs, and the
# axis-of-decay swap as a negative control, that being the exact bug class op 6's
# own header documents shipping once undetected. No device: the algebra is where
# the risk is, and this is the coverage a fork's PR actually runs.
add_executable(k3_kda_chunk_prefill_cpu_test k3_kda_chunk_prefill_cpu_test.cpp)
set_target_properties(k3_kda_chunk_prefill_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME k3_kda_chunk_prefill_cpu_test COMMAND k3_kda_chunk_prefill_cpu_test)

# IQ1_S -> int8 tensor cores (k3_moe_iq1s_mma.cu). The load-bearing claim is that
# IQ1_S IS an int8 format exactly — w = (dl/8)*(8g+s) — and that the 32-value
# sub-block coincides with mma.m16n8k32's k-extent. Both are integer and
# exactly-representable-float facts, so the CPU is the right place to prove them and a
# GPU would only obscure them. Includes the negative control the GPU tests use: the
# plausible simplification (drop the delta) must be measurably wrong.
add_executable(k3_moe_iq1s_mma_cpu_test k3_moe_iq1s_mma_cpu_test.cpp)
set_target_properties(k3_moe_iq1s_mma_cpu_test PROPERTIES CXX_STANDARD 17)
target_include_directories(k3_moe_iq1s_mma_cpu_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME k3_moe_iq1s_mma_cpu_test COMMAND k3_moe_iq1s_mma_cpu_test)

# Prefill batching must not perturb the float program: the batched projection's
# (row, token) cell has to equal what the single-token kernel produces for that token
# alone. Host-only and exact -- it rebuilds both loop nests rather than calling CUDA,
# which is what lets it pin block_for against the decode tier without a GPU.
add_executable(k3_prefill_order_cpu_test k3_prefill_order_cpu_test.cpp)
set_target_properties(k3_prefill_order_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME k3_prefill_order_cpu_test COMMAND k3_prefill_order_cpu_test)

add_executable(k3_mla_prefill_attention_cpu_test k3_mla_prefill_attention_cpu_test.cpp)
set_target_properties(k3_mla_prefill_attention_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME k3_mla_prefill_attention_cpu_test COMMAND k3_mla_prefill_attention_cpu_test)

add_executable(k3_mla_prefill_attention_gpu_test k3_mla_prefill_attention_gpu_test.cu)
target_link_libraries(k3_mla_prefill_attention_gpu_test PRIVATE si_k3)
target_include_directories(k3_mla_prefill_attention_gpu_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME k3_mla_prefill_attention_gpu_test COMMAND k3_mla_prefill_attention_gpu_test)

foreach(k3t k3_kda_step_gpu_test k3_mla_absorb_gpu_test
            k3_proj_1bar_gpu_test k3_moe_iq1s_mma_gpu_test
            k3_moe_batched_iq1s_gpu_test k3_kda_chunk_prefill_gpu_test
            k3_proj_q8_mma_gpu_test)
    add_executable(${k3t} ${k3t}.cu)
    target_link_libraries(${k3t} PRIVATE si_k3)
    target_include_directories(${k3t} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    add_test(NAME ${k3t} COMMAND ${k3t})
endforeach()
