# CPU composition tests — pure C++, validate the wiring math (no GPU needed).

# Tensor-parallel shard math + all-reduce backend selection. Both are CUDA-free by
# construction so they run in ordinary CI, which matters more here than usual: a
# wrong row/column split, a GQA kv group split across ranks, or a silent fall back
# to a different collective all produce a model that loads, runs at full speed,
# and is subtly wrong — the one failure mode a rented 8-GPU box will not surface.
add_executable(tp_shard_cpu_test tp_shard_cpu_test.cpp ../src/tp/shard.cpp)
set_target_properties(tp_shard_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME tp_shard_cpu_test COMMAND tp_shard_cpu_test)

add_executable(tp_backend_cpu_test tp_backend_cpu_test.cpp ../src/tp/backend_select.cpp)
set_target_properties(tp_backend_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME tp_backend_cpu_test COMMAND tp_backend_cpu_test)

# Tensor -> shard-rule mapping. Worst-consequence layer in the TP stack: a wrong
# rule is not a crash and not a shape mismatch — every rank computes a well-formed
# activation over the wrong slice, the all-reduce sums eight wrong partials into a
# plausible answer, and the model stays fluent. Asserted against the real GGUF
# tensor names the loader reads.
add_executable(tp_weight_plan_cpu_test tp_weight_plan_cpu_test.cpp
               ../src/tp/weight_plan.cpp ../src/tp/shard.cpp)
set_target_properties(tp_weight_plan_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME tp_weight_plan_cpu_test COMMAND tp_weight_plan_cpu_test)

add_executable(decode_layer_cpu_test decode_layer_cpu_test.cpp)
set_target_properties(decode_layer_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME decode_layer_cpu_test COMMAND decode_layer_cpu_test)

add_executable(qwen35_cpu_test qwen35_cpu_test.cpp)
set_target_properties(qwen35_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME qwen35_cpu_test COMMAND qwen35_cpu_test)

add_executable(qwen36_config_cpu_test qwen36_config_cpu_test.cpp ../src/gguf.cpp)
set_target_properties(qwen36_config_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME qwen36_config_cpu_test COMMAND qwen36_config_cpu_test)

# Kimi K3 GGUF metadata parse — same shape as qwen36_config_cpu_test, but also
# exercises GGUF integer-array capture (per-layer head_count_kv) and the
# required-key refusal that stops silent situ/attn_res/latent defaults.
add_executable(kimi_k3_config_cpu_test kimi_k3_config_cpu_test.cpp ../src/gguf.cpp)
set_target_properties(kimi_k3_config_cpu_test PROPERTIES CXX_STANDARD 17)
target_include_directories(kimi_k3_config_cpu_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME kimi_k3_config_cpu_test COMMAND kimi_k3_config_cpu_test)

# Multi-shard GGUF merge + K3 tensor-manifest validation (no GPU / no real weights).
add_executable(kimi_k3_manifest_cpu_test kimi_k3_manifest_cpu_test.cpp ../src/gguf.cpp)
set_target_properties(kimi_k3_manifest_cpu_test PROPERTIES CXX_STANDARD 17)
target_include_directories(kimi_k3_manifest_cpu_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME kimi_k3_manifest_cpu_test COMMAND kimi_k3_manifest_cpu_test)

# Thermal governor — pure tiering policy (no GPU); links the runtime for classify()/pace() symbols.
add_executable(thermal_governor_cpu_test thermal_governor_cpu_test.cpp)
target_link_libraries(thermal_governor_cpu_test PRIVATE sparkinfer_runtime)
set_target_properties(thermal_governor_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME thermal_governor_cpu_test COMMAND thermal_governor_cpu_test)

add_executable(scheduler_cpu_test scheduler_cpu_test.cpp)
target_link_libraries(scheduler_cpu_test PRIVATE sparkinfer_runtime)
set_target_properties(scheduler_cpu_test PROPERTIES CXX_STANDARD 17)
add_test(NAME scheduler_cpu_test COMMAND scheduler_cpu_test)

add_executable(batch_engine_gpu_test batch_engine_gpu_test.cpp)
target_link_libraries(batch_engine_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME batch_engine_gpu_test COMMAND batch_engine_gpu_test)

# GPU integration tests — run the real device path; skip without a device.
add_executable(decode_runner_gpu_test decode_runner_gpu_test.cpp)
target_link_libraries(decode_runner_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME decode_runner_gpu_test COMMAND decode_runner_gpu_test)

add_executable(qwen35_gpu_test qwen35_gpu_test.cpp)
target_link_libraries(qwen35_gpu_test PRIVATE sparkinfer_runtime CUDA::cudart)
add_test(NAME qwen35_gpu_test COMMAND qwen35_gpu_test)

# GGUF bytes -> per-rank device buffers. Links gguf.cpp for the public block table:
# the residency planner MUST size quantised tensors with the same table the reader
# uses, since a second copy would be free to drift.
#
# Worth its own test because of one asymmetry in GGUF layout — quant blocks tile ne0,
# and ne0 is the CONTRACTED dim, so a column shard cuts inside every block row. A cut
# that lands mid-block yields half a block: no scale, misaligned codepoints, noise on
# dequant, and nothing downstream can tell. K3's leading-dense ffn_down is exactly
# that case at tp=8 (33792/8 = 16.5 blocks of 256), so this is not a theoretical guard.
add_executable(tp_weight_residency_cpu_test tp_weight_residency_cpu_test.cpp
               ../src/tp/weight_residency.cpp ../src/tp/weight_plan.cpp
               ../src/tp/shard.cpp ../src/gguf.cpp)
set_target_properties(tp_weight_residency_cpu_test PROPERTIES CXX_STANDARD 17)
target_include_directories(tp_weight_residency_cpu_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME tp_weight_residency_cpu_test COMMAND tp_weight_residency_cpu_test)

# The K3 decode schedule: op order, widths, and collective placement as DATA, so the
# properties that would otherwise only show up as degraded output are assertable
# without a GPU. Strongest check inside is the plan/manifest cross-check — two tensor
# sets written from different sources (kernel signatures vs the reference loader) that
# must name exactly the same tensors.
add_executable(kimi_k3_decode_plan_cpu_test kimi_k3_decode_plan_cpu_test.cpp
               ../src/models/kimi_k3_decode_plan.cpp
               ../src/tp/weight_plan.cpp ../src/tp/shard.cpp ../src/gguf.cpp)
set_target_properties(kimi_k3_decode_plan_cpu_test PROPERTIES CXX_STANDARD 17)
target_include_directories(kimi_k3_decode_plan_cpu_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME kimi_k3_decode_plan_cpu_test COMMAND kimi_k3_decode_plan_cpu_test)
