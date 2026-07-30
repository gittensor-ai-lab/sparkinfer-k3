cmake_minimum_required(VERSION 3.20)
project(sparkinfer_runtime LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# Primary target sm_120 (RTX 5090). Also Ada/Hopper/datacenter-Blackwell.
set(CMAKE_CUDA_ARCHITECTURES "89;90;100;120;121" CACHE STRING "CUDA architectures")

option(BUILD_TESTS    "Build unit / integration tests" ON)
option(BUILD_EXAMPLES "Build example programs"          ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)   # thermal governor's inter-token pace uses std::this_thread

# This repo is the integrator: pull in sparkinfer-kernels and sparkinfer-moe.
# In a superbuild the targets may already exist; standalone, add the siblings.
if(NOT TARGET sparkinfer_kernels)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../kernels ${CMAKE_BINARY_DIR}/_kernels)
endif()
if(NOT TARGET sparkinfer_moe)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../moe ${CMAKE_BINARY_DIR}/_moe)
endif()

include_directories(include)

file(GLOB_RECURSE RUNTIME_SRC "src/*.cpp" "csrc/cuda/*.cu")
if(WIN32)
  add_library(sparkinfer_runtime STATIC ${RUNTIME_SRC})
else()
  add_library(sparkinfer_runtime SHARED ${RUNTIME_SRC})
endif()
target_include_directories(sparkinfer_runtime PUBLIC include)
target_link_libraries(sparkinfer_runtime PUBLIC sparkinfer_moe sparkinfer_kernels CUDA::cudart Threads::Threads)

# GPU observability (heat/power/clock) uses NVML when available; VRAM works without it. NVML ships
# with the CUDA toolkit as CUDA::nvml — link + flag it when present so the build degrades cleanly
# on boxes/toolkits without the NVML stub.
if(TARGET CUDA::nvml)
    target_link_libraries(sparkinfer_runtime PUBLIC CUDA::nvml)
    target_compile_definitions(sparkinfer_runtime PRIVATE SPARKINFER_HAVE_NVML)
    message(STATUS "sparkinfer: NVML found — GPU temperature/power observability enabled")
else()
    message(STATUS "sparkinfer: NVML not found — GPU observability limited to VRAM")
endif()
target_compile_options(sparkinfer_runtime PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-lineinfo>)
set_target_properties(sparkinfer_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Tensor parallelism (multi-GPU) ----------------------------------------
# OFF by default, deliberately. src/tp/*.cpp always compiles (the shard math and
# backend selection are plain C++ and carry no CUDA-multi-GPU dependency), so the
# default single-GPU build is unchanged and TP=1 takes the same code path it
# always did. Turning this ON only adds the NCCL collective.
option(SPARKINFER_TP "Enable multi-GPU tensor parallelism (requires NCCL)" OFF)

if(SPARKINFER_TP)
    # NCCL is not exported as a CMake package by the CUDA toolkit, so look for it
    # where the toolkit and the distro packages put it.
    find_path(NCCL_INCLUDE_DIR nccl.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} ENV NCCL_HOME ENV CUDA_HOME
        PATH_SUFFIXES include)
    find_library(NCCL_LIBRARY NAMES nccl
        HINTS ${CUDAToolkit_LIBRARY_DIR} ENV NCCL_HOME ENV CUDA_HOME
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

    if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
        target_include_directories(sparkinfer_runtime PUBLIC ${NCCL_INCLUDE_DIR})
        target_link_libraries(sparkinfer_runtime PUBLIC ${NCCL_LIBRARY})
        target_compile_definitions(sparkinfer_runtime PUBLIC SPARKINFER_TP_NCCL)
        message(STATUS "sparkinfer: TP enabled — NCCL ${NCCL_LIBRARY}")

        # ---- fast collectives: peer-oneshot + multimem -----------------------
        # Added to the target only under SPARKINFER_TP, because they need the CUDA
        # driver API (multicast VMM) and multimem.* PTX. Both are internally
        # arch-guarded (__CUDA_ARCH__ >= 900 around the PTX, CUDA_VERSION >= 12010
        # around the driver calls), so they still COMPILE for the sm_89 entry in
        # the default arch list — the multimem path just compiles out there and
        # multimem_allreduce_supported() reports false at run time.
        #
        # NOT VALIDATED ON HARDWARE. Their own headers say so. make_collective()
        # falls back to NCCL when construction fails, and tp_allreduce_check is
        # what proves them. Attribution for the vLLM-derived barrier: see NOTICE.
        file(GLOB TP_FAST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/csrc/tp/*.cu")
        if(TP_FAST_SRC)
            target_sources(sparkinfer_runtime PRIVATE ${TP_FAST_SRC})
            target_compile_definitions(sparkinfer_runtime PUBLIC SPARKINFER_TP_FAST_COLLECTIVES)
            # csrc/tp/*.cu include tp_allreduce.cuh / tp_error.hpp as siblings.
            target_include_directories(sparkinfer_runtime PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/csrc/tp)
            message(STATUS "sparkinfer: TP fast collectives compiled (peer-oneshot, multimem)")
        endif()

        # The CUDA driver API is required by the multimem backend (cuMulticast*)
        # and used to probe NVLS support. Without it the probe reports multicast
        # unsupported, which is the safe answer, and multimem construction fails
        # into the NCCL fallback.
        if(TARGET CUDA::cuda_driver)
            target_link_libraries(sparkinfer_runtime PUBLIC CUDA::cuda_driver)
            target_compile_definitions(sparkinfer_runtime PUBLIC SPARKINFER_TP_CUDA_DRIVER)
            message(STATUS "sparkinfer: TP multicast probe enabled (CUDA driver API)")
        else()
            message(WARNING "sparkinfer: CUDA driver API not found — the multimem "
                            "backend will be unavailable at run time (NCCL fallback)")
        endif()
    else()
        # Hard error, not a silent downgrade: -DSPARKINFER_TP=ON is an explicit
        # request for multi-GPU, and quietly producing a single-GPU binary would
        # let a "TP" benchmark measure one card.
        message(FATAL_ERROR
            "SPARKINFER_TP=ON but NCCL was not found. Install NCCL (libnccl-dev) or set "
            "NCCL_HOME, or configure with -DSPARKINFER_TP=OFF.")
    endif()
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_executable(qwen35_generate examples/qwen35_generate.cpp)
    target_include_directories(qwen35_generate PRIVATE include)
    target_link_libraries(qwen35_generate PRIVATE sparkinfer_runtime CUDA::cudart)
    add_executable(qwen3_gguf_generate examples/qwen3_gguf_generate.cpp)
    target_include_directories(qwen3_gguf_generate PRIVATE include)
    find_package(Threads REQUIRED)   # background GPU-observability sampler uses std::thread
    target_link_libraries(qwen3_gguf_generate PRIVATE sparkinfer_runtime CUDA::cudart Threads::Threads)
    add_executable(qwen3_gguf_bench examples/qwen3_gguf_bench.cpp)
    target_include_directories(qwen3_gguf_bench PRIVATE include)
    target_link_libraries(qwen3_gguf_bench PRIVATE sparkinfer_runtime CUDA::cudart)
    add_executable(qwen3_gguf_cb_bench examples/qwen3_gguf_cb_bench.cpp)
    target_include_directories(qwen3_gguf_cb_bench PRIVATE include)
    find_package(Threads REQUIRED)
    target_link_libraries(qwen3_gguf_cb_bench PRIVATE sparkinfer_runtime CUDA::cudart Threads::Threads)
    add_executable(qwen3_gguf_score examples/qwen3_gguf_score.cpp)
    target_include_directories(qwen3_gguf_score PRIVATE include)
    target_link_libraries(qwen3_gguf_score PRIVATE sparkinfer_runtime CUDA::cudart)
    add_executable(qwen3_gguf_prefill_check examples/qwen3_gguf_prefill_check.cpp)
    target_include_directories(qwen3_gguf_prefill_check PRIVATE include)
    target_link_libraries(qwen3_gguf_prefill_check PRIVATE sparkinfer_runtime CUDA::cudart)
    add_executable(qwen3_gguf_dflash_check examples/qwen3_gguf_dflash_check.cpp)
    target_include_directories(qwen3_gguf_dflash_check PRIVATE include)
    target_link_libraries(qwen3_gguf_dflash_check PRIVATE sparkinfer_runtime CUDA::cudart)
    add_executable(qwen3_gguf_dflash_bench examples/qwen3_gguf_dflash_bench.cpp)
    target_include_directories(qwen3_gguf_dflash_bench PRIVATE include)
    target_link_libraries(qwen3_gguf_dflash_bench PRIVATE sparkinfer_runtime CUDA::cudart)
    # Validate the K3 decode plan + per-rank residency against REAL weights. Reads only
    # the GGUF index, so it is fast even on the 802 GiB model, and it is the tool that
    # turns the plan's inferred widths into pinned ones.
    add_executable(kimi_k3_plan_check examples/kimi_k3_plan_check.cpp)
    target_link_libraries(kimi_k3_plan_check PRIVATE sparkinfer_runtime)
    set_target_properties(kimi_k3_plan_check PROPERTIES CXX_STANDARD 17)

    # Loads real weights for one layer (from a possibly-partial shard set) and runs
    # kimi_k3_forward_layer on a synthetic hidden vector: does it load, execute
    # without a CUDA error, and produce finite, non-degenerate output? First real
    # exercise of the executor — numerical comparison against llama.cpp is separate.
    add_executable(kimi_k3_smoke_test examples/kimi_k3_smoke_test.cpp)
    target_link_libraries(kimi_k3_smoke_test PRIVATE sparkinfer_runtime CUDA::cudart)
    set_target_properties(kimi_k3_smoke_test PROPERTIES CXX_STANDARD 17)

    # Full-layer numerical validation: layer 0 computed twice on the same real
    # weights — once by kimi_k3_forward_layer, once by an independent float64 CPU
    # reference that dequantizes the same GGUF bytes by hand. Proves the EXECUTOR's
    # wiring (not just each kernel in isolation, already covered by
    # kimi_k3_numeric_test) against ground truth built from the real file.
    add_executable(kimi_k3_layer0_ref_check examples/kimi_k3_layer0_ref_check.cpp)
    target_link_libraries(kimi_k3_layer0_ref_check PRIVATE sparkinfer_runtime CUDA::cudart)
    set_target_properties(kimi_k3_layer0_ref_check PROPERTIES CXX_STANDARD 17)

    add_executable(thermal_sweep examples/thermal_sweep.cpp)   # force each thermal mode, measure W/°C/tok/s
    target_include_directories(thermal_sweep PRIVATE include)
    target_link_libraries(thermal_sweep PRIVATE sparkinfer_runtime CUDA::cudart Threads::Threads)

    # Validate the TP collective on real hardware BEFORE any forward path trusts
    # it: correctness on every rank first, then per-call latency at the payloads
    # that matter. Built unconditionally (it runs and reports "no CUDA device"
    # without a GPU, and reports the no-op backend without -DSPARKINFER_TP=ON),
    # because a check you have to opt into is a check nobody runs.
    add_executable(tp_allreduce_check examples/tp_allreduce_check.cpp)
    target_include_directories(tp_allreduce_check PRIVATE include)
    target_link_libraries(tp_allreduce_check PRIVATE sparkinfer_runtime CUDA::cudart)
endif()
