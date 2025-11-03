cmake_minimum_required(VERSION 3.15)
project(BitKernels VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_PYTHON "Build Python bindings" OFF)
option(FORCE_COMMON_KERNELS "Force use of common/portable kernels instead of optimized ones" OFF)

# ============================================================================
# Compiler Detection and Flags
# ============================================================================

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    set(ARCH_ARM TRUE)
    message(STATUS "Detected ARM architecture")
    add_compile_definitions(BITKERNELS_ARM)
    
    # ARM-specific flags
    if(NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+fp+simd")
    endif()
    
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(ARCH_X86 TRUE)
    message(STATUS "Detected x86_64 architecture")
    add_compile_definitions(BITKERNELS_X86)
    
    # x86-specific flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
else()
    message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# ============================================================================
# Find OpenMP (optional but recommended)
# ============================================================================

if(APPLE)
    # Help CMake find libomp on macOS with Homebrew
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(LIBOMP_PREFIX)
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    add_compile_definitions(USE_OPENMP)
else()
    message(WARNING "OpenMP not found. Building without parallel support.")
    if(APPLE)
        message(STATUS "To install OpenMP on macOS: brew install libomp")
    endif()
endif()

# ============================================================================
# Compiler Flags
# ============================================================================

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type" FORCE)
endif()

# Option to disable architecture-specific optimizations for testing
if(FORCE_COMMON_KERNELS)
    add_compile_definitions(FORCE_COMMON_KERNELS)
    message(STATUS "FORCE_COMMON_KERNELS enabled: using common/portable kernels only")
endif()

# ============================================================================
# Source Files
# ============================================================================

# Common sources (always compiled)
set(COMMON_SOURCES
    src/common/types.cpp
    src/bitlinear/prepare_weights.cpp
    src/bitlinear/dispatch.cpp
)

# Common/portable kernels (always compiled as fallback)
set(COMMON_KERNEL_SOURCES
    src/bitlinear/common/gemm.cpp
    src/bitlinear/common/gemv.cpp
)

# Architecture-specific sources
set(KERNEL_SOURCES "")

if(ARCH_ARM)
    list(APPEND KERNEL_SOURCES
        src/bitlinear/arm_neon/gemm.cpp
        src/bitlinear/arm_neon/gemv.cpp
    )
elseif(ARCH_X86)
    # Add x86 kernels here when implemented
    # list(APPEND KERNEL_SOURCES
    #     src/bitlinear/x86_avx/gemm.cpp
    #     src/bitlinear/x86_avx/gemv.cpp
    # )
    message(STATUS "Using common/portable kernels (x86 optimized kernels not yet implemented)")
endif()

# ============================================================================
# BitKernels Library
# ============================================================================

add_library(bitkernels STATIC
    ${COMMON_SOURCES}
    ${COMMON_KERNEL_SOURCES}
    ${KERNEL_SOURCES}
)

target_include_directories(bitkernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(bitkernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(bitkernels PUBLIC OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# Tests
# ============================================================================

if(BUILD_TESTS)
    add_executable(test_bitlinear
        tests/cpp/test_bitlinear.cpp
    )
    
    target_link_libraries(test_bitlinear PRIVATE bitkernels)
    
    target_include_directories(test_bitlinear PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(BUILD_BENCHMARKS)
    add_executable(bench_bitlinear
        benchmarks/bench_bitlinear.cpp
    )
    
    target_link_libraries(bench_bitlinear PRIVATE bitkernels)
    
    target_include_directories(bench_bitlinear PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()

# ============================================================================
# Python Bindings
# ============================================================================

if(BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
    
    pybind11_add_module(_C
        python/bitkernels/_C.cpp
    )
    
    target_link_libraries(_C PRIVATE bitkernels)
    
    target_include_directories(_C PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS bitkernels
    EXPORT BitKernelsTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/bitkernels
    DESTINATION include
)

if(BUILD_TESTS)
    install(TARGETS test_bitlinear DESTINATION bin)
endif()

if(BUILD_BENCHMARKS)
    install(TARGETS bench_bitlinear DESTINATION bin)
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "BitKernels Configuration Summary:")
message(STATUS "  Version:         ${PROJECT_VERSION}")
message(STATUS "  Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Compiler:    ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ Standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "  Architecture:    ${CMAKE_SYSTEM_PROCESSOR}")
if(ARCH_ARM)
    message(STATUS "  Optimized:       ARM NEON")
elseif(ARCH_X86)
    message(STATUS "  Optimized:       None (using common kernels)")
else()
    message(STATUS "  Optimized:       None (using common kernels)")
endif()
message(STATUS "  Fallback:        Common/portable kernels")
message(STATUS "  OpenMP:          ${OpenMP_CXX_FOUND}")
message(STATUS "  Build tests:     ${BUILD_TESTS}")
message(STATUS "  Build benchmarks:${BUILD_BENCHMARKS}")
message(STATUS "  Build Python:    ${BUILD_PYTHON}")
message(STATUS "")
