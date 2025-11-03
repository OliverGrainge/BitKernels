cmake_minimum_required(VERSION 3.15)
project(bitgemm VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find and enable OpenMP (optional)
# On macOS with Homebrew, help CMake find libomp
if(APPLE)
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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_compile_definitions(USE_OPENMP)
else()
    message(WARNING "OpenMP not found. Building without parallel support.")
    message(STATUS "To install OpenMP on macOS: brew install libomp")
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

# Kernel selection - can be overridden via command line
# Example: cmake -DKERNEL_SOURCE=kernels/bitlinear_optimized.cpp ..
if(NOT DEFINED KERNEL_SOURCE)
    set(KERNEL_SOURCE "kernels/bitlinear_naive.cpp")
endif()

message(STATUS "Using kernel: ${KERNEL_SOURCE}")

# Source files
set(KERNEL_SOURCES
    ${KERNEL_SOURCE}
)

# Header files
set(HEADER_FILES
    bitlinear.h
)

# Benchmark executable
add_executable(bench
    bench.cpp
    ${KERNEL_SOURCES}
)

# Test/validation executable
add_executable(test
    test.cpp
    ${KERNEL_SOURCES}
)


target_include_directories(bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)


# Optional: Create a library if you want to reuse the code
add_library(bitlinear STATIC
    ${KERNEL_SOURCES}
)

target_include_directories(bitlinear PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link the library to executables
target_link_libraries(bench PRIVATE bitlinear)
target_link_libraries(test PRIVATE bitlinear)

# Link OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(bench PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(test PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(bitlinear PUBLIC OpenMP::OpenMP_CXX)
endif()

# Installation rules (optional)
install(TARGETS bench DESTINATION bin)
install(TARGETS test DESTINATION bin)
install(TARGETS bitlinear DESTINATION lib)
install(FILES ${HEADER_FILES} DESTINATION include)

# Print build configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Flags: ${CMAKE_CXX_FLAGS}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release Flags: ${CMAKE_CXX_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug Flags: ${CMAKE_CXX_FLAGS_DEBUG}")
endif()


