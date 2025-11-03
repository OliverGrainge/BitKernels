#!/bin/bash

# run.sh - Build and benchmark BitGEMM with different kernels and matrix dimensions
#
# Usage:
#   ./run.sh <kernel_file> [M] [K] [N]
#
# Examples:
#   ./run.sh kernels/bitlinear_naive.cpp                 # Use default dimensions (128, 4096, 4096)
#   ./run.sh kernels/bitlinear_naive.cpp 1024 1024 1024  # Custom dimensions
#   ./run.sh kernels/bitlinear_optimized.cpp 512 2048 2048

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build"

# Print usage
print_usage() {
    echo "Usage: $0 <kernel_file> [M] [K] [N]"
    echo ""
    echo "Arguments:"
    echo "  kernel_file    Path to kernel implementation (e.g., kernels/bitlinear_naive.cpp)"
    echo "  M              Batch size (default: 128)"
    echo "  K              Input features - must be divisible by 4 (default: 4096)"
    echo "  N              Output features (default: 4096)"
    echo ""
    echo "Examples:"
    echo "  $0 kernels/bitlinear_naive.cpp"
    echo "  $0 kernels/bitlinear_naive.cpp 1024 1024 1024"
    echo "  $0 kernels/bitlinear_optimized.cpp 512 2048 2048"
    exit 1
}

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Missing kernel file argument${NC}"
    echo ""
    print_usage
fi

KERNEL_FILE="$1"
KERNEL_NAME=$(basename "$KERNEL_FILE" .cpp)

# Check if kernel file exists (handle both absolute and relative paths)
if [ -f "${KERNEL_FILE}" ]; then
    # Absolute path or valid relative path from current directory
    KERNEL_FULL_PATH="${KERNEL_FILE}"
elif [ -f "${SCRIPT_DIR}/${KERNEL_FILE}" ]; then
    # Relative path from script directory
    KERNEL_FULL_PATH="${SCRIPT_DIR}/${KERNEL_FILE}"
else
    echo -e "${RED}Error: Kernel file '${KERNEL_FILE}' not found${NC}"
    exit 1
fi

# Convert to relative path for CMake if it's within the project
if [[ "${KERNEL_FULL_PATH}" == "${SCRIPT_DIR}"* ]]; then
    KERNEL_RELATIVE="${KERNEL_FULL_PATH#${SCRIPT_DIR}/}"
else
    KERNEL_RELATIVE="${KERNEL_FILE}"
fi

# Parse optional matrix dimensions
M="${2:-128}"
K="${3:-4096}"
N="${4:-4096}"

# Validate K is divisible by 4
if [ $((K % 4)) -ne 0 ]; then
    echo -e "${YELLOW}Warning: K must be divisible by 4, rounding up...${NC}"
    K=$(( ((K + 3) / 4) * 4 ))
fi

# Print configuration
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}BitGEMM Benchmark Runner${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Kernel:          ${KERNEL_FILE}"
echo "  Kernel name:     ${KERNEL_NAME}"
echo "  Matrix shape:    M=${M}, K=${K}, N=${N}"
echo "  Build directory: ${BUILD_DIR}"
echo ""

# Create build directory
echo -e "${YELLOW}[1/3] Creating build directory...${NC}"
mkdir -p "${BUILD_DIR}"

# Configure with CMake
echo -e "${YELLOW}[2/3] Configuring with CMake...${NC}"
cd "${BUILD_DIR}"
cmake -DCMAKE_BUILD_TYPE=Release \
      -DKERNEL_SOURCE="${KERNEL_RELATIVE}" \
      "${SCRIPT_DIR}"

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

echo -e "${GREEN}  ✓ CMake configuration successful${NC}"

# Build
echo -e "${YELLOW}[3/3] Building benchmark...${NC}"
cmake --build . --target bench

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}  ✓ Build successful${NC}"
echo ""

# Set OpenMP environment for optimal performance
if command -v sysctl &> /dev/null; then
    # macOS
    NUM_CORES=$(sysctl -n hw.physicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "1")
elif command -v nproc &> /dev/null; then
    # Linux
    NUM_CORES=$(nproc)
else
    NUM_CORES=1
fi

export OMP_NUM_THREADS=${NUM_CORES}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Run benchmark
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Running Benchmark${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}OpenMP Configuration:${NC}"
echo "  Threads: ${OMP_NUM_THREADS}"
echo "  Proc bind: ${OMP_PROC_BIND}"
echo "  Places: ${OMP_PLACES}"
echo ""

./bench ${M} ${K} ${N}

BENCHMARK_EXIT_CODE=$?

echo ""
if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Benchmark completed successfully!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
else
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}Benchmark failed with exit code: ${BENCHMARK_EXIT_CODE}${NC}"
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
    exit $BENCHMARK_EXIT_CODE
fi

