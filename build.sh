#!/bin/bash
# build.sh - Build BitKernels library, tests, and benchmarks

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}BitKernels Build Script${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Parse options
BUILD_TYPE="${1:-Release}"
CLEAN_BUILD="${2:-false}"

if [[ "$CLEAN_BUILD" == "clean" ]]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "${BUILD_DIR}"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DBUILD_TESTS=ON \
      -DBUILD_BENCHMARKS=ON \
      "${SCRIPT_DIR}"

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Configuration complete${NC}"
echo ""

# Build
echo -e "${YELLOW}Building...${NC}"
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Build complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Executables:"
echo "  Tests:      ${BUILD_DIR}/test_bitlinear"
echo "  Benchmarks: ${BUILD_DIR}/bench_bitlinear"
echo ""
echo "To run:"
echo "  ./build/test_bitlinear"
echo "  ./build/bench_bitlinear [M] [K] [N]"
echo ""

