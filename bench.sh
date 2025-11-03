#!/bin/bash
# bench.sh - Run benchmarks for BitKernels

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
M="${1:-128}"
K="${2:-4096}"
N="${3:-4096}"

echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}BitKernels Benchmark Runner${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Configuration:"
echo "  M (batch):  ${M}"
echo "  K (input):  ${K}"
echo "  N (output): ${N}"
echo ""

# Build if needed
if [ ! -f "build/bench_bitlinear" ]; then
    echo -e "${YELLOW}Building benchmarks...${NC}"
    ./build.sh Release
    echo ""
fi

# Set OpenMP environment
if command -v sysctl &> /dev/null; then
    NUM_CORES=$(sysctl -n hw.physicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "1")
elif command -v nproc &> /dev/null; then
    NUM_CORES=$(nproc)
else
    NUM_CORES=1
fi

export OMP_NUM_THREADS=${NUM_CORES}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo -e "${YELLOW}OpenMP Configuration:${NC}"
echo "  Threads: ${OMP_NUM_THREADS}"
echo "  Proc bind: ${OMP_PROC_BIND}"
echo "  Places: ${OMP_PLACES}"
echo ""

# Run benchmark
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}Running Benchmarks${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

./build/bench_bitlinear ${M} ${K} ${N}

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}Benchmark complete!${NC}"
echo -e "${GREEN}============================================================${NC}"

