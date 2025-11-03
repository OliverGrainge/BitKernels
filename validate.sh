#!/bin/bash

# validate.sh - Quick validation of Python/C++ equivalence
#
# Usage:
#   ./validate.sh [kernel_file]
#
# Examples:
#   ./validate.sh                                  # Use default kernel
#   ./validate.sh kernels/bitlinear_naive.cpp      # Validate specific kernel
#   ./validate.sh kernels/bitlinear_optimized.cpp  # Validate optimized kernel

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# Parse kernel argument (default to naive)
KERNEL_FILE="${1:-kernels/bitlinear_naive.cpp}"
KERNEL_NAME=$(basename "$KERNEL_FILE" .cpp)

# Check if kernel file exists
if [ -f "${KERNEL_FILE}" ]; then
    KERNEL_FULL_PATH="${KERNEL_FILE}"
elif [ -f "${SCRIPT_DIR}/${KERNEL_FILE}" ]; then
    KERNEL_FULL_PATH="${SCRIPT_DIR}/${KERNEL_FILE}"
else
    echo -e "${RED}Error: Kernel file '${KERNEL_FILE}' not found${NC}"
    exit 1
fi

# Convert to relative path for CMake
if [[ "${KERNEL_FULL_PATH}" == "${SCRIPT_DIR}"* ]]; then
    KERNEL_RELATIVE="${KERNEL_FULL_PATH#${SCRIPT_DIR}/}"
else
    KERNEL_RELATIVE="${KERNEL_FILE}"
fi

echo -e "${BLUE}🔍 Running Python/C++ Equivalence Validation${NC}"
echo -e "${YELLOW}Kernel: ${KERNEL_RELATIVE}${NC}"
echo ""

# Step 1: Generate test data with Python
echo -e "${YELLOW}[1/3] Generating test data and computing Python result...${NC}"
python bitlinear.py
echo ""

# Step 2: Build C++ validation with specified kernel
echo -e "${YELLOW}[2/3] Building C++ test with ${KERNEL_NAME}...${NC}"
cd build
cmake -DKERNEL_SOURCE="${KERNEL_RELATIVE}" .. > /dev/null 2>&1
cmake --build . --target test --config Release > /dev/null 2>&1
cd ..
echo -e "${GREEN}  ✓ Build complete${NC}"
echo ""

# Step 3: Run validation
echo -e "${YELLOW}[3/3] Running validation...${NC}"
./build/test

echo ""
echo -e "${GREEN}✅ Validation complete for ${KERNEL_NAME}!${NC}"

