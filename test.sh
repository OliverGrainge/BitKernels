#!/bin/bash
# test.sh - Run tests for BitKernels

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}BitKernels Test Runner${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Step 1: Generate test data with Python
echo -e "${YELLOW}[1/3] Generating test data with Python...${NC}"
python bitlinear.py
echo ""

# Step 2: Build if needed
if [ ! -f "build/test_bitlinear" ]; then
    echo -e "${YELLOW}[2/3] Building tests...${NC}"
    ./build.sh Release
else
    echo -e "${YELLOW}[2/3] Tests already built${NC}"
fi
echo ""

# Step 3: Run tests
echo -e "${YELLOW}[3/3] Running C++ tests...${NC}"
./build/test_bitlinear

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo -e "${GREEN}============================================================${NC}"
else
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}✗ Some tests failed!${NC}"
    echo -e "${RED}============================================================${NC}"
fi

exit $EXIT_CODE

