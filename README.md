# BitKernels

High-performance quantized neural network kernels with support for BitLinear (ternary weight × INT8 activation) operations.

## Features

- **BitLinear Operations**: Efficient ternary weight {-1, 0, 1} × INT8 activation linear layers
- **Multiple Backends**: Optimized implementations for ARM NEON (x86 AVX coming soon)
- **Automatic Dispatch**: Runtime selection of optimal kernel (GEMM vs GEMV)
- **Python Bindings**: Easy-to-use Python interface via pybind11
- **Optimized Performance**: Tiled computation, SIMD vectorization, OpenMP parallelization

## Architecture

```
BitKernels/
├── include/bitkernels/    # Public API headers
├── src/                    # Implementation
│   ├── common/            # Common utilities
│   ├── bitlinear/         # BitLinear operations
│   │   ├── arm_neon/     # ARM NEON kernels
│   │   └── x86_avx/      # x86 AVX kernels (future)
├── python/                # Python bindings
├── tests/                 # C++ tests
├── benchmarks/            # Performance benchmarks
└── data/                  # Test data
```

## Building from Source

### Prerequisites

- CMake 3.15+
- C++17 compatible compiler
- OpenMP (optional but recommended)
- pybind11 (for Python bindings)

### C++ Library and Tools

```bash
# Quick build
./build.sh

# Or manually
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

This builds:
- `libbitkernels.a` - Static library
- `test_bitlinear` - Test suite
- `bench_bitlinear` - Benchmark tool

### Python Package

```bash
cd python
pip install -e .
```

## Usage

### C++ API

```cpp
#include "bitkernels/bitlinear.h"

// Prepare weights (one-time operation)
bitkernels::PackedWeights weights;
bitkernels::prepare_weights(W_fp32, N, K, weights);

// Forward pass (automatic dispatch)
bitkernels::bitlinear_forward(X_fp32, M, K, weights, Y_fp32);

// Or use explicit kernels
bitkernels::bitlinear_gemm(X_fp32, M, K, weights, Y_fp32);  // Matrix-matrix
bitkernels::bitlinear_gemv(X_fp32, K, weights, Y_fp32);     // Matrix-vector
```

### Python API

```python
import numpy as np
import bitkernels

# Prepare weights
W = np.random.randn(N, K).astype(np.float32)
packed_weights = bitkernels.prepare_weights(W)

# Forward pass
X = np.random.randn(M, K).astype(np.float32)
Y = bitkernels.bitlinear_forward(X, packed_weights)
```

## Testing

```bash
# Generate test data and run tests
./test.sh

# Or manually
python bitlinear.py              # Generate test data
./build/test_bitlinear           # Run C++ tests
```

## Benchmarking

```bash
# Default dimensions (M=128, K=4096, N=4096)
./bench.sh

# Custom dimensions
./bench.sh 256 2048 2048

# Or directly
./build/bench_bitlinear [M] [K] [N]
```

## Performance

On Apple M3 Max (ARM):
- **GEMM** (M=128, K=4096, N=4096): ~150 GFLOPS
- **GEMV** (K=4096, N=4096): ~50 GFLOPS

Features:
- 4× weight compression (2-bit ternary encoding)
- Tiled computation for cache efficiency
- NEON SDOT/VMULL vectorization
- OpenMP parallelization

## API Reference

### `prepare_weights`

Quantize and pack weights to ternary format.

```cpp
void prepare_weights(
    const float* weights_fp32,  // [N×K] weight matrix
    size_t N,                    // Number of output features
    size_t K,                    // Number of input features (must be divisible by 4)
    PackedWeights& packed_out,   // Output packed weights
    float eps = 1e-6f           // Numerical stability epsilon
);
```

### `bitlinear_forward`

Forward pass with automatic kernel dispatch.

```cpp
void bitlinear_forward(
    const float* X_fp32,              // [M×K] input activations
    size_t M,                          // Batch size
    size_t K,                          // Input features
    const PackedWeights& packed_weights, // Prepared weights
    float* Y_fp32,                     // [M×N] output (pre-allocated)
    float eps = 1e-6f                 // Numerical stability epsilon
);
```

### `bitlinear_gemm` / `bitlinear_gemv`

Explicit kernel selection for matrix-matrix or matrix-vector operations.

## Adding New Architectures

To add support for a new architecture (e.g., x86 AVX):

1. Create `src/bitlinear/x86_avx/gemm.cpp` and `gemv.cpp`
2. Implement `x86_avx::bitlinear_gemm_impl()` and `bitlinear_gemv_impl()`
3. Update `src/bitlinear/dispatch.cpp` to detect and use new kernels
4. Update `CMakeLists.txt` to compile new sources

## License

MIT License (or your choice)

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- Tests pass (`./test.sh`)
- Benchmarks run without errors (`./bench.sh`)

