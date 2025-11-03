#include "bitlinear.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip>

// Load binary matrix (row-major, float32)
float* load_matrix(const std::string& filename, size_t rows, size_t cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return nullptr;
    }
    
    float* data = new float[rows * cols];
    file.read(reinterpret_cast<char*>(data), rows * cols * sizeof(float));
    
    if (!file) {
        std::cerr << "Error: Failed to read " << filename << std::endl;
        delete[] data;
        return nullptr;
    }
    
    std::cout << "Loaded " << filename << ": [" << rows << " x " << cols << "]" << std::endl;
    return data;
}

// Compute error metrics
struct ErrorMetrics {
    double max_abs_error;
    double mean_abs_error;
    double relative_error;
    size_t num_elements;
    
    void print() const {
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "  Max absolute error:  " << max_abs_error << std::endl;
        std::cout << "  Mean absolute error: " << mean_abs_error << std::endl;
        std::cout << "  Relative error:      " << relative_error * 100.0 << "%" << std::endl;
    }
    
    bool is_close(double abs_tol = 1e-4, double rel_tol = 1e-3) const {
        return (max_abs_error < abs_tol) || (relative_error < rel_tol);
    }
};

ErrorMetrics compare_outputs(const float* y_cpp, const float* y_python, size_t M, size_t N) {
    ErrorMetrics metrics = {0.0, 0.0, 0.0, M * N};
    
    double sum_abs_error = 0.0;
    double sum_abs_python = 0.0;
    
    for (size_t i = 0; i < M * N; ++i) {
        double error = std::abs(y_cpp[i] - y_python[i]);
        sum_abs_error += error;
        sum_abs_python += std::abs(y_python[i]);
        
        if (error > metrics.max_abs_error) {
            metrics.max_abs_error = error;
        }
    }
    
    metrics.mean_abs_error = sum_abs_error / (M * N);
    metrics.relative_error = sum_abs_python > 0 ? sum_abs_error / sum_abs_python : 0.0;
    
    return metrics;
}

void print_statistics(const float* Y, size_t M, size_t N, const std::string& name) {
    double sum = 0.0, sum_sq = 0.0;
    float min_val = Y[0], max_val = Y[0];
    
    for (size_t i = 0; i < M * N; ++i) {
        sum += Y[i];
        sum_sq += Y[i] * Y[i];
        if (Y[i] < min_val) min_val = Y[i];
        if (Y[i] > max_val) max_val = Y[i];
    }
    
    double mean = sum / (M * N);
    double variance = (sum_sq / (M * N)) - (mean * mean);
    double std_dev = std::sqrt(variance);
    
    std::cout << name << " statistics:" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Mean:   " << mean << std::endl;
    std::cout << "  Std:    " << std_dev << std::endl;
    std::cout << "  Min:    " << min_val << std::endl;
    std::cout << "  Max:    " << max_val << std::endl;
    std::cout << "  Sample: ";
    for (size_t i = 0; i < 5 && i < M * N; ++i) {
        std::cout << Y[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "BitLinear C++/Python Equivalence Validation" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    // Dimensions (must match Python)
    const size_t M = 128;
    const size_t K = 256;
    const size_t N = 256;
    
    // Epsilon for numerical stability (must match Python)
    const float eps = 1e-6f;
    
    std::cout << "\nDimensions: M=" << M << ", K=" << K << ", N=" << N << std::endl;
    std::cout << "Epsilon: " << eps << std::endl;
    
    // Load inputs
    std::cout << "\n[1/4] Loading inputs..." << std::endl;
    float* X = load_matrix("data/test_X.bin", M, K);
    float* W = load_matrix("data/test_W.bin", N, K);
    
    if (!X || !W) {
        std::cerr << "Failed to load inputs. Run bitlinear.py first to generate test data!" << std::endl;
        return 1;
    }
    
    // Load Python output
    std::cout << "\n[2/4] Loading Python output..." << std::endl;
    float* Y_python = load_matrix("data/test_Y_python.bin", M, N);
    if (!Y_python) {
        delete[] X;
        delete[] W;
        return 1;
    }
    
    // Compute C++ output
    std::cout << "\n[3/4] Computing C++ result..." << std::endl;
    bitlinear::PackedTernaryWeights packed_weights;
    bitlinear::prepare_weights(W, N, K, packed_weights, eps);
    
    float* Y_cpp = new float[M * N];
    bitlinear::linear(X, M, K, packed_weights, Y_cpp, eps);
    
    std::cout << "  C++ computation complete!" << std::endl;
    
    // Compare outputs
    std::cout << "\n[4/4] Comparing outputs..." << std::endl;
    std::cout << "\n";
    print_statistics(Y_python, M, N, "Python");
    std::cout << "\n";
    print_statistics(Y_cpp, M, N, "C++");
    
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "Error Metrics:" << std::endl;
    ErrorMetrics metrics = compare_outputs(Y_cpp, Y_python, M, N);
    metrics.print();
    
    // Verdict
    std::cout << "\n" << std::string(60, '=') << std::endl;
    if (metrics.is_close(1e-4, 1e-3)) {
        std::cout << "✓ PASS: C++ and Python implementations match!" << std::endl;
    } else {
        std::cout << "✗ FAIL: Significant differences detected!" << std::endl;
        std::cout << "\nPossible issues:" << std::endl;
        std::cout << "  - Quantization logic differences" << std::endl;
        std::cout << "  - Packing/unpacking errors" << std::endl;
        std::cout << "  - Numerical precision issues" << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl;
    
    // Cleanup
    delete[] X;
    delete[] W;
    delete[] Y_python;
    delete[] Y_cpp;
    
    return metrics.is_close() ? 0 : 1;
}