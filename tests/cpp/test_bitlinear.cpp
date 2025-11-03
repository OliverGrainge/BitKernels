#include "bitkernels/bitlinear.h"
#include "bitkernels/types.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <cstring>

// ============================================================================
// Utility Functions
// ============================================================================

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

// Error metrics
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

// ============================================================================
// Test Functions
// ============================================================================

bool test_gemm(size_t M, size_t K, size_t N, float eps) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing GEMM (M=" << M << ", K=" << K << ", N=" << N << ")" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Load test data
    float* X = load_matrix("data/test_X.bin", M, K);
    float* W = load_matrix("data/test_W.bin", N, K);
    float* bias = load_matrix("data/test_bias.bin", 1, N);
    float* Y_python_no_bias = load_matrix("data/test_Y_python_no_bias.bin", M, N);
    float* Y_python = load_matrix("data/test_Y_python.bin", M, N);
    
    if (!X || !W || !bias || !Y_python_no_bias || !Y_python) {
        std::cerr << "Failed to load test data. Run bitlinear.py first!" << std::endl;
        delete[] X;
        delete[] W;
        delete[] bias;
        delete[] Y_python_no_bias;
        delete[] Y_python;
        return false;
    }
    
    // Prepare weights
    bitkernels::PackedWeights packed_weights;
    bitkernels::prepare_weights(W, N, K, packed_weights, eps);
    
    // Test without bias
    std::cout << "\n--- Testing without bias ---" << std::endl;
    float* Y_cpp_no_bias = new float[M * N];
    std::memset(Y_cpp_no_bias, 0, M * N * sizeof(float));
    bitkernels::bitlinear_forward(X, M, K, packed_weights, Y_cpp_no_bias, nullptr, eps);
    
    ErrorMetrics metrics_no_bias = compare_outputs(Y_cpp_no_bias, Y_python_no_bias, M, N);
    std::cout << "Error Metrics (no bias):" << std::endl;
    metrics_no_bias.print();
    bool passed_no_bias = metrics_no_bias.is_close(1e-4, 1e-3);
    
    // Test with bias
    std::cout << "\n--- Testing with bias ---" << std::endl;
    float* Y_cpp = new float[M * N];
    std::memset(Y_cpp, 0, M * N * sizeof(float));
    bitkernels::bitlinear_forward(X, M, K, packed_weights, Y_cpp, bias, eps);
    
    ErrorMetrics metrics = compare_outputs(Y_cpp, Y_python, M, N);
    std::cout << "Error Metrics (with bias):" << std::endl;
    metrics.print();
    bool passed_with_bias = metrics.is_close(1e-4, 1e-3);
    
    bool passed = passed_no_bias && passed_with_bias;
    
    // Cleanup
    delete[] X;
    delete[] W;
    delete[] bias;
    delete[] Y_python_no_bias;
    delete[] Y_python;
    delete[] Y_cpp_no_bias;
    delete[] Y_cpp;
    
    return passed;
}

bool test_gemv(size_t K, size_t N, float eps) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing GEMV (K=" << K << ", N=" << N << ")" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Load test data (use first row of test data)
    float* X_full = load_matrix("data/test_X.bin", 128, K);
    float* W = load_matrix("data/test_W.bin", N, K);
    float* bias = load_matrix("data/test_bias.bin", 1, N);
    float* Y_python_no_bias_full = load_matrix("data/test_Y_python_no_bias.bin", 128, N);
    float* Y_python_full = load_matrix("data/test_Y_python.bin", 128, N);
    
    if (!X_full || !W || !bias || !Y_python_no_bias_full || !Y_python_full) {
        std::cerr << "Failed to load test data." << std::endl;
        delete[] X_full;
        delete[] W;
        delete[] bias;
        delete[] Y_python_no_bias_full;
        delete[] Y_python_full;
        return false;
    }
    
    // Use first row only
    float* X = X_full;
    float* Y_python_no_bias = Y_python_no_bias_full;
    float* Y_python = Y_python_full;
    
    // Prepare weights
    bitkernels::PackedWeights packed_weights;
    bitkernels::prepare_weights(W, N, K, packed_weights, eps);
    
    // Test without bias
    std::cout << "\n--- Testing without bias ---" << std::endl;
    float* Y_cpp_no_bias = new float[N];
    std::memset(Y_cpp_no_bias, 0, N * sizeof(float));
    bitkernels::bitlinear_gemv(X, K, packed_weights, Y_cpp_no_bias, nullptr, eps);
    
    ErrorMetrics metrics_no_bias = compare_outputs(Y_cpp_no_bias, Y_python_no_bias, 1, N);
    std::cout << "Error Metrics (no bias):" << std::endl;
    metrics_no_bias.print();
    bool passed_no_bias = metrics_no_bias.is_close(1e-4, 1e-3);
    
    // Test with bias
    std::cout << "\n--- Testing with bias ---" << std::endl;
    float* Y_cpp = new float[N];
    std::memset(Y_cpp, 0, N * sizeof(float));
    bitkernels::bitlinear_gemv(X, K, packed_weights, Y_cpp, bias, eps);
    
    ErrorMetrics metrics = compare_outputs(Y_cpp, Y_python, 1, N);
    std::cout << "Error Metrics (with bias):" << std::endl;
    metrics.print();
    bool passed_with_bias = metrics.is_close(1e-4, 1e-3);
    
    bool passed = passed_no_bias && passed_with_bias;
    
    // Cleanup
    delete[] X_full;
    delete[] W;
    delete[] bias;
    delete[] Y_python_no_bias_full;
    delete[] Y_python_full;
    delete[] Y_cpp_no_bias;
    delete[] Y_cpp;
    
    return passed;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "BitKernels Test Suite" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    const float eps = 1e-6f;
    
    // Test configuration (must match Python test data)
    const size_t M = 128;
    const size_t K = 256;
    const size_t N = 256;
    
    bool all_passed = true;
    
    // Test 1: GEMM
    bool gemm_passed = test_gemm(M, K, N, eps);
    all_passed &= gemm_passed;
    
    // Test 2: GEMV
    bool gemv_passed = test_gemv(K, N, eps);
    all_passed &= gemv_passed;
    
    // Summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "  GEMM: " << (gemm_passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "  GEMV: " << (gemv_passed ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    if (all_passed) {
        std::cout << "✓ ALL TESTS PASSED!" << std::endl;
    } else {
        std::cout << "✗ SOME TESTS FAILED!" << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl;
    
    return all_passed ? 0 : 1;
}

