#include "bitkernels/bitlinear.h"
#include "bitkernels/types.h"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct BenchConfig {
    size_t M, K, N;
    int num_warmup;
    int num_runs;
    float eps;
};

// ============================================================================
// Benchmark Results
// ============================================================================

struct BenchResults {
    double time_ms;
    double gflops;
    double bandwidth_gb_s;
    size_t memory_bytes;
    
    void print() const {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Time:       " << time_ms << " ms" << std::endl;
        std::cout << "  GFLOPS:     " << gflops << std::endl;
        std::cout << "  Bandwidth:  " << bandwidth_gb_s << " GB/s" << std::endl;
        std::cout << "  Memory:     " << memory_bytes / 1e6 << " MB" << std::endl;
    }
};

// ============================================================================
// Benchmark Functions
// ============================================================================

BenchResults benchmark_gemm(const BenchConfig& cfg) {
    std::cout << "\nBenchmarking GEMM (M=" << cfg.M << ", K=" << cfg.K 
              << ", N=" << cfg.N << ")" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    // Allocate and initialize data
    float* W = new float[cfg.N * cfg.K];
    float* X = new float[cfg.M * cfg.K];
    float* Y = new float[cfg.M * cfg.N];
    
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    
    for (size_t i = 0; i < cfg.N * cfg.K; ++i) W[i] = dist(gen);
    for (size_t i = 0; i < cfg.M * cfg.K; ++i) X[i] = dist(gen);
    
    // Prepare weights
    bitkernels::PackedWeights packed_weights;
    bitkernels::prepare_weights(W, cfg.N, cfg.K, packed_weights, cfg.eps);
    
    // Warmup
    for (int i = 0; i < cfg.num_warmup; ++i) {
        bitkernels::bitlinear_gemm(X, cfg.M, cfg.K, packed_weights, Y, cfg.eps);
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cfg.num_runs; ++i) {
        bitkernels::bitlinear_gemm(X, cfg.M, cfg.K, packed_weights, Y, cfg.eps);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double time_s = elapsed.count() / cfg.num_runs;
    double time_ms = time_s * 1000.0;
    
    // Calculate metrics
    double operations = 2.0 * cfg.M * cfg.N * cfg.K;
    double gflops = (operations / 1e9) / time_s;
    
    size_t bytes_read_X = cfg.M * cfg.K * sizeof(float);
    size_t bytes_read_W = cfg.N * cfg.K / 4;  // Packed
    size_t bytes_write_Y = cfg.M * cfg.N * sizeof(float);
    size_t total_bytes = bytes_read_X + bytes_read_W + bytes_write_Y;
    double bandwidth = (total_bytes / 1e9) / time_s;
    
    // Cleanup
    delete[] W;
    delete[] X;
    delete[] Y;
    
    return {time_ms, gflops, bandwidth, total_bytes};
}

BenchResults benchmark_gemv(const BenchConfig& cfg) {
    std::cout << "\nBenchmarking GEMV (K=" << cfg.K << ", N=" << cfg.N << ")" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    // Allocate and initialize data
    float* W = new float[cfg.N * cfg.K];
    float* X = new float[cfg.K];
    float* Y = new float[cfg.N];
    
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    
    for (size_t i = 0; i < cfg.N * cfg.K; ++i) W[i] = dist(gen);
    for (size_t i = 0; i < cfg.K; ++i) X[i] = dist(gen);
    
    // Prepare weights
    bitkernels::PackedWeights packed_weights;
    bitkernels::prepare_weights(W, cfg.N, cfg.K, packed_weights, cfg.eps);
    
    // Warmup
    for (int i = 0; i < cfg.num_warmup; ++i) {
        bitkernels::bitlinear_gemv(X, cfg.K, packed_weights, Y, cfg.eps);
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cfg.num_runs; ++i) {
        bitkernels::bitlinear_gemv(X, cfg.K, packed_weights, Y, cfg.eps);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double time_s = elapsed.count() / cfg.num_runs;
    double time_ms = time_s * 1000.0;
    
    // Calculate metrics
    double operations = 2.0 * cfg.N * cfg.K;  // M=1
    double gflops = (operations / 1e9) / time_s;
    
    size_t bytes_read_X = cfg.K * sizeof(float);
    size_t bytes_read_W = cfg.N * cfg.K / 4;  // Packed
    size_t bytes_write_Y = cfg.N * sizeof(float);
    size_t total_bytes = bytes_read_X + bytes_read_W + bytes_write_Y;
    double bandwidth = (total_bytes / 1e9) / time_s;
    
    // Cleanup
    delete[] W;
    delete[] X;
    delete[] Y;
    
    return {time_ms, gflops, bandwidth, total_bytes};
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "============================================================" << std::endl;
    std::cout << "BitKernels Benchmark Suite" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    // Parse command-line arguments
    size_t M = 128;
    size_t K = 4096;
    size_t N = 4096;
    
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
        
        if (K % 4 != 0) {
            std::cerr << "Error: K must be divisible by 4" << std::endl;
            K = ((K + 3) / 4) * 4;
            std::cout << "Adjusted K to " << K << std::endl;
        }
    } else if (argc > 1) {
        std::cout << "Usage: " << argv[0] << " [M] [K] [N]" << std::endl;
        std::cout << "Using default dimensions..." << std::endl;
    }
    
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  M (batch):      " << M << std::endl;
    std::cout << "  K (input):      " << K << std::endl;
    std::cout << "  N (output):     " << N << std::endl;
    
    BenchConfig cfg = {M, K, N, 3, 10, 1e-6f};
    
    // Run benchmarks
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Running Benchmarks" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    BenchResults gemm_results = benchmark_gemm(cfg);
    std::cout << "\nGEMM Results:" << std::endl;
    gemm_results.print();
    
    BenchResults gemv_results = benchmark_gemv(cfg);
    std::cout << "\nGEMV Results:" << std::endl;
    gemv_results.print();
    
    // Summary table
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Summary" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << std::left << std::setw(15) << "Kernel" 
              << std::right << std::setw(12) << "Time (ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(15) << "BW (GB/s)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::left << std::setw(15) << "GEMM"
              << std::right << std::setw(12) << gemm_results.time_ms
              << std::setw(12) << gemm_results.gflops
              << std::setw(15) << gemm_results.bandwidth_gb_s << std::endl;
    std::cout << std::left << std::setw(15) << "GEMV"
              << std::right << std::setw(12) << gemv_results.time_ms
              << std::setw(12) << gemv_results.gflops
              << std::setw(15) << gemv_results.bandwidth_gb_s << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}

