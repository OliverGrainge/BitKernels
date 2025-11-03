#include "bitlinear.h"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstdlib>

int main(int argc, char* argv[]) {
    std::cout << "BitLinear Ternary Weight × INT8 Activation Benchmark" << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << std::endl;
    
    // Parse command-line arguments for matrix dimensions
    // Usage: ./bitlinear_benchmark [M] [K] [N]
    size_t M = 128;   // Batch size (default)
    size_t K = 4096;  // Input features (default, must be divisible by 4)
    size_t N = 4096;  // Output features (default)
    
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
        
        // Validate K is divisible by 4
        if (K % 4 != 0) {
            std::cerr << "Error: K must be divisible by 4 for packing" << std::endl;
            std::cerr << "Provided K=" << K << ", rounding up to " << ((K + 3) / 4) * 4 << std::endl;
            K = ((K + 3) / 4) * 4;
        }
    } else if (argc > 1) {
        std::cout << "Usage: " << argv[0] << " [M] [K] [N]" << std::endl;
        std::cout << "Using default dimensions..." << std::endl;
        std::cout << std::endl;
    }
    
    // Epsilon for numerical stability
    const float eps = 1e-6f;
    
    std::cout << "Matrix dimensions:" << std::endl;
    std::cout << "  M (batch size):     " << M << std::endl;
    std::cout << "  K (input features): " << K << std::endl;
    std::cout << "  N (output features):" << N << std::endl;
    std::cout << "  Epsilon:            " << eps << std::endl;
    std::cout << std::endl;
    
    // Allocate and initialize weights
    std::cout << "Initializing weights..." << std::endl;
    float* W_fp32 = new float[N * K];
    
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    
    for (size_t i = 0; i < N * K; ++i) {
        W_fp32[i] = dist(gen);
    }
    
    // Prepare (quantize and pack) weights
    std::cout << "Quantizing weights to ternary..." << std::endl;
    bitlinear::PackedTernaryWeights packed_weights;
    bitlinear::prepare_weights(W_fp32, N, K, packed_weights, eps);
    
    std::cout << "  Weight scale: " << packed_weights.scale << std::endl;
    std::cout << "  Packed size:  " << (N * K / 4) << " bytes" << std::endl;
    std::cout << std::endl;
    
    // Allocate and initialize input activations
    std::cout << "Initializing activations..." << std::endl;
    float* X_fp32 = new float[M * K];
    
    for (size_t i = 0; i < M * K; ++i) {
        X_fp32[i] = dist(gen);
    }
    
    // Allocate output
    float* Y_fp32 = new float[M * N];
    
    // Warmup run
    std::cout << "Warmup run..." << std::endl;
    bitlinear::linear(X_fp32, M, K, packed_weights, Y_fp32, eps);
    std::cout << std::endl;
    
    // Benchmark
    const int num_runs = 10;
    std::cout << "Running benchmark (" << num_runs << " iterations)..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; ++i) {
        bitlinear::linear(X_fp32, M, K, packed_weights, Y_fp32, eps);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    double time_per_run_ms = (elapsed.count() * 1000.0) / num_runs;
    double time_per_run_s = time_per_run_ms / 1000.0;
    
    std::cout << std::endl;
    std::cout << "Benchmark Results:" << std::endl;
    std::cout << "==================" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Average time per run: " << time_per_run_ms << " ms" << std::endl;
    std::cout << std::endl;
    

        // Memory bandwidth estimate
    // We read: M*K activations (FP32) + N*K/4 packed weights (UINT8)
    // We write: M*N outputs (FP32)
    double bytes_read_activations = M * K * sizeof(float);
    double bytes_read_weights = N * K / 4.0;  // Packed weights
    double bytes_written = M * N * sizeof(float);
    double total_bytes = bytes_read_activations + bytes_read_weights + bytes_written;
    double bandwidth_gb_s = (total_bytes / 1e9) / time_per_run_s;
    
    std::cout << "Memory Bandwidth:" << std::endl;
    std::cout << "  Data transferred:     " << total_bytes / 1e6 << " MB" << std::endl;
    std::cout << "  Bandwidth:            " << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << std::endl;
    
    // Compression statistics
    size_t original_weight_bytes = N * K * sizeof(float);
    size_t packed_weight_bytes = N * K / 4;
    double compression_ratio = static_cast<double>(original_weight_bytes) / packed_weight_bytes;
    
    std::cout << "Compression Statistics:" << std::endl;
    std::cout << "  Original weights:     " << original_weight_bytes / 1e6 << " MB" << std::endl;
    std::cout << "  Packed weights:       " << packed_weight_bytes / 1e6 << " MB" << std::endl;
    std::cout << "  Compression ratio:    " << compression_ratio << "x" << std::endl;
    std::cout << std::endl;


    // Calculate GFLOPS
    // For matrix multiplication Y[M×N] = X[M×K] @ W[N×K]^T
    // Number of operations = M × N × K × 2 (multiply-add)
    double operations = 2.0 * M * N * K;
    double gflops = (operations / 1e9) / time_per_run_s;
    
    std::cout << "  Total operations:     " << operations / 1e9 << " billion" << std::endl;
    std::cout << "  FLOPS:                " << gflops << " GFLOPS" << std::endl;
    std::cout << std::endl;

    
    // Cleanup
    delete[] W_fp32;
    delete[] X_fp32;
    delete[] Y_fp32;
    
    return 0;
}
