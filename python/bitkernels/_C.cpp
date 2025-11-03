#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <bitkernels/bitlinear.h>
#include <bitkernels/types.h>
#include <memory>

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    m.doc() = "BitKernels: Optimized quantized neural network kernels";
    
    // PackedWeights wrapper
    py::class_<bitkernels::PackedWeights>(m, "PackedWeights")
        .def(py::init<>())
        .def_readonly("N", &bitkernels::PackedWeights::N, "Number of output features")
        .def_readonly("K", &bitkernels::PackedWeights::K, "Number of input features")
        .def_readonly("scale", &bitkernels::PackedWeights::scale, "Weight scale factor")
        .def("__repr__", [](const bitkernels::PackedWeights& w) {
            return "<PackedWeights N=" + std::to_string(w.N) + 
                   " K=" + std::to_string(w.K) + 
                   " scale=" + std::to_string(w.scale) + ">";
        });
    
    // prepare_weights
    m.def("prepare_weights",
        [](py::array_t<float> W, float eps) {
            auto buf = W.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Expected 2D array for weights");
            }
            
            size_t N = buf.shape[0];
            size_t K = buf.shape[1];
            
            if (K % 4 != 0) {
                throw std::runtime_error("K must be divisible by 4");
            }
            
            auto packed = std::make_unique<bitkernels::PackedWeights>();
            bitkernels::prepare_weights(
                static_cast<float*>(buf.ptr), N, K, *packed, eps
            );
            return packed;
        },
        py::arg("weights"), 
        py::arg("eps") = 1e-6f,
        "Prepare and quantize weights to ternary packed format\n\n"
        "Args:\n"
        "    weights: Weight matrix [N x K] as numpy array (float32)\n"
        "    eps: Epsilon for numerical stability (default: 1e-6)\n\n"
        "Returns:\n"
        "    PackedWeights: Quantized and packed weight structure"
    );
    
    // bitlinear_forward (automatic dispatch)
    m.def("bitlinear_forward",
        [](py::array_t<float> X, bitkernels::PackedWeights& W, float eps) {
            auto x_buf = X.request();
            if (x_buf.ndim != 2) {
                throw std::runtime_error("Expected 2D array for activations");
            }
            
            size_t M = x_buf.shape[0];
            size_t K = x_buf.shape[1];
            
            if (K != W.K) {
                throw std::runtime_error("Activation K=" + std::to_string(K) + 
                                       " does not match weight K=" + std::to_string(W.K));
            }
            
            auto Y = py::array_t<float>({(py::ssize_t)M, (py::ssize_t)W.N});
            auto y_buf = Y.request();
            
            bitkernels::bitlinear_forward(
                static_cast<float*>(x_buf.ptr), M, K, W,
                static_cast<float*>(y_buf.ptr), eps
            );
            return Y;
        },
        py::arg("X"), 
        py::arg("weights"), 
        py::arg("eps") = 1e-6f,
        "Forward pass: Y = X @ W^T with ternary weights and INT8 activations\n\n"
        "Automatically dispatches to optimal kernel (GEMM/GEMV) based on input shape.\n\n"
        "Args:\n"
        "    X: Input activations [M x K] as numpy array (float32)\n"
        "    weights: Prepared packed weights\n"
        "    eps: Epsilon for numerical stability (default: 1e-6)\n\n"
        "Returns:\n"
        "    Y: Output activations [M x N] as numpy array (float32)"
    );
    
    // bitlinear_gemm (explicit matrix-matrix)
    m.def("bitlinear_gemm",
        [](py::array_t<float> X, bitkernels::PackedWeights& W, float eps) {
            auto x_buf = X.request();
            if (x_buf.ndim != 2) {
                throw std::runtime_error("Expected 2D array for activations");
            }
            
            size_t M = x_buf.shape[0];
            size_t K = x_buf.shape[1];
            
            if (K != W.K) {
                throw std::runtime_error("Activation K does not match weight K");
            }
            
            auto Y = py::array_t<float>({(py::ssize_t)M, (py::ssize_t)W.N});
            auto y_buf = Y.request();
            
            bitkernels::bitlinear_gemm(
                static_cast<float*>(x_buf.ptr), M, K, W,
                static_cast<float*>(y_buf.ptr), eps
            );
            return Y;
        },
        py::arg("X"), 
        py::arg("weights"), 
        py::arg("eps") = 1e-6f,
        "Matrix-matrix multiplication (optimized for M > 1)"
    );
    
    // bitlinear_gemv (explicit matrix-vector)
    m.def("bitlinear_gemv",
        [](py::array_t<float> X, bitkernels::PackedWeights& W, float eps) {
            auto x_buf = X.request();
            if (x_buf.ndim != 1) {
                throw std::runtime_error("Expected 1D array for GEMV");
            }
            
            size_t K = x_buf.shape[0];
            
            if (K != W.K) {
                throw std::runtime_error("Activation K does not match weight K");
            }
            
            auto Y = py::array_t<float>((py::ssize_t)W.N);
            auto y_buf = Y.request();
            
            bitkernels::bitlinear_gemv(
                static_cast<float*>(x_buf.ptr), K, W,
                static_cast<float*>(y_buf.ptr), eps
            );
            return Y;
        },
        py::arg("X"), 
        py::arg("weights"), 
        py::arg("eps") = 1e-6f,
        "Matrix-vector multiplication (optimized for single input vector)"
    );
}

