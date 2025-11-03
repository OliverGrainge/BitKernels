import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

def bitlinear(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    """
    BitLinear forward pass with ternary weights and INT8 activations.
    
    Args:
        x: Input tensor [M × K]
        w: Weight tensor [N × K]
        bias: Optional bias tensor [N]
        eps: Epsilon for numerical stability
        
    Returns:
        Output tensor [M × N]
    """
    # Per-channel (per-row) activation quantization
    sx = x.abs().max(dim=-1, keepdim=True).values / 127.0
    sx = sx.clamp(min=eps)
    qx = (x / sx).round().clamp(-127, 127)
    
    # Per-tensor weight quantization
    sw = w.abs().mean()
    sw = sw.clamp(min=eps)
    qw = (w / sw).round().clamp(-1, 1)
    
    # Integer matmul (simulated in float) + dequantization
    return F.linear(qx * sx, qw * sw, bias)


def save_matrix(filename: str, tensor: torch.Tensor):
    """Save tensor as binary file (float32, row-major)"""
    import os
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', filename)
    np.array(tensor.numpy(), dtype=np.float32).tofile(filepath)
    print(f"Saved {filepath}: shape {tuple(tensor.shape)}")


def load_matrix(filename: str, shape: tuple) -> torch.Tensor:
    """Load binary file as tensor"""
    import os
    filepath = os.path.join('data', filename)
    data = np.fromfile(filepath, dtype=np.float32).reshape(shape)
    return torch.from_numpy(data)


def main():
    # Test dimensions (K must be divisible by 4 for C++ packing)
    M, K, N = 128, 256, 256
    
    print("="*60)
    print("BitLinear Python/C++ Validation")
    print("="*60)
    print(f"\nDimensions: M={M}, K={K}, N={N}")
    
    # Generate test data with fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    X = torch.randn(M, K)  # Activations
    W = torch.randn(N, K)  # Weights
    bias = torch.randn(N)  # Bias
    
    print("\n[1/5] Generating test data...")
    print(f"  X (activations): {X.shape}")
    print(f"  W (weights):     {W.shape}")
    print(f"  bias:            {bias.shape}")
    
    # Save inputs for C++
    print("\n[2/5] Saving inputs to binary files...")
    save_matrix('test_X.bin', X)
    save_matrix('test_W.bin', W)
    save_matrix('test_bias.bin', bias)
    
    # Compute Python result without bias
    print("\n[3/5] Computing Python result (no bias)...")
    Y_python_no_bias = bitlinear(X, W, bias=None)
    save_matrix('test_Y_python_no_bias.bin', Y_python_no_bias)
    print(f"  Y_python (no bias): {Y_python_no_bias.shape}")
    
    # Compute Python result with bias
    print("\n[4/5] Computing Python result (with bias)...")
    Y_python = bitlinear(X, W, bias=bias)
    save_matrix('test_Y_python.bin', Y_python)
    print(f"  Y_python (with bias): {Y_python.shape}")
    
    # Statistics
    print("\n[5/5] Python output statistics:")
    print(f"  Mean:   {Y_python.mean().item():.6f}")
    print(f"  Std:    {Y_python.std().item():.6f}")
    print(f"  Min:    {Y_python.min().item():.6f}")
    print(f"  Max:    {Y_python.max().item():.6f}")
    print(f"  Sample: {Y_python[0, :5].tolist()}")
    
    print("\n" + "="*60)
    print("Test data saved to data/ directory")
    print("Run validation with: ./test.sh")
    print("="*60)


if __name__ == "__main__":
    main()