"""
Workload Characterization & Silicon Architecture Demonstration
==============================================================
Lesson 01: Computer Architecture for Edge AI
Objective:
    Demonstrate from first principles WHY modern AI hardware requires a
    triad of CPU + GPU + NPU rather than simply a faster CPU.

Workloads:
    1. Branch-Heavy Control Flow (CPU Domain: branch prediction, Out-of-Order execution)
    2. Memory-Bound Streaming Vector Add (Memory Wall & Low Arithmetic Intensity)
    3. Compute-Bound Dense Matrix Multiplication (GEMM: High Arithmetic Intensity)
    4. Hardware Execution Comparison: CPU (Scalar vs. Vectorized) vs. GPU (MPS/CUDA)
"""

import time
import math
import random
import sys
import os

try:
    import numpy as np
except ImportError:
    print("[ERROR] numpy is required. Run 'pip install numpy'")
    sys.exit(1)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


# =====================================================================
# Workload 1: Control-Heavy / Branching Logic (CPU Specialty)
# =====================================================================
def benchmark_branching(size: int = 5_000_000, iterations: int = 5):
    """
    Demonstrates the impact of branch predictability on CPU execution pipelines.
    - Predictable: Sorted data -> branch predictor is >99% accurate -> no pipeline flushes.
    - Unpredictable: Random data -> branch predictor fails ~50% -> deep pipeline stalls.
    """
    print_header("Workload 1: Control-Heavy / Branching (The CPU Domain)")
    print(f"Array Size: {size:,} elements | Repeats: {iterations}")
    print("Concept: Modern CPUs devote massive silicon area to branch prediction.")
    print("GPUs and NPUs do NOT have sophisticated branch predictors and stall violently on branch divergence.\n")

    # Generate random data
    data_random = [random.randint(0, 255) for _ in range(size)]
    data_sorted = sorted(data_random)

    # 1. Benchmark Predictable Branches (Sorted)
    latencies_sorted = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        count = 0
        for val in data_sorted:
            if val > 128:
                count += 1
        t1 = time.perf_counter()
        latencies_sorted.append((t1 - t0) * 1000)

    # 2. Benchmark Unpredictable Branches (Random)
    latencies_random = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        count = 0
        for val in data_random:
            if val > 128:
                count += 1
        t1 = time.perf_counter()
        latencies_random.append((t1 - t0) * 1000)

    med_sorted = sorted(latencies_sorted)[len(latencies_sorted) // 2]
    med_random = sorted(latencies_random)[len(latencies_random) // 2]
    penalty = (med_random - med_sorted) / med_sorted * 100

    print(f"  [Sorted Array - Predictable Branches]   Median Latency: {med_sorted:8.2f} ms")
    print(f"  [Random Array - Unpredictable Branches] Median Latency: {med_random:8.2f} ms")
    print(f"  -> Branch Misprediction Pipeline Penalty: +{penalty:.1f}%\n")
    print("  Key Takeaway: Control flow, OS scheduling, and dynamic parsing require")
    print("  deep branch prediction and OoO logic. This is why edge systems ALWAYS need a CPU.")


# =====================================================================
# Workload 2: Memory-Bound vs. Compute-Bound (Roofline Model)
# =====================================================================
def benchmark_memory_vs_compute(n_elem: int = 10_000_000, matrix_n: int = 1024, iterations: int = 10):
    """
    Demonstrates the difference between:
    1. Low Arithmetic Intensity (Vector Add): 1 FLOP / 12 bytes = 0.083 FLOPs/byte
       Bottleneck: Memory Bandwidth (DRAM Bus)
    2. High Arithmetic Intensity (Matrix Mult GEMM): 2*N^3 FLOPs / (3*N^2 * 4 bytes) = N / 6 FLOPs/byte
       At N=1024: ~170 FLOPs/byte
       Bottleneck: Compute Execution Units (ALUs / Systolic Arrays)
    """
    print_header("Workload 2: Memory Wall vs. Compute Saturation (Roofline Model)")

    # --- Part A: Memory-Bound Vector Addition ---
    print(f"--- Part A: Memory-Bound Vector Addition ({n_elem:,} elements) ---")
    a = np.ones(n_elem, dtype=np.float32)
    b = np.ones(n_elem, dtype=np.float32)

    # 1 FLOP per element (a[i] + b[i])
    # Memory: Read a (4B), Read b (4B), Write c (4B) = 12 Bytes per element
    total_flops_vec = n_elem
    total_bytes_vec = n_elem * 12
    arithmetic_intensity_vec = total_flops_vec / total_bytes_vec

    # Warmup
    for _ in range(3):
        _ = a + b

    times_vec = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        c = a + b
        t1 = time.perf_counter()
        times_vec.append(t1 - t0)

    med_time_vec = sorted(times_vec)[len(times_vec) // 2]
    bw_vec = (total_bytes_vec / (1024 ** 3)) / med_time_vec  # GB/s
    gflops_vec = (total_flops_vec / 1e9) / med_time_vec

    print(f"  Arithmetic Intensity : {arithmetic_intensity_vec:.4f} FLOPs/Byte (EXTREMELY LOW)")
    print(f"  Median Latency       : {med_time_vec * 1000:8.2f} ms")
    print(f"  Effective Bandwidth  : {bw_vec:8.2f} GB/s")
    print(f"  Attained Compute     : {gflops_vec:8.4f} GFLOPS")
    print("  -> Notice how low the GFLOPS number is! The ALU is starving, waiting for DRAM.\n")

    # --- Part B: Compute-Bound Dense GEMM ---
    print(f"--- Part B: Compute-Bound Matrix Multiplication ({matrix_n}x{matrix_n}) ---")
    A = np.random.randn(matrix_n, matrix_n).astype(np.float32)
    B = np.random.randn(matrix_n, matrix_n).astype(np.float32)

    # FLOPs: 2 * N^3
    total_flops_gemm = 2 * (matrix_n ** 3)
    # Memory footprint: 3 * N^2 * 4 bytes
    total_bytes_gemm = 3 * (matrix_n ** 2) * 4
    arithmetic_intensity_gemm = total_flops_gemm / total_bytes_gemm

    # Warmup
    for _ in range(3):
        _ = np.dot(A, B)

    times_gemm = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        C = np.dot(A, B)
        t1 = time.perf_counter()
        times_gemm.append(t1 - t0)

    med_time_gemm = sorted(times_gemm)[len(times_gemm) // 2]
    gflops_gemm = (total_flops_gemm / 1e9) / med_time_gemm

    print(f"  Arithmetic Intensity : {arithmetic_intensity_gemm:.2f} FLOPs/Byte (HIGH DATA REUSE)")
    print(f"  Median Latency       : {med_time_gemm * 1000:8.2f} ms")
    print(f"  Attained Compute     : {gflops_gemm:8.2f} GFLOPS")
    print(f"  -> Compute Speedup vs Vector Add: {gflops_gemm / gflops_vec:.1f}x higher ALU utilization!")
    print("  Key Takeaway: AI models (Convolutions, Linear layers) have high arithmetic intensity.")
    print("  That's why dedicated matrix hardware (GPUs / NPUs) delivers massive speedups for AI,")
    print("  while providing zero benefit for memory-bound vector operations.")


# =====================================================================
# Workload 3: CPU vs. GPU Acceleration Comparison
# =====================================================================
def benchmark_cpu_vs_gpu():
    """
    Compares CPU execution vs GPU execution (via PyTorch MPS or CUDA).
    Shows:
    - Small matrix: CPU wins due to GPU launch overhead and PCIe/bus dispatch.
    - Large matrix: GPU obliterates CPU due to massive parallel ALU density.
    """
    print_header("Workload 3: Hardware Comparison - CPU vs. GPU")

    if not TORCH_AVAILABLE:
        print("PyTorch is not available. Skipping GPU comparison.")
        return

    device_name = "cpu"
    if torch.backends.mps.is_available():
        gpu_device = torch.device("mps")
        device_name = "Apple Silicon GPU (MPS)"
    elif torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        device_name = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
    else:
        gpu_device = None
        print("No GPU accelerator (CUDA or MPS) found on this machine.")
        return

    print(f"Detected GPU Accelerator: {device_name}\n")
    print(f"{'Matrix Size':<12} | {'CPU Latency (ms)':<18} | {'GPU Latency (ms)':<18} | {'Speedup':<10} | {'Winner':<10}")
    print("-" * 75)

    test_sizes = [64, 256, 1024, 2048, 4096]

    for N in test_sizes:
        # Generate Tensors on CPU
        x_cpu = torch.randn(N, N, dtype=torch.float32)
        y_cpu = torch.randn(N, N, dtype=torch.float32)

        # CPU Warmup & Measurement
        for _ in range(3):
            _ = torch.matmul(x_cpu, y_cpu)
        
        t0 = time.perf_counter()
        reps = 10 if N <= 1024 else 3
        for _ in range(reps):
            _ = torch.matmul(x_cpu, y_cpu)
        t_cpu = ((time.perf_counter() - t0) / reps) * 1000

        # GPU Warmup & Measurement
        x_gpu = x_cpu.to(gpu_device)
        y_gpu = y_cpu.to(gpu_device)

        # Warmup and sync
        for _ in range(3):
            z = torch.matmul(x_gpu, y_gpu)
        if gpu_device.type == "mps":
            torch.mps.synchronize()
        elif gpu_device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(reps):
            z = torch.matmul(x_gpu, y_gpu)
            if gpu_device.type == "mps":
                torch.mps.synchronize()
            elif gpu_device.type == "cuda":
                torch.cuda.synchronize()
        t_gpu = ((time.perf_counter() - t0) / reps) * 1000

        speedup = t_cpu / t_gpu
        winner = "CPU" if speedup < 1.0 else "GPU"

        print(f"{f'{N}x{N}':<12} | {t_cpu:15.3f} ms | {t_gpu:15.3f} ms | {speedup:8.2f}x | {winner:<10}")

    print("\n  Engineering Insight:")
    print("  1. For small matrices (64x64), CPU is faster because GPU kernel launch overhead,")
    print("     command queue dispatch, and context switching dominate execution.")
    print("  2. For large matrices (1024+), GPU dominates because the thousands of parallel ALUs")
    print("     are fully saturated with work.")
    print("  3. An NPU takes this one step further: hardwiring the matrix multiplication into")
    print("     a 2D systolic array with local SRAM, achieving 10x higher TOPS/Watt than the GPU!")


if __name__ == "__main__":
    benchmark_branching(size=2_000_000, iterations=5)
    benchmark_memory_vs_compute(n_elem=10_000_000, matrix_n=1024, iterations=10)
    benchmark_cpu_vs_gpu()
    print("\n[Done] All workload characterization experiments complete.")
