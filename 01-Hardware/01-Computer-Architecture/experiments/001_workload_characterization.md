# Experiment #001: Silicon Workload Characterization & Triad Architecture

## Metadata
- **Date**: 2026-10-01
- **Hardware**: Apple M5 Pro (ARM64, Unified Memory Architecture)
- **Host OS**: macOS Darwin
- **Language / Runtime**: Python 3.9.6, NumPy 2.0.2, PyTorch 2.8.0 (MPS / Accelerate BLAS)
- **Target Question**: *Why does modern AI hardware need CPU + GPU + NPU instead of simply using a faster CPU?*

---

## 1. Theory
1. **Branch-Heavy Control Flow**: CPUs commit significant die area to branch prediction tables and speculative execution. When branches are random/unpredictable, pipeline flushes cause a measurable latency penalty. GPUs and NPUs lack deep branch predictors; divergence in SIMT/systolic execution leads to serialization or wasted cycles.
2. **Memory Wall & Arithmetic Intensity (Roofline Model)**:
   - For elementwise operations ($C = A + B$), arithmetic intensity is $\frac{1 \text{ FLOP}}{12 \text{ Bytes}} \approx 0.083 \text{ FLOPs/Byte}$. Memory bandwidth will throttle performance, regardless of how many TFLOPS of compute are available.
   - For dense matrix multiplication ($C = A \times B$), arithmetic intensity is $\frac{2N^3}{12N^2} = \frac{N}{6} \text{ FLOPs/Byte}$ (for $N=1024$, ~170.7 FLOPs/Byte). Compute ALUs can be saturated due to high data reuse.
3. **Hardware Dispatch Overhead**: Accelerators (GPU/NPU) have driver, command queue, and kernel launch overheads (~0.1 to 0.5 ms). For small tensors, the CPU will run circles around the GPU.

---

## 2. Empirical Results

### Workload 1: Branch Predictability on CPU
| Condition | Array Size | Median Latency | Branch Misprediction Penalty |
| :--- | :--- | :--- | :--- |
| **Sorted Data (Predictable Branches)** | 2,000,000 | 25.99 ms | Baseline |
| **Random Data (Unpredictable Branches)** | 2,000,000 | 31.45 ms | **+21.0%** |

### Workload 2: The Memory Wall (Vector Add vs. Dense GEMM)
| Workload | Arithmetic Intensity | Median Latency | Effective Bandwidth | Attained Compute | ALU Utilization Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vector Add (10M floats)** | **0.0833 FLOPs/Byte** | 1.00 ms | 111.88 GB/s | **10.01 GFLOPS** | Stalled on DRAM |
| **Dense GEMM ($1024 \times 1024$)** | **170.67 FLOPs/Byte** | 0.94 ms | - | **2,292.99 GFLOPS** | **229.1x higher ALU throughput** |

### Workload 3: CPU vs. GPU Acceleration Matrix
| Matrix Dimension | CPU Latency | GPU (MPS) Latency | Speedup | Winner | Dominant Factor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$64 \times 64$** | **0.001 ms** | 0.310 ms | 0.003x | **CPU (310x faster)** | GPU Kernel Launch Overhead |
| **$256 \times 256$** | **0.023 ms** | 0.327 ms | 0.07x | **CPU (14x faster)** | Dispatch Queue Overhead |
| **$1024 \times 1024$** | 1.256 ms | **1.109 ms** | 1.13x | **GPU** | Parallel ALU Saturation begins |
| **$2048 \times 2048$** | 10.578 ms | **3.103 ms** | 3.41x | **GPU** | Massive SIMT ALU parallelization |
| **$4096 \times 4096$** | 86.751 ms | **23.262 ms** | 3.73x | **GPU** | Massive compute dominance |

---

## 3. Explanation & Engineering Takeaways

1. **Why we can't use CPU for everything**:
   - For large dense matrix operations (the backbone of neural networks), the CPU maxes out its ALUs early and takes 86.7 ms for a $4096 \times 4096$ GEMM. The GPU finishes in 23.2 ms.
   - Furthermore, the CPU burns 10x-50x more Joules per operation because its instructions must pass through fetch, decode, rename, reorder buffers, and branch prediction stages.
2. **Why we can't use GPU for everything**:
   - For small matrices ($64 \times 64$, typical in small ML embeddings or early pipeline layers), dispatching to GPU took **0.310 ms** while CPU took **0.001 ms**. The GPU was **300x slower** due to dispatch and queue latency.
   - For branch-heavy parsing, OS execution, and file I/O, GPUs stall.
3. **Where the NPU fits**:
   - NPUs hardwire the matrix calculation directly into a 2D grid of MACs (systolic array) fed by high-speed SRAM buffers.
   - They avoid GPU instruction scheduling overhead, avoid CPU branch/OoO overhead, and eliminate DRAM round-trips via spatial data reuse, achieving **1 to 3 picojoules per MAC** compared to **50+ pJ on CPU**.
