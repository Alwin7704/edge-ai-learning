# Edge AI Benchmark Knowledge Base

This database maintains empirical performance records across all experiments conducted on edge and host silicon.

## Protocol
1. **Single-Variable Discipline**: Only change one variable per benchmark run (Precision, Batch, Hardware Backend, Resolution, or Runtime).
2. **Warmup & Cooldown**: Discard the first 10 runs (cold cache/compilation overhead), record 100+ steady-state runs, report median + standard deviation or 95th percentile.
3. **Thermal State**: Note starting and ending die temperature when benchmarking sustained workloads.

---

## Benchmark Log

| ID | Date | Workload / Kernel | Hardware Backend | Precision | Input Dim | Median Latency | Throughput (FPS/GFLOPS) | Memory (RSS) | Power / Temp | Notes / Bottleneck |
|---|---|---|---|---|---|---:|---:|---:|---:|---|
| **#001a** | 2026-10-01 | Vector Add ($C = A + B$) | Apple M5 Pro CPU | FP32 | 10M elements | 1.00 ms | 10.01 GFLOPS (111.88 GB/s) | ~120 MB | Ambient | **Memory-Bound**: $I = 0.083$ FLOPs/Byte. ALU starved. |
| **#001b** | 2026-10-01 | Dense GEMM ($C = A \times B$) | Apple M5 Pro CPU (Accelerate) | FP32 | 1024x1024 | 1.26 ms | ~1,710 GFLOPS | ~16 MB | Ambient | **Compute-Bound**: $I = 170.7$ FLOPs/Byte. CPU ALUs saturated. |
| **#001c** | 2026-10-01 | Dense GEMM ($C = A \times B$) | Apple M5 Pro GPU (MPS) | FP32 | 1024x1024 | 1.11 ms | ~1,940 GFLOPS | ~16 MB | Ambient | GPU reaches crossover point against CPU. |
| **#001d** | 2026-10-01 | Dense GEMM ($C = A \times B$) | Apple M5 Pro CPU (Accelerate) | FP32 | 4096x4096 | 86.75 ms | ~1,585 GFLOPS | ~64 MB | Ambient | Large matrix: CPU core count limits compute. |
| **#001e** | 2026-10-01 | Dense GEMM ($C = A \times B$) | Apple M5 Pro GPU (MPS) | FP32 | 4096x4096 | 23.26 ms | ~5,910 GFLOPS | ~64 MB | Ambient | **3.73x Speedup**: High thread parallelism fully saturates GPU. |
| **#001f** | 2026-10-01 | Micro GEMM ($C = A \times B$) | Apple M5 Pro CPU | FP32 | 64x64 | 0.001 ms | - | <1 MB | Ambient | **CPU wins by 310x**: Dispatch latency dominates GPU (0.31 ms). |

---
