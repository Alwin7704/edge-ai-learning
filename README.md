# Edge AI Engineering Curriculum: From Silicon to Intelligence

> **Engineering Mastery Model:**
> `Understand → Explain → Build → Measure → Break → Optimize → Explain again`

Welcome to the engineering notebook and experimental testbed for Edge AI systems engineering. This repository tracks first-principles computer architecture, edge hardware acceleration (CPU, GPU, NPU/DSP), model optimization (quantization, compilation, runtime tuning), computer vision, and end-to-end edge deployment.

---

## 🗺️ Phase 1: Hardware Architecture for Edge AI

| Lesson | Topic | Status | Focus / Key Metric |
| :--- | :--- | :--- | :--- |
| **01** | [Computer Architecture](file:///Users/alwinglifferd/development/Hearsight%20Learning/01-Hardware/01-Computer-Architecture/README.md) | **In Progress** | Von Neumann bottleneck, Arithmetic Intensity, Why CPU vs GPU vs NPU |
| **02** | `CPU Architecture & SIMD` | Planned | Out-of-Order execution, Vector pipelines (Neon, AVX), cache misses |
| **03** | `Memory Hierarchy & Bandwidth` | Planned | L1/L2/L3, DDR/LPDDR, Memory Wall, Roofline model |
| **04** | `GPU Architecture` | Planned | SIMT, Warp/Wavefront execution, Massively parallel tensor ops |
| **05** | `AI Accelerators & Systolic Arrays` | Planned | Spatial architectures, TPU/MAC arrays, Data reuse |
| **06** | `System-on-Chip (SoC) Integration` | Planned | Interconnects (NoC, AXI), Unified memory, Shared cache |
| **07** | `Qualcomm SoC Architecture` | Planned | Snapdragon fabric, Kryo, Adreno, Hexagon, Spectra ISP |
| **08** | `Hexagon NPU & DSP Deep Dive` | Planned | HVX (Hexagon Vector eXtensions), HMX (Matrix), QNN execution |
| **09** | `Camera, ISP & MIPI CSI-2` | Planned | Sensor pipeline, Bayer raw, ISP hardware blocks, zero-copy |
| **10** | `Edge Audio Architecture` | Planned | I2S, PDM microphones, low-power voice wake DSP |
| **11** | `Power Management & DVFS` | Planned | Dynamic voltage/frequency scaling, power rails, joules/inference |
| **12** | `Thermal Throttling & Dissipation` | Planned | Thermal limits, sustained vs burst performance, passive cooling |
| **13** | `Full Hardware Architecture Integration`| Planned | End-to-end SoC dataflow: Camera → ISP → Memory → NPU → Display |

---

## 📊 Benchmark Knowledge Base

Empirical benchmark records are kept in [`benchmarks/benchmark_db.md`](file:///Users/alwinglifferd/development/Hearsight%20Learning/benchmarks/benchmark_db.md).
Every claim is backed by real hardware measurements under controlled single-variable testing.
