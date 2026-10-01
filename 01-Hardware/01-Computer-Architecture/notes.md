# Deep Dive Notes: Computer Architecture for Edge AI

## 1. The Core Mental Models

### A. The Von Neumann Bottleneck vs. Modern Heterogeneous SoCs
In 1945, John von Neumann formulated the stored-program computer:
- Shared memory for instructions and data.
- Central Processing Unit (ALU + Control Unit).
- A narrow bus connecting memory and processing.

#### The Problem:
An ALU can perform an operation in ~0.2 nanoseconds (several GHz clock). But fetching an operand from off-chip DRAM takes 50–100 nanoseconds (~200–500 clock cycles). This means an ALU running without local cache or data reuse spends **>99% of its life waiting for the bus**.

#### The Modern SoC Solution:
Instead of sending every operation through one general-purpose bus to one processor:
1. **Hierarchical Caching**: L1 ($) $\rightarrow$ L2 ($) $\rightarrow$ L3 / System-Level Cache (SLC).
2. **Unified Memory Architecture (UMA)**: CPU, GPU, ISP, and NPU share a single high-bandwidth physical memory pool, avoiding expensive PCIe or copy buffers between discrete memory pools.
3. **Specialized Compute Fabrics**:
   - Workloads are dispatched to dedicated silicon tailored to that workload's mathematical and dataflow profile.

---

### B. Flynn's Taxonomy for Edge Processors

| Category | Full Name | Hardware Example | Operation Profile | Best Workload |
| :--- | :--- | :--- | :--- | :--- |
| **SISD** | Single Instruction, Single Data | Classical scalar CPU core (e.g. legacy ARM Cortex-M0) | One instruction operates on one data value at a time. | Sequential logic, pointer manipulation, state machines. |
| **SIMD** | Single Instruction, Multiple Data | ARM Neon, Intel AVX-512, Hexagon HVX | One instruction broadcasts across a 128-bit to 1024-bit vector register containing multiple values. | Audio filters, image transforms, vector math. |
| **MIMD** | Multiple Instruction, Multiple Data | Multi-core CPU clusters (e.g., 4x Cortex-A78 + 4x Cortex-A55) | Independent cores running completely independent instruction streams on different data. | Multi-threaded OS tasks, concurrent services. |
| **SIMT** | Single Instruction, Multiple Threads | GPUs (Qualcomm Adreno, Apple Metal, NVIDIA) | Thousands of threads execute the same kernel instruction in lockstep across SIMD lanes (warps/wavefronts). | 3D graphics rasterization, massively parallel matrix math. |
| **Spatial / Systolic** | Dataflow / Systolic Array | NPUs (Qualcomm Hexagon NPU, Google TPU, Apple ANE) | Instructions do not steer each individual MAC; data streams through a 2D mesh of connected processing elements. | 2D GEMM (Matrix Multiply), 2D Convolution. |

---

### C. The Roofline Model: Arithmetic Intensity

The Roofline Model defines the maximum attainable performance ($P$) of a processor as a function of **Arithmetic Intensity** ($I$):
$$I = \frac{\text{Total Operations (FLOPs or MACs)}}{\text{Total Bytes Transferred from Memory (Bytes)}}$$

$$P = \min\left(\text{Peak Compute Performance (GFLOPS)}, \; I \times \text{Peak Memory Bandwidth (GB/s)}\right)$$

```text
Attained Performance (GFLOPS)
       ▲
 Peak  │                  ┌─────────────────────────────── (Compute-Bound Ceiling)
Compute│                 /
Ceiling│                /
       │               /
       │              /
       │             /
       │            /
       │           /  (Memory-Bound Slope = Peak Bandwidth)
       │          /
       └─────────┴────────────────────────────────────────►
                 I_ridge      Arithmetic Intensity (FLOPs/Byte)
```

#### The Two Regimes:
1. **Memory-Bound Regime ($I < I_{\text{ridge}}$)**:
   - Performance is constrained entirely by how fast DRAM can supply bytes.
   - Adding more ALUs or increasing clock speed yields **zero performance gain**.
   - Optimization strategy: Model compression, quantization (FP32 $\rightarrow$ INT8 cuts bytes by 4x), operator fusion (e.g., fusing Conv + BatchNorm + ReLU so intermediate tensors stay in cache).
2. **Compute-Bound Regime ($I > I_{\text{ridge}}$)**:
   - The memory bus is supplying bytes faster than the ALUs can finish computing.
   - Performance hits the flat ceiling of peak FLOPS.
   - Optimization strategy: Utilize lower-precision math (FP16/INT8) to unlock higher MAC throughput per clock, improve ALU instruction scheduling.

---

### D. The Energy Equation: Why NPU > GPU > CPU for Edge AI

Why does a dedicated NPU achieve **15–30 TOPS/Watt** while an edge CPU struggles to reach **0.5 TOPS/Watt**?

```text
CPU Instruction Overhead:
[Fetch] -> [Decode] -> [Rename] -> [Reorder Buffer] -> [Branch Predict] -> [ALU Execute] -> [Retire]
   ▲                                                                             │
   └─────────────────────── >85% of Energy is Control Overhead ──────────────────┘

NPU Systolic Execution:
[Data Stream In] ---> [MAC] ---> [MAC] ---> [MAC] ---> [Output Stream Out]
   ▲                   │          │          │
   └── Local SRAM ─────┴──────────┴──────────┴──── >90% of Energy goes to MATH
```

1. **Instruction Overhead Elimination**:
   - In a CPU, executing `c = a * b` requires instruction cache fetch, decode logic, register renaming, out-of-order dependency checking, and writeback. The actual arithmetic consumes less than 10% of the total energy of the instruction cycle.
   - In an NPU, a single matrix instruction sets up the systolic array to execute millions of MAC operations in lockstep without fetching new instructions for every operation.
2. **Memory Proximity (Data Locality)**:
   - A CPU fetches from L3 cache or DRAM for almost every cache line.
   - An NPU uses large dedicated on-chip **Tightly-Coupled SRAM (TCM)** right next to the MAC array. Weights are loaded once and stationary in the array, while activations flow through horizontally.

---

## 2. Answers to the "5 Questions" Rule

### 1. What is it?
A heterogeneous computer architecture combining general-purpose serial processors (CPUs), data-parallel throughput processors (GPUs), and domain-specific spatial tensor accelerators (NPUs) connected via a shared, high-bandwidth interconnect and unified memory system.

### 2. Why does it exist?
Because Dennard scaling broke down, thermal envelopes on edge devices are strictly capped (2W–5W), and no single silicon architecture can simultaneously deliver microsecond serial response time (OS/control), massive thread throughput (rendering/computer vision), and high-efficiency tensor math (neural networks) within that power budget.

### 3. How does it work?
- The **CPU** orchestrates system state, executes the operating system, handles network/sensor I/O, parses dynamic logic, and schedules inference graph execution.
- The **GPU** executes massively parallel floating-point shaders, preprocessing pipelines, and non-standard tensor operations via SIMT workgroups.
- The **NPU** executes compiled tensor execution graphs (subgraphs with fixed shapes and supported operators) using dedicated 2D systolic MAC arrays and local SRAM buffers to minimize DRAM traffic.

### 4. When should I use it?
- **Use CPU**: Sequential control logic, pipeline orchestration, pre/post-processing with unpredictable branching (e.g. Non-Maximum Suppression with variable bounding box counts, tree traversals, JSON/sensor parsing), and small tensor workloads where accelerator dispatch overhead exceeds execution time.
- **Use GPU**: Complex image transformations, rendering, and custom neural network layers with unsupported operators that require floating-point flexibility without CPU serialization.
- **Use NPU**: Continuous, real-time deep neural network inference (YOLO, MobileNet, Whisper, embeddings) under strict battery, thermal, and frame-rate constraints.

### 5. What can go wrong?
- **Host-Device Dispatch Bottleneck**: Offloading small kernels to NPU/GPU causes driver dispatch and context-switching overhead that dwarfs the execution time (as observed in Experiment #001: 64x64 matrix was 310x slower on GPU than CPU).
- **Unsupported Operator Fallback**: If an NPU compiler encounters an unsupported layer (e.g., custom activation, dynamic reshaping), the runtime falls back to CPU. The resulting round-trip memory copy across the bus destroys frame latency.
- **Memory Bandwidth Choking**: Low-arithmetic-intensity layers (e.g. depthwise separable convs, layer norm, elementwise add) starve the compute units, causing the accelerator to sit idle while DRAM bandwidth is saturated.
- **Thermal Throttling**: Sustained 100% utilization of both CPU and NPU can exceed the device's passive thermal dissipation capacity, triggering dynamic frequency scaling (DVFS) and degrading FPS over time.

---

## 3. Knowledge Gate Checklist (Self-Test)

Before proceeding to **Lesson 02 (CPU Architecture & SIMD)**, verify you can answer these without looking at notes:

- [x] **Q1**: What is the difference between latency-oriented and throughput-oriented architectures?
  - *Answer*: Latency-oriented architectures (CPUs) use large caches, deep pipelines, branch predictors, and out-of-order execution to make a single thread complete as quickly as possible. Throughput-oriented architectures (GPUs/NPUs) prioritize aggregate work done per unit time across thousands of threads or data elements, tolerating individual thread latency by context switching or pipelining dataflow.
- [x] **Q2**: Why is reading a 32-bit float from DRAM ~1,000x more expensive than computing a 32-bit addition in an ALU?
  - *Answer*: Charging long microscopic copper interconnect traces across the silicon die and PCB memory bus to charge/discharge DRAM capacitive cells requires orders of magnitude more electrical charge ($\approx 1000\text{--}2000\text{ pJ}$) than switching a few dozen local logic gates inside an ALU ($\approx 0.1\text{ pJ}$).
- [x] **Q3**: What is Arithmetic Intensity and why does it dictate model performance?
  - *Answer*: It is the ratio of operations performed to bytes of memory transferred ($\text{FLOPs/Byte}$). If the intensity is lower than the hardware's ridge point, performance is throttled by memory bandwidth (memory-bound). If it is higher, performance is limited by ALU capacity (compute-bound).
- [x] **Q4**: Why does a $64 \times 64$ matrix multiplication run faster on a CPU than on a GPU?
  - *Answer*: The overhead of preparing the command buffer, kernel launch queue dispatch, driver scheduling, and hardware synchronization takes hundreds of microseconds ($\approx 0.3\text{ ms}$), whereas the CPU can execute the few thousand math operations directly in local registers and L1 cache in $\approx 0.001\text{ ms}$.
- [x] **Q5**: Why can't we simply increase a CPU's clock frequency from 3 GHz to 30 GHz to run AI models?
  - *Answer*: Power dissipation scales with $V^2 \cdot f$. Due to the end of Dennard scaling (leakage current at sub-micron scales preventing voltage scaling), pushing clock frequency past 3–5 GHz causes power and heat to spike exponentially, exceeding the thermal design power (TDP) limit of physical cooling systems.
