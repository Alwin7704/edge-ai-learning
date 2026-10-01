# Lesson 01: Computer Architecture for Edge AI

> **Sequence**: Understand → Explain → Build → Measure → Break → Optimize → Explain Again  
> **Core Focus**: Why modern edge systems require a heterogeneous mix of CPU + GPU + NPU instead of a faster CPU.

---

## 📑 Table of Contents
1. [Level 1: Vocabulary & Definitions](#1-level-1-vocabulary--definitions)
2. [Level 2: Mental Model & First Principles](#2-level-2-mental-model--first-principles)
   - [The Von Neumann Architecture & The Memory Wall](#the-von-neumann-architecture--the-memory-wall)
   - [The Breakdown of Dennard Scaling & TDP Limits](#the-breakdown-of-dennard-scaling--the-tdp-limit)
   - [The Energy Equation of Computation](#the-energy-equation-of-computation)
   - [The Triad: CPU vs. GPU vs. NPU](#the-triad-cpu-vs-gpu-vs-npu)
3. [Level 3: Implementation & Empirical Code](#3-level-3-implementation--empirical-code)
   - [Experiment #001: Branchy Control vs. Memory Streaming vs. Dense GEMM](#experiment-001-workload-characteristics)
4. [Level 4: Hardware Measurement & Roofline Analysis](#4-level-4-hardware-measurement--roofline-analysis)
5. [Level 5: Engineering Breakdown & Failure Modes](#5-level-5-engineering-breakdown--failure-modes)
6. [The 5 Questions Checklist](#the-5-questions-checklist)
7. [Knowledge Gate Verification](#knowledge-gate-verification)

---

## 1. Level 1: Vocabulary & Definitions

Before engineering, lock down the terminology:

| Term | Definition | Edge AI Relevance |
| :--- | :--- | :--- |
| **ALU (Arithmetic Logic Unit)** | Hardware circuit that executes elementary arithmetic (`+`, `*`) and logic (`AND`, `OR`) operations. | How many ALUs a chip packs determines peak raw compute capacity (FLOPs). |
| **Control Unit (CU)** | Silicon block that fetches instructions, decodes opcodes, manages out-of-order scheduling, and directs dataflow. | CPUs spend >60% of die area on control logic to minimize latency on arbitrary code; NPUs discard almost all of it. |
| **Register File** | Tiny array of ultra-fast storage registers located directly adjacent to ALUs (sub-nanosecond access). | Lowest energy operand access (~0.05 pJ). Reusing registers prevents bus trips. |
| **Cache Hierarchy (L1, L2, L3/SLC)** | Staged SRAM buffers bridging the speed gap between lightning-fast ALUs and slow external DRAM. | Misses in cache incur severe latency penalties (100–300 CPU cycles of stalled execution). |
| **DRAM / LPDDR** | Dynamic RAM external to the processor die (or stacked via Package-on-Package). High density, volatile. | Main memory. The single largest consumer of power on an edge AI board during inference. |
| **Von Neumann Bottleneck** | The physical throughput and latency limit imposed by sharing or transferring data across a bus between compute and memory. | Driving motivation behind modern NPU on-chip SRAM scratchpads and near-memory computing. |
| **Arithmetic (Operational) Intensity** | Ratio of mathematical operations (FLOPs or MACs) performed per byte of memory loaded from DRAM: $\text{Intensity} = \frac{\text{FLOPs}}{\text{Bytes transferred}}$. | Dictates whether a layer is **memory-bandwidth bound** or **compute bound**. |
| **MAC (Multiply-Accumulate)** | Operation: $a \leftarrow a + (b \times c)$. Equal to 2 FLOPs. | The atomic primitive of all Deep Learning (dense linear layers, convolutions, attention projection). |
| **Roofline Model** | Theoretical bound graphing attainable compute performance against arithmetic intensity. | The visual compass for finding where your model bottlenecks on hardware. |
| **SISD / SIMD / SIMT** | Flynn's taxonomy: Single Instruction Single Data (scalar CPU), Single Instruction Multiple Data (vector/Neon), Single Instruction Multiple Threads (GPU warps). | Understanding how parallel execution units schedule parallel instructions. |

---

## 2. Level 2: Mental Model & First Principles

### The Fundamental Question
> **"Why does modern AI hardware need CPU + GPU + NPU instead of simply using a faster CPU?"**

To answer this, we must look at physics and silicon architecture from first principles.

### The Von Neumann Architecture & The Memory Wall
In classical Von Neumann computer architecture, the CPU and memory are separated by a bus:

```text
┌──────────────────────────────────────────────┐
│                    CPU                       │
│  ┌──────────────┐       ┌─────────────────┐  │
│  │ Control Unit │◄─────►│ Registers / ALU │  │
│  └──────┬───────┘       └────────┬────────┘  │
│         │                        │           │
│         ▼                        ▼           │
│  ┌────────────────────────────────────────┐  │
│  │        L1 / L2 / L3 SRAM Cache         │  │
│  └───────────────────┬────────────────────┘  │
└──────────────────────┼───────────────────────┘
                       │ System Bus (Narrow, High Latency)
                       ▼
┌──────────────────────────────────────────────┐
│                  DRAM (RAM)                  │
│       Stores Instructions + Data             │
└──────────────────────────────────────────────┘
```

Between 1980 and 2005, CPU compute speed grew at **~50% per year**, while DRAM access latency only improved at **~7% per year**. This created the **Memory Wall**: a CPU core capable of executing billions of instructions per second spends 70–90% of its clock cycles idle ("stalled"), waiting for data to arrive from external DRAM.

### The Breakdown of Dennard Scaling & The TDP Limit
Until ~2006, **Dennard Scaling** held true: as transistors shrank, their power density remained constant because operating voltage could drop proportionally with gate length:
$$P = \alpha \cdot C \cdot V^2 \cdot f$$
- $\alpha$: Activity factor
- $C$: Capacitance
- $V$: Voltage
- $f$: Clock frequency

Around 2006, voltage could no longer be reduced without leakage current (quantum tunneling) skyrocketing. Clock frequencies stalled at ~3–5 GHz. Single-thread CPU performance plateaued. The only way to increase compute was **parallelism** (adding more execution units rather than clocking a single core faster).

On edge devices (e.g., smart glasses, drones, phones), the total thermal envelope is bounded by passive cooling:
$$\text{TDP (Thermal Design Power)} \le 2\text{W to } 5\text{W}$$
You cannot dump 150W of desktop power into a device resting on someone's face or running on a battery.

### The Energy Equation of Computation
Here is the single most critical realization for Edge AI engineers:

| Operation | Energy Cost (approximate, 7nm node) | Relative Cost |
| :--- | :--- | :--- |
| **8-bit Integer Add (INT8 ALU)** | ~0.03 pJ | 1x |
| **32-bit Floating Add (FP32 ALU)** | ~0.1 pJ | 3.3x |
| **32-bit Floating Multiply (FP32 ALU)** | ~1.0 pJ | 33x |
| **Read 32-bit from Register File** | ~0.1 pJ | 3.3x |
| **Read 32-bit from L1 SRAM Cache** | ~1.5 pJ | 50x |
| **Read 32-bit from L2 / L3 SRAM Cache** | ~5–20 pJ | 160–660x |
| **Read 32-bit across bus from DRAM (LPDDR4/5)** | **~1,000–2,000 pJ** | **33,000–66,000x** |

> 💥 **First Principle**:
> **Math is almost free. Data movement is where all your battery and thermal budget goes.**  
> If an algorithm forces 1 billion numbers to cross the memory bus, the chip will melt its thermal budget even if the math itself took 0.001 seconds.

---

### The Triad: CPU vs. GPU vs. NPU

Why three distinct processor types? Because they solve fundamentally different problems with different trade-offs in die area:

```text
       CPU                                 GPU                                 NPU
┌─────────────────┐                ┌─────────────────┐                ┌─────────────────┐
│ Control  │ Cache│                │Ctrl│    Cache    │                │Ctrl│  SRAM Buf  │
│ (Branch, │      │                ├──┬─┴─────────────┤                ├──┬─┴─────────────┤
│ Out-of-  │      │                │  │  ALU  ALU ALU │                │  │ ┌───────────┐ │
│ Order)   │      │                │  │  ALU  ALU ALU │                │  │ │ 2D Systolic│ │
├──────────┴──────┤                │  │  ALU  ALU ALU │                │  │ │ MAC Array │ │
│   Few Huge ALUs │                │  │  ALU  ALU ALU │                │  │ └───────────┘ │
└─────────────────┘                └──────────────────┘                └─────────────────┘
 ~15% Die = ALUs                    ~50% Die = ALUs                     ~85% Die = ALUs
 Latency-Optimized                  Throughput-Optimized                Dataflow / Energy-Optimized
```

#### 1. CPU (Central Processing Unit) — *The Master Coordinator*
- **Primary Goal**: Minimize latency on single-threaded, unpredictable, serial execution.
- **Hardware Architecture**:
  - Deep instruction pipelines (15–20+ stages).
  - Massive branch predictors (>95% accuracy) and speculative execution engines.
  - Out-of-order (OoO) execution logic to reorder instructions around stalled memory loads.
  - Huge multi-megabyte cache hierarchies (L1, L2, L3) to keep code and data close.
- **Why it struggles on AI**: An AI matrix multiplication has zero branch unpredictability. Spending 70% of silicon area on branch prediction and out-of-order reorder buffers is wasted silicon when doing billions of simple multiplications.

#### 2. GPU (Graphics Processing Unit) — *The Parallel Throughput Engine*
- **Primary Goal**: Maximize aggregate throughput for thousands of identical, data-parallel tasks (SIMT: Single Instruction, Multiple Threads).
- **Hardware Architecture**:
  - Thousands of small, lightweight ALUs grouped into Compute Units / Streaming Multiprocessors.
  - Very simple control logic (no complex out-of-order execution, minimal branch prediction).
  - Extreme memory bandwidth (hundreds of GB/s via wide 128/256/384-bit memory interfaces).
  - Latency hiding: When a thread stalls on memory, the hardware scheduler instantly context-switches to another thread group (warp/wavefront) at zero cycle cost.
- **Why it's not optimal for low-power edge**: GPUs are power-hungry (typically 15W–300W+). Even mobile GPUs draw significant power due to register file size, thread management overhead, and high-frequency memory bus operation.

#### 3. NPU (Neural Processing Unit) — *The Dedicated Tensor Engine*
- **Primary Goal**: Maximum energy efficiency (TOPS/Watt) and deterministic low latency for neural network tensor graphs (GEMM, Conv, Activation).
- **Hardware Architecture**:
  - **Spatial Architecture / Systolic Arrays**: 2D grid of multiply-accumulate (MAC) units connected directly to their immediate neighbors.
  - **Data Reuse**: Weights and activations flow through the array cell-by-cell. An operand loaded once from memory is reused across hundreds of MAC calculations before being discarded.
  - **Local SRAM Scratchpads**: Megabytes of high-speed on-chip memory directly coupled to the array, eliminating trips to DRAM.
  - **Minimal Control Overhead**: No branch prediction, no OoO execution, fixed-precision integer math (INT8/INT4/FP16).
- **Trade-off**: Inflexible. If an algorithm requires recursive trees, pointer-chasing, or dynamic branch execution, an NPU cannot run it.

---

## 3. Level 3: Implementation & Empirical Code

To internalize these architectures, we inspect how different workload structures stress hardware:
1. **Control-Heavy Workload**: Branching logic with data dependencies (CPU domain).
2. **Streaming Memory-Bound Workload**: Elementwise operations (Memory bus stress).
3. **Compute-Bound Dense Matrix Workload**: 2D GEMM (Tensor/Accelerator domain).

See the implementation in [`code/workload_characterization.py`](file:///Users/alwinglifferd/development/Hearsight%20Learning/01-Hardware/01-Computer-Architecture/code/workload_characterization.py).
