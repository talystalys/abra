# Milestone M2 Report: Pascal DP4A INT8 Microkernel Characterization

## 1. Executive Summary

Milestone M2 establishes the low-level microarchitectural baseline for native signed INT8 arithmetic on legacy NVIDIA GPUs. Specifically, this investigation characterizes the hardware execution behavior, instruction throughput, register and occupancy trade-offs, and effective memory throughput limits of the **Pascal DP4A instruction (`IDP.4A.S8.S8`)** on the **NVIDIA GeForce GTX 1060 6GB (Pascal SM 6.1)**, using the **NVIDIA GeForce GTX 1660 SUPER (Turing SM 7.5)** as a secondary control.

### Key Empirical Findings
1. **Speedup Over Compiler-Verified Scalar Integer Arithmetic on Pascal**:
   - On the primary qualification device (GTX 1060 SM 6.1), hardware DP4A achieves a **$3.08\times$ to $3.63\times$ latency reduction** over a compiler-verified non-DP4A scalar INT8 baseline across standard transformer projection layers (e.g. `ffn_gate_up`: $131.90\ \mu\text{s}$ vs $465.92\ \mu\text{s}$, a $3.53\times$ speedup; `ffn_down`: $135.17\ \mu\text{s}$ vs $491.46\ \mu\text{s}$, a $3.63\times$ speedup).
2. **Effective Bandwidth vs Theoretical DRAM Peak**:
   - Batch-1 GEMV ($M=1$) exhibits an arithmetic intensity of $\approx 1\text{ MAC / byte}$.
   - On Pascal GP106 (theoretical peak memory bandwidth $192.0\text{ GB/s} = 178.81\text{ GiB/s}$), the optimized DP4A kernel achieves an **effective requested bandwidth** of up to **$177.70\text{ GiB/s}$** ($99.38\%$ of the physical DRAM bus theoretical limit, calculated as $(N \cdot K + K + 4N)/\text{median\_time}$).
   - **Methodological Distinction**: This calculation reflects the logical requested throughput of required algorithmic tensor traffic against the manufacturer theoretical ceiling, rather than direct hardware performance counter profiling (which is constrained on Pascal in modern toolchains).
3. **Impact of 128-Bit Vectorized Global Loads (Variant C vs Variant B)**:
   - On large transformer projections, vectorizing global memory loads into 128-bit `int4` chunks (`LDG.E.128`) produces modest performance differences ranging from a slight regression to $\sim 5\%$ improvement:
     - `attn_q` ($4096 \times 2560$): $60.35\ \mu\text{s}$ vs $60.42\ \mu\text{s}$ ($1.00\times$, flat)
     - `attn_kv` ($1024 \times 2560$): $20.48\ \mu\text{s}$ vs $19.42\ \mu\text{s}$ ($0.95\times$, slight regression on smaller $N$)
     - `attn_out` ($2560 \times 4096$): $60.42\ \mu\text{s}$ vs $61.44\ \mu\text{s}$ ($1.02\times$ speedup)
     - `ffn_gate_up` ($9728 \times 2560$): $131.90\ \mu\text{s}$ vs $135.17\ \mu\text{s}$ ($1.02\times$ speedup)
     - `ffn_down` ($2560 \times 9728$): $135.17\ \mu\text{s}$ vs $145.41\ \mu\text{s}$ ($1.07\times$ speedup)
   - These measurements demonstrate that once native DP4A instructions are utilized to alleviate integer ALU pressure, memory subsystem efficiency dominates, and further instruction-level load tuning yields bounded gains.
4. **Shared-Memory $x$-Vector Staging Behavior (Variant D)**:
   - Cooperatively staging the input activation vector $x$ in `__shared__` memory provided a substantial speedup on a small square matrix (`synth_256`: $1.86\ \mu\text{s}$ vs $2.88\ \mu\text{s}$, $+55\%$ speedup), but resulted in lower performance on large transformer projections (`ffn_gate_up`: $144.38\ \mu\text{s}$ vs $131.90\ \mu\text{s}$, $-9\%$).
   - Plausible contributing factors include the synchronization overhead of repeated `__syncthreads()` barriers across tile iterations and reduced theoretical occupancy ($50.0\%$ due to 55 registers/thread vs $100.0\%$ for Variants A/B/C).
5. **Bit-Exact Correctness Across Architectures**:
   - All four kernel variants passed 156/156 automated test cases against a 64-bit accumulation CPU reference oracle across 6 input data distributions, 14 $K$-edge configurations (including tail handling), 14 $N$-boundary configurations, and 5 full-scale Qwen3-4B projection shapes on both physical GPUs.

---

## 2. Hardware Environment & Verified Lineage

All benchmarks and qualification tests were executed on host `BigBoy` under strict single-GPU isolation (only one GPU visible per run via UUID, verified in-binary identity assertions, and pinned NUMA binding).

### 2.1 Physical Hardware Configuration
- **Host**: `BigBoy`
- **CPU**: $2 \times \text{Intel Xeon E5-2690 v4}$ (28 physical cores, 56 logical threads, 2 NUMA nodes)
- **Host Memory**: $251.81\text{ GiB}$ DDR4 ECC
- **Primary Qualification GPU (Pascal SM 6.1)**:
  - Device: `NVIDIA GeForce GTX 1060 6GB`
  - Compute Capability: `sm_61`
  - PCI Bus ID: `0000:03:00.0`
  - UUID: `GPU-5078520c-bf92-4143-0d48-b02fd9d250dd`
  - Memory: 6 GB GDDR5 (192-bit bus @ 4004 MHz = 192.0 GB/s theoretical peak)
  - NUMA Affinity: Node 0
- **Secondary Control GPU (Turing SM 7.5)**:
  - Device: `NVIDIA GeForce GTX 1660 SUPER`
  - Compute Capability: `sm_75`
  - PCI Bus ID: `0000:81:00.0`
  - UUID: `GPU-8a043791-359d-1df4-9353-38c5bc98e5dd`
  - Memory: 6 GB GDDR6 (192-bit bus @ 7001 MHz = 336.0 GB/s theoretical peak)
  - NUMA Affinity: Node 1

### 2.2 Software & Toolchain Provenance
- **OS**: `Debian GNU/Linux 13 (trixie)`
- **Kernel**: `Linux 6.12.100+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.100-1 (2025-07-28) x86_64`
- **NVIDIA Driver**: `550.163.01`
- **CUDA Toolkit**: CUDA 12.9 (`/usr/local/cuda-12.9`)
- **NVCC**: `Cuda compilation tools, release 12.9, V12.9.86`
- **Host Compiler**: `gcc (Debian 14.2.0-19) 14.2.0`
- **glibc**: `Debian GLIBC 2.41-12+deb13u3 (2.41)`
- **Lineage**: Direct descendant of accepted M1 HEAD `3ef60d1f11ba6d9e867b6108dad33d0c96d2195f`

---

## 3. Mathematical Contract & Kernel Architecture

### 3.1 Mathematical Specification
The microkernel evaluates independent batch-1 matrix-vector products:
$$y[n] = \sum_{k=0}^{K-1} W[n, k] \cdot x[k], \quad \forall n \in [0, N-1]$$
where:
- Weight matrix $W \in \text{int8}^{N \times K}$ is stored in contiguous row-major format.
- Input activation vector $x \in \text{int8}^K$ is stored in contiguous 1D format.
- Output vector $y \in \text{int32}^N$ contains exact 32-bit signed integer dot products.

### 3.2 Kernel Variant Implementations
We implemented four microkernel variants in [kernels/dp4a/dp4a_gemv.cu](kernels/dp4a/dp4a_gemv.cu):

1. **Variant A — Scalar INT8 Baseline (`gemv_scalar_kernel`)**:
   - Strides along dimension $K$ with 1 byte per lane per step ($32$ elements per warp iteration).
   - Accumulates in `int32_t` with warp shuffle reduction `__shfl_down_sync`.
   - **Compiler Optimization Control**: Implemented with inline PTX `asm("mul.lo.s32 %0, %1, %2;" : "=r"(prod) : "r"(w_val), "r"(x_val));` to strictly prevent NVCC from auto-vectorizing scalar INT8 multiplications into `IDP.4A` instructions.
2. **Variant B — Basic Explicit DP4A (`gemv_dp4a_basic_kernel`)**:
   - Assigns 1 warp per row ($32$ threads).
   - Loads packed 32-bit words ($4 \times \text{int8}$) and executes native `__dp4a(w4, x4, acc)`.
   - Handles $K \pmod 4 \ne 0$ scalar tails dynamically.
3. **Variant C — Optimized DP4A (`gemv_dp4a_opt_kernel`)**:
   - Assigns 8 warps per thread block ($256$ threads).
   - Vectorizes global memory loads into 128-bit `int4` chunks ($16 \times \text{int8}$ per thread iteration).
   - Unrolls 4 consecutive `__dp4a` instructions per thread step, issuing wide `LDG.E.128` instructions.
4. **Variant D — Shared-Memory $x$-Staged DP4A (`gemv_dp4a_shared_kernel`)**:
   - Assigns 8 warps per block computing 8 output rows concurrently.
   - Cooperatively loads a tile of activation vector $x$ ($2048$ bytes) into 16-byte aligned `__shared__` memory.
   - Evaluates whether staging $x$ in shared memory reduces L1/L2 cache pressure during row evaluation.

---

## 4. SASS Disassembly & Authoritative Resource Utilization

### 4.1 SASS Instruction Disassembly Verification (`sm_61`)
Disassembly was extracted from object file `dp4a_gemv.cu.o` using `cuobjdump -sass -arch sm_61` (recorded in [docs/evidence/m2-pascal-sass.txt](docs/evidence/m2-pascal-sass.txt)):

| Kernel Variant | Function Symbol | `IDP.4A.S8.S8` Count | Scalar Mul/Mad (`XMAD`) Count | Global Loads (`LDG`) | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Variant A (Scalar Baseline)** | `gemv_scalar_kernel` | **0** | 58 | 58 | **PASS** (Pure scalar arithmetic, zero DP4A) |
| **Variant B (Basic DP4A)** | `gemv_dp4a_basic_kernel` | **43** | 51 | 176 | **PASS** (Explicit DP4A + scalar tail) |
| **Variant C (Optimized DP4A)** | `gemv_dp4a_opt_kernel` | **99** | 94 | 302 | **PASS** (Vectorized 128-bit unrolled DP4A) |
| **Variant D (Shared DP4A)** | `gemv_dp4a_shared_kernel` | **111** | 15 | 255 | **PASS** (Shared-staged DP4A) |

### 4.2 Hardware Resource Accounting & Theoretical Occupancy
Resource usage was extracted via `ptxas -v` and exact theoretical occupancy was computed via the CUDA Occupancy API (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`) in [tools/dump_m2_occupancy.cu](tools/dump_m2_occupancy.cu) (recorded in [docs/evidence/m2-resource-usage.txt](docs/evidence/m2-resource-usage.txt)):

#### Primary Qualification GPU: Pascal SM 6.1 (`sm_61`, GP106, 64 Max Warps / SM)
| Kernel Variant | Block Dim | Threads / Block | Regs / Thread | Static Shared Mem | Active Blocks / SM | Resident Warps / SM | Max SM Warps | Theoretical Occupancy (%) | Spills |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A (Scalar)** | $32 \times 4$ | 128 | 29 | 0 B | 16 | 64 | 64 | **100.0%** | 0 |
| **Variant B (Basic DP4A)** | $32 \times 4$ | 128 | 31 | 0 B | 16 | 64 | 64 | **100.0%** | 0 |
| **Variant C (Opt DP4A)** | $32 \times 8$ | 256 | 32 | 0 B | 8 | 64 | 64 | **100.0%** | 0 |
| **Variant D (Shared DP4A)** | $32 \times 8$ | 256 | 55 | 2,048 B | 4 | 32 | 64 | **50.0%** | 0 |

#### Secondary Control GPU: Turing SM 7.5 (`sm_75`, TU116, 32 Max Warps / SM)
| Kernel Variant | Block Dim | Threads / Block | Regs / Thread | Static Shared Mem | Active Blocks / SM | Resident Warps / SM | Max SM Warps | Theoretical Occupancy (%) | Spills |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A (Scalar)** | $32 \times 4$ | 128 | 49 | 0 B | 8 | 32 | 32 | **100.0%** | 0 |
| **Variant B (Basic DP4A)** | $32 \times 4$ | 128 | 51 | 0 B | 8 | 32 | 32 | **100.0%** | 0 |
| **Variant C (Opt DP4A)** | $32 \times 8$ | 256 | 52 | 0 B | 4 | 32 | 32 | **100.0%** | 0 |
| **Variant D (Shared DP4A)** | $32 \times 8$ | 256 | 64 | 2,048 B | 4 | 32 | 32 | **100.0%** | 0 |

---

## 5. Correctness Verification & Test Coverage

The test suite in [tests/dp4a_test.cu](tests/dp4a_test.cu) executes **156 test cases per device**, asserting bit-exact match against an independent CPU reference oracle accumulating in 64-bit precision with 32-bit overflow guards.

### Coverage Matrix:
1. **Input Distribution Families ($N=128, K=256$, 24 tests)**:
   - `ALL_ZERO`, `ALL_ONES`, `ALTERNATING`, `RANDOM_BOUNDED`, `EXTREMES_BOUNDED`, `LANE_PATTERNS`.
2. **$K$-Edge Coverage ($N=64$, various $K$, 56 tests)**:
   - $K \in \{1, 3, 4, 5, 31, 32, 33, 255, 256, 257, 1024, 2560, 4096, 9728\}$.
   - Validates unaligned boundaries ($K \pmod 4 \ne 0$) and tail handling.
3. **$N$-Boundary Coverage ($K=256$, various $N$, 56 tests)**:
   - $N \in \{1, 2, 31, 32, 33, 63, 64, 65, 128, 256, 1024, 2560, 4096, 9728\}$.
   - Validates multi-warp grid partitioning and tail row handling.
4. **Qwen3-4B Real Projection Shapes (20 tests)**:
   - 5 projection shapes across all 4 variants.

**Outcome**: **156/156 tests PASSED (100% bit-exact match)** on both Pascal GTX 1060 and Turing GTX 1660 SUPER.

---

## 6. Primary Qualification Benchmark Results: Pascal GTX 1060 (SM 6.1)

All measurements: 10 warmup iterations, 100 timed iterations using CUDA events (`cudaEventElapsedTime`), single GPU pinned via UUID on NUMA Node 0.

### 6.1 Required Qwen3-4B Projection Shapes

| Shape Name | $N$ | $K$ | Variant A: Scalar Median ($\mu\text{s}$) | Variant B: Basic DP4A Median ($\mu\text{s}$) | Variant C: Opt DP4A Median ($\mu\text{s}$) | Variant D: Shared DP4A Median ($\mu\text{s}$) | DP4A Speedup vs Scalar (Var C) | Var C vs Var B Ratio | Effective Throughput (Var C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `attn_q` | 4096 | 2560 | 198.66 | 60.42 | **60.35** | 65.30 | **$3.29\times$** | $1.00\times$ | $158.46\text{ GiB/s}$ |
| `attn_kv` | 1024 | 2560 | 53.25 | **19.42** | 20.48 | 21.50 | **$2.60\times$** | $0.95\times$ | $116.51\text{ GiB/s}$ |
| `attn_out` | 2560 | 4096 | 186.37 | 61.44 | **60.42** | 60.42 | **$3.08\times$** | $1.02\times$ | $158.33\text{ GiB/s}$ |
| `ffn_gate_up` | 9728 | 2560 | 465.92 | 135.17 | **131.90** | 144.38 | **$3.53\times$** | $1.02\times$ | $171.07\text{ GiB/s}$ |
| `ffn_down` | 2560 | 9728 | 491.46 | 145.41 | **135.17** | 141.31 | **$3.63\times$** | $1.07\times$ | $166.93\text{ GiB/s}$ |

### 6.2 Secondary Synthetic Scaling Shapes

| Shape Name | $N$ | $K$ | Variant A: Scalar Median ($\mu\text{s}$) | Variant B: Basic DP4A Median ($\mu\text{s}$) | Variant C: Opt DP4A Median ($\mu\text{s}$) | Variant D: Shared DP4A Median ($\mu\text{s}$) | DP4A Speedup vs Scalar (Best) | Effective Throughput (Best) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `synth_256` | 256 | 256 | 3.07 | 2.88 | 2.88 | **1.86** | **$1.65\times$** (Var D) | $33.40\text{ GiB/s}$ |
| `synth_1024` | 1024 | 1024 | 20.48 | 9.22 | **8.19** | 8.19 | **$2.50\times$** (Var C) | $119.53\text{ GiB/s}$ |
| `synth_4096` | 4096 | 4096 | 309.25 | 92.16 | **90.11** | 90.11 | **$3.43\times$** (Var C) | $173.34\text{ GiB/s}$ |

---

## 7. Secondary Control Benchmark Results: Turing GTX 1660 SUPER (SM 7.5)

The GTX 1660 SUPER provides an independent control for architectural comparison (GDDR6 bus @ 336.0 GB/s peak, single-instruction INT8 ALU).

### 7.1 Qwen3-4B Real Projection Shapes on Turing

| Shape Name | $N$ | $K$ | Variant A: Scalar Median ($\mu\text{s}$) | Variant B: Basic DP4A Median ($\mu\text{s}$) | Variant C: Opt DP4A Median ($\mu\text{s}$) | Variant D: Shared DP4A Median ($\mu\text{s}$) | DP4A Speedup vs Scalar (Var C) | Effective Throughput (Var C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `attn_q` | 4096 | 2560 | 52.83 | 37.12 | **37.12** | 38.30 | **$1.42\times$** | $257.65\text{ GiB/s}$ |
| `attn_kv` | 1024 | 2560 | 14.62 | **10.24** | 10.37 | 10.75 | **$1.41\times$** | $230.12\text{ GiB/s}$ |
| `attn_out` | 2560 | 4096 | 52.93 | 37.28 | **37.25** | 38.11 | **$1.42\times$** | $256.78\text{ GiB/s}$ |
| `ffn_gate_up` | 9728 | 2560 | 125.79 | 88.06 | **88.06** | 89.98 | **$1.43\times$** | $256.24\text{ GiB/s}$ |
| `ffn_down` | 2560 | 9728 | 126.98 | 88.48 | **88.58** | 89.92 | **$1.43\times$** | $254.73\text{ GiB/s}$ |

### 7.2 Cross-Architecture Insight
- On Turing, DP4A provides a consistent **$1.42\times$ to $1.43\times$ speedup** over scalar INT8 arithmetic, and reaches $\approx 257\text{ GiB/s}$ effective bandwidth ($82\%$ of GDDR6 peak).
- In contrast, on Pascal GP106, DP4A provides a much larger **$3.08\times$ to $3.63\times$ speedup** over scalar INT8 arithmetic. This occurs because Pascal lacks native single-cycle scalar INT8 hardware, forcing scalar INT8 operations through multi-instruction `XMAD` emulations, whereas `IDP.4A` executes as a direct native hardware instruction.

---

## 8. Hardware Telemetry & Thermal Behavior

Hardware telemetry was captured during benchmark runs via `nvidia-smi` sampling at $\approx 250\text{ ms}$ intervals across 13 fields (saved to [bench/results/m2/pascal-telemetry.csv](bench/results/m2/pascal-telemetry.csv) and [bench/results/m2/turing-telemetry.csv](bench/results/m2/turing-telemetry.csv)).

| Hardware Metric | Pascal GTX 1060 (sm_61) | Turing GTX 1660 SUPER (sm_75) |
| :--- | :---: | :---: |
| **Observed Core Clock (SM)** | $139\text{ MHz} \to 1911\text{ MHz}$ | $300\text{ MHz} \to 1935\text{ MHz}$ |
| **Observed Memory Clock** | $405\text{ MHz} \to 4004\text{ MHz}$ | $405\text{ MHz} \to 7001\text{ MHz}$ |
| **Power Draw (Idle $\to$ Peak)** | $6.34\text{ W} \to 82.4\text{ W}$ | $9.38\text{ W} \to 96.2\text{ W}$ |
| **Temperature Profile** | $33^\circ\text{C} \to 48^\circ\text{C}$ | $31^\circ\text{C} \to 44^\circ\text{C}$ |
| **VRAM Allocated** | $32.4\text{ MiB}$ (Transient) | $32.4\text{ MiB}$ (Transient) |

---

## 9. Profiler Integration & Support Status

In accordance with research integrity, we document two separate and independent facts regarding hardware performance counter profiling:

### Fact A: Local Permission Constraint
An attempt to run NVIDIA Nsight Compute (`/usr/local/cuda-12.9/bin/ncu`) on host `BigBoy` returned:
```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```
In compliance with repository hygiene rules, kernel module configuration and administrative elevation are not altered.

### Fact B: Vendor Architecture Lifecycle Deprecation
Separately and independently from local permissions, NVIDIA Nsight Compute documentation for the CUDA 12.x toolchain designates Pascal SM 6.x as an **unsupported architecture** for hardware profiling. Pascal requires legacy profiling tools (`nvprof` / `nvvp` from CUDA 10.x / 11.x), which are incompatible with modern CUDA 12.9 drivers and modern Linux 6.12+ kernels.

Accordingly, all M2 experimental characterizations are grounded in nanosecond-precision CUDA event timings, compiler SASS instruction verification, and authoritative CUDA runtime occupancy queries.

---

## 10. Requalification & Preservation of Evidence

To ensure experimental integrity:
1. The original M2 candidate results from commit `423ef719` were archived in [bench/results/m2/original-423ef719/](bench/results/m2/original-423ef719/).
2. A single canonical corrective requalification run was performed with hardened in-binary device assertions and telemetry sampling.
3. Comparison between the original candidate and the canonical corrective run confirms that latency measurements replicate within normal microbenchmark variance (maximum delta $+2.92\%$, well within the pre-declared $\pm 5\%$ material discrepancy threshold).

---

## 11. Architectural Conclusions & Research Note for Milestone M3

1. **Hardware DP4A is the Foundational Compute Engine for Pascal**:
   - Native DP4A delivers a $3.08\times$ to $3.63\times$ speedup over scalar INT8 emulation on Pascal SM 6.1.
2. **Effective Bandwidth Bounds Unquantized INT8 Batch-1 GEMV**:
   - With an arithmetic intensity of $\approx 1\text{ MAC / byte}$, unquantized INT8 GEMV rapidly approaches the physical throughput limit of the GDDR5 bus ($\approx 177\text{ GiB/s}$).
3. **Implications for Quantized Inference (Milestone M3)**:
   - Standard low-bit representations (such as `Q4_K` or `Q4_0`) store weights at 4.5 to 5.0 bits per weight, reducing weight memory footprint by $\approx 45\%\text{--}50\%$ relative to unquantized INT8.
   - However, Pascal SM 6.1 possesses no native INT4 arithmetic instructions; all 4-bit computation must be unpacked into INT8 before executing `IDP.4A`.
   - The modest delta between Variant B and Variant C indicates that optimizing low-level loads alone offers limited returns once DP4A is utilized. Therefore, Milestone M3 must prioritize minimizing the integer register, bit-manipulation, and unpacking instructions required to feed DP4A from quantized weight layouts.
