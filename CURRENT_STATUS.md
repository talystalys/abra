# CURRENT_STATUS.md

Current milestone: M4 (Pascal Autotuned Kernel Family — Complete — Awaiting Technical-Lead Audit & Authorization — DO NOT PROCEED TO M5)
Current branch: m4/autotuned-kernel-family
Lineage: Direct descendant of verified M3 HEAD fb1d5f26162cf659ddc47b5752c257dff0bfa43a and accepted M2 base f10ead5fa94925e0a36980b1f3bdaa4a6998aa84
(Git repository HEAD is authoritative for current commit state)

Execution host: BigBoy
OS: Debian GNU/Linux 13 (trixie) (Kernel Linux 6.12.100+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.100-1 (2026-07-30) x86_64)
CPU: 2 x Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz (32 physical cores, 64 logical threads) (2 NUMA nodes)
RAM: 62.77 GiB Total (65,821,544 KiB)

Pascal qualification GPU: NVIDIA GeForce GTX 1060 6GB
Pascal PCI/UUID: PCI 0000:03:00.0 / UUID GPU-5078520c-bf92-4143-0d48-b02fd9d250dd
Pascal compute capability: 6.1 (NUMA node 0)

Control GPU: NVIDIA GeForce GTX 1660 SUPER
Control PCI/UUID: PCI 0000:81:00.0 / UUID GPU-8a043791-359d-1df4-9353-38c5bc98e5dd
Control compute capability: 7.5 (NUMA node 1)

NVIDIA driver: 550.163.01
CUDA toolkit: CUDA 12.9 (/usr/local/cuda-12.9)
NVCC: nvcc: NVIDIA (R) Cuda compiler driver (release 12.9, V12.9.86)
Host compiler: gcc (Debian 14.2.0-19) 14.2.0
glibc: ldd (Debian GLIBC 2.41-12+deb13u3) 2.41

---

## Milestone M4 Pascal Autotuned Kernel Family Summary

- **Kernel Architecture**: Templated, parameterized C++ CUDA kernel family `q4k_q8_mmvq_family_kernel<WARPS_PER_ROW, WARPS_PER_BLOCK, UNROLL>` spanning 17 candidate variants (`WPR ∈ {1, 2, 4}`, `WPB ∈ {1, 2, 4, 8, 16}`, `UNROLL ∈ {1, 2}`, reduction methods `warp_shuffle_zero_smem` and `inter_warp_smem`).
- **Deadlock-Free Row Predication**: Non-boundary rows participate in `__syncthreads()` unconditionally, ensuring zero thread divergence deadlocks on arbitrary $N$ (verified on $N=33, N=1$).
- **Theoretical Occupancy & Resource Allocation**:
  - Pascal (`sm_61`): $WPR=1$ candidates require 32 registers, 0 B shared memory, reaching 100.0% theoretical occupancy (64 resident warps/SM). $WPR=2, 4$ candidates require 40 registers, 8–64 B shared memory, reaching 75.0% theoretical occupancy (48 resident warps/SM). ZERO register spills across all candidates.
  - Turing (`sm_75`): All candidates achieve 100.0% theoretical occupancy (32 resident warps/SM). ZERO spills.
- **SASS Disassembly (`sm_61`)**: Verified direct emission of hardware `IDP.4A.S8.S8` for dot-product arithmetic.
- **Correctness Suite**: 756 / 756 test cases PASSED cleanly on Pascal GP106 and Turing TU116 (synthetic distributions, $K$-edge, $N$-boundary, all Qwen transformer projections, and real model bytes from `Qwen3-4B-Q4_K_M.gguf`).
- **Canonical Model Aggregate Performance (216 $Q4\_K$ Tensors in Qwen3-4B)**:
  - Total Weighted Upstream MMVQ Latency: **$17,515.01\ \mu\text{s}$**
  - Total Weighted M4 Autotuned Latency: **$15,903.94\ \mu\text{s}$**
  - Aggregate Speedup vs Pinned Upstream: **$1.101\times$** (**$+9.2\%$ latency reduction**).
  - Mode 2 Combined Pipeline Speedup: **$1.100\times$** ($18,228.63\ \mu\text{s} \to 16,575.48\ \mu\text{s}$).

### Pascal SM 6.1 (GTX 1060 6GB) Shape-Level Results (100 Samples):
- **`attn_q`** ($4096 \times 2560$, weight 36): Upstream $61.18\ \mu\text{s}$ $\to$ Autotuned $54.94\ \mu\text{s}$ (**$1.114\times$ speedup**, $100.30\text{ GiB/s}$)
- **`attn_kv`** ($1024 \times 2560$, weight 54): Upstream $18.24\ \mu\text{s}$ $\to$ Autotuned $13.31\ \mu\text{s}$ (**$1.370\times$ speedup**, $103.65\text{ GiB/s}$)
- **`attn_out`** ($2560 \times 4096$, weight 36): Upstream $59.39\ \mu\text{s}$ $\to$ Autotuned $55.92\ \mu\text{s}$ (**$1.062\times$ speedup**, $98.48\text{ GiB/s}$)
- **`ffn_gate_up`** ($9728 \times 2560$, weight 72): Upstream $136.02\ \mu\text{s}$ $\to$ Autotuned $123.73\ \mu\text{s}$ (**$1.099\times$ speedup**, $105.76\text{ GiB/s}$)
- **`ffn_down`** ($2560 \times 9728$, weight 18): Upstream $133.12\ \mu\text{s}$ $\to$ Autotuned $126.98\ \mu\text{s}$ (**$1.048\times$ speedup**, $102.90\text{ GiB/s}$)

### Turing SM 7.5 (GTX 1660 SUPER) Control Evaluation (100 Samples):
- `attn_q`: Upstream $35.07\ \mu\text{s}$ $\to$ Autotuned $33.84\ \mu\text{s}$ ($1.036\times$)
- `attn_kv`: Upstream $12.29\ \mu\text{s}$ $\to$ Autotuned $11.30\ \mu\text{s}$ ($1.088\times$)
- `attn_out`: Upstream $33.46\ \mu\text{s}$ $\to$ Autotuned $32.08\ \mu\text{s}$ ($1.043\times$)
- `ffn_gate_up`: Upstream $76.78\ \mu\text{s}$ $\to$ Autotuned $72.08\ \mu\text{s}$ ($1.065\times$)
- `ffn_down`: Upstream $72.80\ \mu\text{s}$ $\to$ Autotuned $72.80\ \mu\text{s}$ ($1.000\times$)

---

Current meaningful benchmark: Q4_K x Q8_1 MMVQ autotuned kernel family across Qwen3-4B transformer shapes.
Current blocker: None for M4 implementation.
Next required action: Technical-lead M4 final audit and authorization (DO NOT PROCEED TO M5).
