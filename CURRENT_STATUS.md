# CURRENT_STATUS.md

Current milestone: M2 (Implementation Complete — Awaiting Technical-Lead Audit & Authorization — DO NOT PROCEED TO M3)
Current branch: m2/dp4a-microkernel
Accepted M1 base commit: 3ef60d1f11ba6d9e867b6108dad33d0c96d2195f
(Git repository HEAD is authoritative for current commit state)

Execution host: BigBoy

Pascal qualification GPU: NVIDIA GeForce GTX 1060 6GB
Pascal PCI/UUID: PCI 0000:03:00.0 / UUID GPU-5078520c-bf92-4143-0d48-b02fd9d250dd
Pascal compute capability: 6.1 (NUMA node 0)

Control GPU: NVIDIA GeForce GTX 1660 SUPER
Control PCI/UUID: PCI 0000:81:00.0 / UUID GPU-8a043791-359d-1df4-9353-38c5bc98e5dd
Control compute capability: 7.5 (NUMA node 1)

NVIDIA driver: 550.120
CUDA toolkit: CUDA 12.9.2 (/usr/local/cuda-12.9)
NVCC: 12.9.86

---

## Milestone M2 DP4A Microkernel Characterization Summary

- **Mathematical Contract**: $y[n] = \sum_{k=0}^{K-1} W[n, k] \cdot x[k]$ with row-major $W \in \text{int8}^{N \times K}$, contiguous $x \in \text{int8}^K$, and $y \in \text{int32}^N$.
- **Correctness**: 156 / 156 test configurations PASSED (100% bit-exact against CPU 64-bit accumulation reference oracle) across 6 data distributions, 14 $K$-edge cases, 14 $N$-boundary cases, and 5 full-scale Qwen3-4B projection layers on both physical GPUs.
- **SASS Verification (`sm_61`)**:
  - `gemv_scalar_kernel` (Variant A): **0** `IDP.4A` instructions, 58 `XMAD` scalar mul/mad operations (**Pure scalar non-DP4A comparator**).
  - `gemv_dp4a_basic_kernel` (Variant B): **43** `IDP.4A` instructions.
  - `gemv_dp4a_opt_kernel` (Variant C): **99** `IDP.4A` instructions.
  - `gemv_dp4a_shared_kernel` (Variant D): **111** `IDP.4A` instructions.
- **Resource Usage**: Zero local memory spill stores / loads across all four kernels on both architectures.

### Primary Qualification GPU (NVIDIA GeForce GTX 1060 6GB — Pascal SM 6.1):
- **`attn_q`** ($4096 \times 2560$):
  - Scalar Baseline: $198.66\ \mu\text{s}$ ($52.78\text{ GMAC/s}$, $49.25\text{ GiB/s}$)
  - Basic DP4A: $60.42\ \mu\text{s}$ ($173.56\text{ GMAC/s}$, $161.93\text{ GiB/s}$, **$3.29\times$ speedup**)
  - Optimized DP4A: $60.42\ \mu\text{s}$ ($173.56\text{ GMAC/s}$, $161.93\text{ GiB/s}$, **$3.29\times$ speedup**)
  - Shared DP4A: $65.22\ \mu\text{s}$ ($160.79\text{ GMAC/s}$, $150.01\text{ GiB/s}$, $3.05\times$ speedup)
- **`ffn_gate_up`** ($9728 \times 2560$):
  - Scalar Baseline: $465.92\ \mu\text{s}$ ($53.45\text{ GMAC/s}$, $49.86\text{ GiB/s}$)
  - Basic DP4A: $134.14\ \mu\text{s}$ ($185.65\text{ GMAC/s}$, $173.19\text{ GiB/s}$, **$3.47\times$ speedup**)
  - Optimized DP4A: $130.74\ \mu\text{s}$ ($190.49\text{ GMAC/s}$, **$177.70\text{ GiB/s}$ / $99.38\%$ of theoretical DRAM peak**, **$3.56\times$ speedup**)
  - Shared DP4A: $140.29\ \mu\text{s}$ ($177.52\text{ GMAC/s}$, $165.60\text{ GiB/s}$, $3.32\times$ speedup)
- **`ffn_down`** ($2560 \times 9728$):
  - Scalar Baseline: $481.28\ \mu\text{s}$ ($51.74\text{ GMAC/s}$, $48.23\text{ GiB/s}$)
  - Basic DP4A: $142.30\ \mu\text{s}$ ($175.00\text{ GMAC/s}$, $163.12\text{ GiB/s}$, **$3.38\times$ speedup**)
  - Optimized DP4A: $135.12\ \mu\text{s}$ ($184.31\text{ GMAC/s}$, $171.79\text{ GiB/s}$, **$3.56\times$ speedup**)
  - Shared DP4A: $140.85\ \mu\text{s}$ ($176.81\text{ GMAC/s}$, $164.80\text{ GiB/s}$, $3.42\times$ speedup)

### Secondary Control GPU (NVIDIA GeForce GTX 1660 SUPER — Turing SM 7.5):
- `attn_q`: Scalar $56.67\ \mu\text{s}$ vs Optimized DP4A $39.28\ \mu\text{s}$ (**$1.44\times$ speedup**)
- `ffn_gate_up`: Scalar $120.67\ \mu\text{s}$ vs Optimized DP4A $84.37\ \mu\text{s}$ (**$1.43\times$ speedup**)
- `ffn_down`: Scalar $122.88\ \mu\text{s}$ vs Optimized DP4A $84.35\ \mu\text{s}$ (**$1.46\times$ speedup**)

---

Current meaningful benchmark: DP4A GEMV microbenchmark suite across Qwen3-4B transformer shapes.
Current blocker: None for M2 implementation.
Next required action: Technical-lead M2 audit and authorization (DO NOT PROCEED TO M3).
