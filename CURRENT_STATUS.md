# CURRENT_STATUS.md

Current milestone: M2 (Corrective Documentation Synchronization Complete — Awaiting Technical-Lead Audit & Authorization — DO NOT PROCEED TO M3)
Current branch: m2/dp4a-microkernel
Accepted M1 base commit: 3ef60d1f11ba6d9e867b6108dad33d0c96d2195f
(Git repository HEAD is authoritative for current commit state)

Execution host: BigBoy
OS: Debian GNU/Linux 13 (trixie) (Kernel 6.12.100+deb13-amd64 x86_64)

Pascal qualification GPU: NVIDIA GeForce GTX 1060 6GB
Pascal PCI/UUID: PCI 0000:03:00.0 / UUID GPU-5078520c-bf92-4143-0d48-b02fd9d250dd
Pascal compute capability: 6.1 (NUMA node 0)

Control GPU: NVIDIA GeForce GTX 1660 SUPER
Control PCI/UUID: PCI 0000:81:00.0 / UUID GPU-8a043791-359d-1df4-9353-38c5bc98e5dd
Control compute capability: 7.5 (NUMA node 1)

NVIDIA driver: 550.163.01
CUDA toolkit: CUDA 12.9 (/usr/local/cuda-12.9)
NVCC: 12.9.86
Host compiler: gcc 14.2.0 (Debian 14.2.0-19)
glibc: Debian GLIBC 2.41-12+deb13u3 (2.41)

---

## Milestone M2 DP4A Microkernel Characterization Summary

- **Mathematical Contract**: $y[n] = \sum_{k=0}^{K-1} W[n, k] \cdot x[k]$ with row-major $W \in \text{int8}^{N \times K}$, contiguous $x \in \text{int8}^K$, and $y \in \text{int32}^N$.
- **Correctness**: 156 / 156 test configurations PASSED (100% bit-exact against CPU 64-bit accumulation reference oracle) across 6 data distributions, 14 $K$-edge cases, 14 $N$-boundary cases, and 5 full-scale Qwen3-4B projection layers on both physical GPUs.
- **SASS Verification (`sm_61`)**:
  - `gemv_scalar_kernel` (Variant A): **0** `IDP.4A` instructions, 58 `XMAD` scalar mul/mad operations (**Pure scalar non-DP4A comparator**).
  - `gemv_dp4a_basic_kernel` (Variant B): **43** `IDP.4A` instructions.
  - `gemv_dp4a_opt_kernel` (Variant C): **99** `IDP.4A` instructions.
  - `gemv_dp4a_shared_kernel` (Variant D): **111** `IDP.4A` instructions.
- **Hardware Resources & Theoretical Occupancy (CUDA Occupancy API)**:
  - Pascal (`sm_61`): Variants A, B, C achieve **100.0% theoretical occupancy** (64 resident warps/SM, $\le 32$ regs/thread); Variant D achieves **50.0% theoretical occupancy** (32 resident warps/SM, 4 blocks/SM, 55 regs/thread, 2048 B smem). Zero register spills across all variants.
  - Turing (`sm_75`): All variants achieve **100.0% theoretical occupancy** (32 resident warps/SM). Zero spills.

### Primary Qualification GPU (NVIDIA GeForce GTX 1060 6GB — Pascal SM 6.1):
- **`attn_q`** ($4096 \times 2560$):
  - Scalar Baseline: $198.66\ \mu\text{s}$ ($52.78\text{ GMAC/s}$, $49.25\text{ GiB/s}$)
  - Basic DP4A: $60.42\ \mu\text{s}$ ($173.56\text{ GMAC/s}$, $161.93\text{ GiB/s}$, **$3.29\times$ speedup**)
  - Optimized DP4A: $60.35\ \mu\text{s}$ ($173.74\text{ GMAC/s}$, $162.10\text{ GiB/s}$, **$3.29\times$ speedup**)
  - Shared DP4A: $65.30\ \mu\text{s}$ ($160.59\text{ GMAC/s}$, $149.83\text{ GiB/s}$, $3.04\times$ speedup)
- **`attn_kv`** ($1024 \times 2560$):
  - Scalar Baseline: $53.25\ \mu\text{s}$ ($49.23\text{ GMAC/s}$, $45.97\text{ GiB/s}$)
  - Basic DP4A: $19.42\ \mu\text{s}$ ($134.96\text{ GMAC/s}$, $126.01\text{ GiB/s}$, **$2.74\times$ speedup**)
  - Optimized DP4A: $20.48\ \mu\text{s}$ ($128.00\text{ GMAC/s}$, $119.51\text{ GiB/s}$, **$2.60\times$ speedup**)
  - Shared DP4A: $21.50\ \mu\text{s}$ ($121.90\text{ GMAC/s}$, $113.82\text{ GiB/s}$, $2.48\times$ speedup)
- **`attn_out`** ($2560 \times 4096$):
  - Scalar Baseline: $186.37\ \mu\text{s}$ ($56.26\text{ GMAC/s}$, $52.47\text{ GiB/s}$)
  - Basic DP4A: $61.44\ \mu\text{s}$ ($170.67\text{ GMAC/s}$, $159.16\text{ GiB/s}$, **$3.03\times$ speedup**)
  - Optimized DP4A: $60.42\ \mu\text{s}$ ($173.56\text{ GMAC/s}$, $161.86\text{ GiB/s}$, **$3.08\times$ speedup**)
  - Shared DP4A: $60.42\ \mu\text{s}$ ($173.56\text{ GMAC/s}$, $161.86\text{ GiB/s}$, **$3.08\times$ speedup**)
- **`ffn_gate_up`** ($9728 \times 2560$):
  - Scalar Baseline: $465.92\ \mu\text{s}$ ($53.45\text{ GMAC/s}$, $49.86\text{ GiB/s}$)
  - Basic DP4A: $135.17\ \mu\text{s}$ ($184.24\text{ GMAC/s}$, $171.87\text{ GiB/s}$, **$3.45\times$ speedup**)
  - Optimized DP4A: $131.90\ \mu\text{s}$ ($188.80\text{ GMAC/s}$, **$176.13\text{ GiB/s}$ effective bandwidth**, **$3.53\times$ speedup**)
  - Shared DP4A: $144.38\ \mu\text{s}$ ($172.48\text{ GMAC/s}$, $160.90\text{ GiB/s}$, $3.23\times$ speedup)
- **`ffn_down`** ($2560 \times 9728$):
  - Scalar Baseline: $491.46\ \mu\text{s}$ ($50.67\text{ GMAC/s}$, $47.23\text{ GiB/s}$)
  - Basic DP4A: $145.41\ \mu\text{s}$ ($171.27\text{ GMAC/s}$, $159.63\text{ GiB/s}$, **$3.38\times$ speedup**)
  - Optimized DP4A: $135.17\ \mu\text{s}$ ($184.24\text{ GMAC/s}$, **$171.73\text{ GiB/s}$ effective bandwidth**, **$3.64\times$ speedup**)
  - Shared DP4A: $141.31\ \mu\text{s}$ ($176.23\text{ GMAC/s}$, $164.26\text{ GiB/s}$, $3.48\times$ speedup)

### Secondary Control GPU (NVIDIA GeForce GTX 1660 SUPER — Turing SM 7.5):
- `attn_q`: Scalar $53.95\ \mu\text{s}$ vs Basic DP4A $36.91\ \mu\text{s}$ (**$1.46\times$ speedup**, $265.04\text{ GiB/s}$) vs Optimized DP4A $37.06\ \mu\text{s}$ (**$1.46\times$ speedup**, $264.01\text{ GiB/s}$)
- `attn_kv`: Scalar $18.00\ \mu\text{s}$ vs Basic DP4A $12.29\ \mu\text{s}$ (**$1.46\times$ speedup**, $199.19\text{ GiB/s}$) vs Optimized DP4A $12.29\ \mu\text{s}$ (**$1.46\times$ speedup**, $199.19\text{ GiB/s}$)
- `attn_out`: Scalar $55.73\ \mu\text{s}$ vs Basic DP4A $39.36\ \mu\text{s}$ (**$1.42\times$ speedup**, $248.45\text{ GiB/s}$) vs Optimized DP4A $39.12\ \mu\text{s}$ (**$1.42\times$ speedup**, $249.97\text{ GiB/s}$)
- `ffn_gate_up`: Scalar $120.32\ \mu\text{s}$ vs Basic DP4A $82.82\ \mu\text{s}$ (**$1.45\times$ speedup**, $280.52\text{ GiB/s}$) vs Optimized DP4A $84.75\ \mu\text{s}$ (**$1.42\times$ speedup**, $274.12\text{ GiB/s}$)
- `ffn_down`: Scalar $122.88\ \mu\text{s}$ vs Basic DP4A $82.43\ \mu\text{s}$ (**$1.49\times$ speedup**, $281.59\text{ GiB/s}$) vs Optimized DP4A $82.80\ \mu\text{s}$ (**$1.48\times$ speedup**, $280.34\text{ GiB/s}$)

---

Current meaningful benchmark: DP4A GEMV microbenchmark suite across Qwen3-4B transformer shapes.
Current blocker: None for M2 implementation.
Next required action: Technical-lead M2 final audit and authorization (DO NOT PROCEED TO M3).
