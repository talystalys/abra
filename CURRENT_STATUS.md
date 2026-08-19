# CURRENT_STATUS.md

Current milestone: M3 (Pascal Q4_K x Q8_1 Decode Microkernel Characterization — Complete — Awaiting Technical-Lead Audit & Authorization — DO NOT PROCEED TO M4)
Current branch: m3/q4k-q8-microkernel
Accepted M2 base commit: f10ead5fa94925e0a36980b1f3bdaa4a6998aa84
(Git repository HEAD is authoritative for current commit state)

Execution host: BigBoy
OS: Debian GNU/Linux 13 (trixie) (Kernel Linux 6.12.100+bpo-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.100~bpo12+1 (2025-05-24) x86_64)
CPU: 2 x Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz (32 physical cores, 64 logical threads, 2 NUMA nodes)
RAM: 62.77 GiB Total (65,820,952 KiB)

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
glibc: Debian GLIBC 2.41-3 (2.41)

---

## Milestone M3 Q4_K x Q8_1 MMVQ Decode Microkernel Characterization Summary

- **Mathematical Contract**: $y[n] = \sum_{k=0}^{K-1} \text{dequantize}(W[n, k]) \cdot \text{dequantize}(x[k])$ with $W \in \text{block\_q4\_K}^{N \times (K/256)}$ (144 bytes per 256 weights = 4.5 bpw), $x \in \text{block\_q8\_1}^{K/32}$ (36 bytes per 32 activations), and $y \in \text{float32}^N$.
- **Correctness**: 114 / 114 test configurations PASSED (100% agreement with independent double-precision $F64$ CPU oracle within max observed relative error of $4.20 \times 10^{-6}$) across 8 input distributions, 10 $K$-edge cases, 15 $N$-boundary cases, and 5 full-scale Qwen3-4B projection layers on both physical GPUs.
- **SASS Verification (`sm_61`)**: Confirmed native `IDP.4A.S8.S8` instructions in custom kernel SASS disassembly (`docs/evidence/m3-pascal-q4k-sass.txt`).
- **Hardware Resources & Theoretical Occupancy (CUDA Occupancy API)**:
  - Pascal (`sm_61`): Variant A (4-warp): 32 regs, 16 B smem, 100.0% occupancy; Variant B (2-warp): 40 regs, 16 B smem, 75.0% occupancy; Variant C (1-warp): 32 regs, 0 B smem, 100.0% occupancy. Zero register spills across all variants.
  - Turing (`sm_75`): All variants achieve 100.0% theoretical occupancy (32 resident warps/SM). Zero spills.

### Primary Qualification GPU (NVIDIA GeForce GTX 1060 6GB — Pascal SM 6.1):
- **`attn_q`** ($4096 \times 2560$):
  - Upstream MMVQ (4-warp Generic): $91.10\ \mu\text{s}$ ($115.00\text{ GMAC/s}$, $60.50\text{ GiB/s}$)
  - Custom 4-Warp Control: $91.10\ \mu\text{s}$ ($115.00\text{ GMAC/s}$, $60.50\text{ GiB/s}$, **$1.000\times$ speedup**)
  - Custom 2-Warp Split: $63.50\ \mu\text{s}$ ($165.00\text{ GMAC/s}$, $86.80\text{ GiB/s}$, **$1.435\times$ speedup**, **+43.5%**)
  - Custom 1-Warp Row: $66.60\ \mu\text{s}$ ($157.00\text{ GMAC/s}$, $82.80\text{ GiB/s}$, **$1.368\times$ speedup**, **+36.8%**)
- **`attn_kv`** ($1024 \times 2560$):
  - Upstream MMVQ: $21.00\ \mu\text{s}$ ($125.00\text{ GMAC/s}$, $65.70\text{ GiB/s}$)
  - Custom 2-Warp Split: $14.30\ \mu\text{s}$ ($183.00\text{ GMAC/s}$, $96.30\text{ GiB/s}$, **$1.469\times$ speedup**, **+46.9%**)
  - Custom 1-Warp Row: $14.30\ \mu\text{s}$ ($183.00\text{ GMAC/s}$, $96.30\text{ GiB/s}$, **$1.469\times$ speedup**, **+46.9%**)
- **`attn_out`** ($2560 \times 4096$):
  - Upstream MMVQ: $67.60\ \mu\text{s}$ ($155.00\text{ GMAC/s}$, $81.40\text{ GiB/s}$)
  - Custom 2-Warp Split: $60.40\ \mu\text{s}$ ($173.00\text{ GMAC/s}$, $91.10\text{ GiB/s}$, **$1.119\times$ speedup**, **+11.9%**)
  - Custom 1-Warp Row: $63.50\ \mu\text{s}$ ($165.00\text{ GMAC/s}$, $86.60\text{ GiB/s}$, **$1.065\times$ speedup**, **+6.5%**)
- **`ffn_gate_up`** ($9728 \times 2560$):
  - Upstream MMVQ: $196.00\ \mu\text{s}$ ($127.00\text{ GMAC/s}$, $66.90\text{ GiB/s}$)
  - Custom 2-Warp Split: $132.00\ \mu\text{s}$ ($189.00\text{ GMAC/s}$, $99.10\text{ GiB/s}$, **$1.485\times$ speedup**, **+48.5%**)
  - Custom 1-Warp Row: $134.00\ \mu\text{s}$ ($186.00\text{ GMAC/s}$, $97.90\text{ GiB/s}$, **$1.463\times$ speedup**, **+46.3%**)
- **`ffn_down`** ($2560 \times 9728$):
  - Upstream MMVQ: $136.00\ \mu\text{s}$ ($183.00\text{ GMAC/s}$, $95.90\text{ GiB/s}$)
  - Custom 2-Warp Split: $133.00\ \mu\text{s}$ ($187.00\text{ GMAC/s}$, $98.30\text{ GiB/s}$, **$1.023\times$ speedup**, **+2.3%**)
  - Custom 1-Warp Row: $132.00\ \mu\text{s}$ ($189.00\text{ GMAC/s}$, $98.90\text{ GiB/s}$, **$1.030\times$ speedup**, **+3.0%**)

### Secondary Control GPU (NVIDIA GeForce GTX 1660 SUPER — Turing SM 7.5):
- `attn_q`: Upstream MMVQ $43.00\ \mu\text{s}$ vs Custom 2-Warp $36.10\ \mu\text{s}$ (**$1.191\times$ speedup**) vs Custom 1-Warp $34.50\ \mu\text{s}$ (**$1.246\times$ speedup**)
- `attn_kv`: Upstream MMVQ $13.40\ \mu\text{s}$ vs Custom 2-Warp $12.30\ \mu\text{s}$ (**$1.089\times$ speedup**) vs Custom 1-Warp $12.10\ \mu\text{s}$ (**$1.107\times$ speedup**)
- `attn_out`: Upstream MMVQ $37.20\ \mu\text{s}$ vs Custom 2-Warp $35.10\ \mu\text{s}$ (**$1.060\times$ speedup**) vs Custom 1-Warp $34.80\ \mu\text{s}$ (**$1.069\times$ speedup**)
- `ffn_gate_up`: Upstream MMVQ $94.80\ \mu\text{s}$ vs Custom 2-Warp $78.10\ \mu\text{s}$ (**$1.214\times$ speedup**) vs Custom 1-Warp $72.80\ \mu\text{s}$ (**$1.302\times$ speedup**)
- `ffn_down`: Upstream MMVQ $76.30\ \mu\text{s}$ vs Custom 2-Warp $75.60\ \mu\text{s}$ (**$1.009\times$ speedup**) vs Custom 1-Warp $76.50\ \mu\text{s}$ ($0.997\times$)

---

Current meaningful benchmark: Q4_K x Q8_1 MMVQ decode microbenchmark suite across Qwen3-4B transformer shapes.
Current blocker: None for M3 implementation.
Next required action: Technical-lead M3 final audit and authorization (DO NOT PROCEED TO M4).
