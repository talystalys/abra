# CURRENT_STATUS.md

Current milestone: M3 (Pascal Q4_K x Q8_1 Decode Microkernel Characterization — Complete — Awaiting Technical-Lead Audit & Authorization — DO NOT PROCEED TO M4)
Current branch: m3/q4k-q8-microkernel
Accepted M2 base commit: f10ead5fa94925e0a36980b1f3bdaa4a6998aa84
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
NVCC: nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_May_27_02:21:03_PDT_2025
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0
Host compiler: gcc (Debian 14.2.0-19) 14.2.0
glibc: ldd (Debian GLIBC 2.41-12+deb13u3) 2.41

---

## Milestone M3 Q4_K x Q8_1 MMVQ Decode Microkernel Characterization Summary

- **Mathematical Contract**: $y[n] = \sum_{k=0}^{K-1} \text{dequantize}(W[n, k]) \cdot \text{dequantize}(x[k])$ with $W \in \text{block\_q4\_K}^{N \times (K/256)}$ (144 bytes per 256 weights = 4.5 bpw), $x \in \text{block\_q8\_1}^{K/32}$ (36 bytes per 32 activations), and $y \in \text{float32}^N$.
- **Correctness**: 168 / 168 test configurations PASSED (100% agreement with independent double-precision $F64$ CPU oracle and real model-byte evaluations from `Qwen3-4B-Q4_K_M.gguf`).
- **SASS Verification (`sm_61`)**: Confirmed native `IDP.4A.S8.S8` instructions in custom kernel SASS disassembly (`docs/evidence/m3-pascal-q4k-sass.txt`).
- **Hardware Resources & Theoretical Occupancy (CUDA Occupancy API)**:
  - Pascal (`sm_61`): Variant A (4-warp): 32 regs, 16 B smem, 100.0% occupancy; Variant B (2-warp): 40 regs, 16 B smem, 75.0% occupancy; Variant C (1-warp): 32 regs, 0 B smem, 100.0% occupancy. Zero register spills across all variants.
  - Turing (`sm_75`): All variants achieve 100.0% theoretical occupancy (32 resident warps/SM). Zero spills.

### Primary Qualification GPU (NVIDIA GeForce GTX 1060 6GB — Pascal SM 6.1):
- **`attn_q`** ($4096 \times 2560$):
  - Upstream MMVQ (4-warp Generic): $71.30\ \mu\text{s}$ ($147.00\text{ GMAC/s}$, $77.30\text{ GiB/s}$)
  - Custom 4-Warp Control: $90.80\ \mu\text{s}$ ($115.00\text{ GMAC/s}$, $60.70\text{ GiB/s}$, **$0.785\times$ vs U**)
  - Custom 2-Warp Split: $61.30\ \mu\text{s}$ ($171.00\text{ GMAC/s}$, $90.00\text{ GiB/s}$, **$1.163\times$ vs U**, **+16.3%**)
  - Custom 1-Warp Row: $66.60\ \mu\text{s}$ ($158.00\text{ GMAC/s}$, $82.80\text{ GiB/s}$, **$1.071\times$ vs U**, **+7.1%**)
- **`attn_kv`** ($1024 \times 2560$):
  - Upstream MMVQ: $21.50\ \mu\text{s}$ ($122.00\text{ GMAC/s}$, $64.20\text{ GiB/s}$)
  - Custom 2-Warp Split: $16.40\ \mu\text{s}$ ($160.00\text{ GMAC/s}$, $84.20\text{ GiB/s}$, **$1.311\times$ vs U**, **+31.1%**)
  - Custom 1-Warp Row: $16.40\ \mu\text{s}$ ($160.00\text{ GMAC/s}$, $84.20\text{ GiB/s}$, **$1.311\times$ vs U**, **+31.1%**)
- **`attn_out`** ($2560 \times 4096$):
  - Upstream MMVQ: $64.60\ \mu\text{s}$ ($162.00\text{ GMAC/s}$, $85.30\text{ GiB/s}$)
  - Custom 2-Warp Split: $61.40\ \mu\text{s}$ ($171.00\text{ GMAC/s}$, $89.60\text{ GiB/s}$, **$1.052\times$ vs U**, **+5.2%**)
  - Custom 1-Warp Row: $63.40\ \mu\text{s}$ ($165.00\text{ GMAC/s}$, $86.80\text{ GiB/s}$, **$1.019\times$ vs U**, **+1.9%**)
- **`ffn_gate_up`** ($9728 \times 2560$):
  - Upstream MMVQ: $161.00\ \mu\text{s}$ ($155.00\text{ GMAC/s}$, $81.40\text{ GiB/s}$)
  - Custom 2-Warp Split: $138.00\ \mu\text{s}$ ($180.00\text{ GMAC/s}$, $94.70\text{ GiB/s}$, **$1.167\times$ vs U**, **+16.7%**)
  - Custom 1-Warp Row: $135.00\ \mu\text{s}$ ($185.00\text{ GMAC/s}$, $97.00\text{ GiB/s}$, **$1.193\times$ vs U**, **+19.3%**)
- **`ffn_down`** ($2560 \times 9728$):
  - Upstream MMVQ: $145.00\ \mu\text{s}$ ($172.00\text{ GMAC/s}$, $90.10\text{ GiB/s}$)
  - Custom 2-Warp Split: $141.00\ \mu\text{s}$ ($177.00\text{ GMAC/s}$, $92.90\text{ GiB/s}$, **$1.028\times$ vs U**, **+2.8%**)
  - Custom 1-Warp Row: $140.00\ \mu\text{s}$ ($177.00\text{ GMAC/s}$, $93.10\text{ GiB/s}$, **$1.036\times$ vs U**, **+3.6%**)

### Secondary Control GPU (NVIDIA GeForce GTX 1660 SUPER — Turing SM 7.5):
- `attn_q`: Upstream MMVQ $37.10\ \mu\text{s}$ vs Custom 2-Warp $38.80\ \mu\text{s}$ (**$0.956\times$ vs U**) vs Custom 1-Warp $36.70\ \mu\text{s}$ (**$1.011\times$ vs U**)
- `attn_kv`: Upstream MMVQ $12.60\ \mu\text{s}$ vs Custom 2-Warp $12.20\ \mu\text{s}$ (**$1.033\times$ vs U**) vs Custom 1-Warp $12.10\ \mu\text{s}$ (**$1.041\times$ vs U**)
- `attn_out`: Upstream MMVQ $36.80\ \mu\text{s}$ vs Custom 2-Warp $37.20\ \mu\text{s}$ (**$0.989\times$ vs U**) vs Custom 1-Warp $36.90\ \mu\text{s}$ (**$0.997\times$ vs U**)
- `ffn_gate_up`: Upstream MMVQ $76.80\ \mu\text{s}$ vs Custom 2-Warp $78.50\ \mu\text{s}$ (**$0.978\times$ vs U**) vs Custom 1-Warp $73.40\ \mu\text{s}$ (**$1.046\times$ vs U**)
- `ffn_down`: Upstream MMVQ $73.80\ \mu\text{s}$ vs Custom 2-Warp $77.80\ \mu\text{s}$ (**$0.949\times$ vs U**) vs Custom 1-Warp $76.60\ \mu\text{s}$ (**$0.963\times$ vs U**)

---

Current meaningful benchmark: Q4_K x Q8_1 MMVQ decode microbenchmark suite across Qwen3-4B transformer shapes.
Current blocker: None for M3/M3R1 implementation.
Next required action: Technical-lead M3 final audit and authorization (DO NOT PROCEED TO M4).
