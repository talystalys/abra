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
- **Correctness**: 168 / 168 test configurations PASSED (100% agreement with independent double-precision $F64$ CPU oracle and real model-byte evaluations from `Qwen3-4B-Q4_K_M.gguf`).
- **SASS Verification (`sm_61`)**: Confirmed native `IDP.4A.S8.S8` instructions in custom kernel SASS disassembly (`docs/evidence/m3-pascal-q4k-sass.txt`).
- **Hardware Resources & Theoretical Occupancy (CUDA Occupancy API)**:
  - Pascal (`sm_61`): Variant A (4-warp): 32 regs, 16 B smem, 100.0% occupancy; Variant B (2-warp): 40 regs, 16 B smem, 75.0% occupancy; Variant C (1-warp): 32 regs, 0 B smem, 100.0% occupancy. Zero register spills across all variants.
  - Turing (`sm_75`): All variants achieve 100.0% theoretical occupancy (32 resident warps/SM). Zero spills.

### Primary Qualification GPU (NVIDIA GeForce GTX 1060 6GB — Pascal SM 6.1):
- **`attn_q`** ($4096 \times 2560$):
  - Upstream MMVQ (4-warp Generic): $71.30\ \mu\text{s}$ ($147.00\text{ GMAC/s}$, $77.30\text{ GiB/s}$)
  - Custom 4-Warp Control: $90.90\ \mu\text{s}$ ($115.00\text{ GMAC/s}$, $60.60\text{ GiB/s}$, **$0.784\times$ vs U**)
  - Custom 2-Warp Split: $62.50\ \mu\text{s}$ ($168.00\text{ GMAC/s}$, $88.20\text{ GiB/s}$, **$1.141\times$ vs U**, **+14.1%**)
  - Custom 1-Warp Row: $66.60\ \mu\text{s}$ ($158.00\text{ GMAC/s}$, $82.80\text{ GiB/s}$, **$1.071\times$ vs U**, **+7.1%**)
- **`attn_kv`** ($1024 \times 2560$):
  - Upstream MMVQ: $21.50\ \mu\text{s}$ ($122.00\text{ GMAC/s}$, $64.20\text{ GiB/s}$)
  - Custom 2-Warp Split: $16.40\ \mu\text{s}$ ($160.00\text{ GMAC/s}$, $84.20\text{ GiB/s}$, **$1.311\times$ vs U**, **+31.1%**)
  - Custom 1-Warp Row: $16.10\ \mu\text{s}$ ($162.00\text{ GMAC/s}$, $85.50\text{ GiB/s}$, **$1.335\times$ vs U**, **+33.5%**)
- **`attn_out`** ($2560 \times 4096$):
  - Upstream MMVQ: $65.30\ \mu\text{s}$ ($161.00\text{ GMAC/s}$, $84.40\text{ GiB/s}$)
  - Custom 2-Warp Split: $62.00\ \mu\text{s}$ ($169.00\text{ GMAC/s}$, $88.80\text{ GiB/s}$, **$1.053\times$ vs U**, **+5.3%**)
  - Custom 1-Warp Row: $64.20\ \mu\text{s}$ ($163.00\text{ GMAC/s}$, $85.70\text{ GiB/s}$, **$1.017\times$ vs U**, **+1.7%**)
- **`ffn_gate_up`** ($9728 \times 2560$):
  - Upstream MMVQ: $161.00\ \mu\text{s}$ ($155.00\text{ GMAC/s}$, $81.40\text{ GiB/s}$)
  - Custom 2-Warp Split: $141.00\ \mu\text{s}$ ($176.00\text{ GMAC/s}$, $92.70\text{ GiB/s}$, **$1.142\times$ vs U**, **+14.2%**)
  - Custom 1-Warp Row: $147.00\ \mu\text{s}$ ($170.00\text{ GMAC/s}$, $89.10\text{ GiB/s}$, **$1.095\times$ vs U**, **+9.5%**)
- **`ffn_down`** ($2560 \times 9728$):
  - Upstream MMVQ: $145.00\ \mu\text{s}$ ($171.00\text{ GMAC/s}$, $89.90\text{ GiB/s}$)
  - Custom 2-Warp Split: $141.00\ \mu\text{s}$ ($176.00\text{ GMAC/s}$, $92.60\text{ GiB/s}$, **$1.028\times$ vs U**, **+2.8%**)
  - Custom 1-Warp Row: $141.00\ \mu\text{s}$ ($176.00\text{ GMAC/s}$, $92.50\text{ GiB/s}$, **$1.028\times$ vs U**, **+2.8%**)

### Secondary Control GPU (NVIDIA GeForce GTX 1660 SUPER — Turing SM 7.5):
- `attn_q`: Upstream MMVQ $38.40\ \mu\text{s}$ vs Custom 2-Warp $38.80\ \mu\text{s}$ (**$0.990\times$ vs U**) vs Custom 1-Warp $36.80\ \mu\text{s}$ (**$1.043\times$ vs U**)
- `attn_kv`: Upstream MMVQ $14.10\ \mu\text{s}$ vs Custom 2-Warp $12.30\ \mu\text{s}$ (**$1.146\times$ vs U**) vs Custom 1-Warp $12.20\ \mu\text{s}$ (**$1.156\times$ vs U**)
- `attn_out`: Upstream MMVQ $35.20\ \mu\text{s}$ vs Custom 2-Warp $37.50\ \mu\text{s}$ (**$0.939\times$ vs U**) vs Custom 1-Warp $36.90\ \mu\text{s}$ (**$0.954\times$ vs U**)
- `ffn_gate_up`: Upstream MMVQ $77.70\ \mu\text{s}$ vs Custom 2-Warp $78.50\ \mu\text{s}$ (**$0.990\times$ vs U**) vs Custom 1-Warp $75.40\ \mu\text{s}$ (**$1.031\times$ vs U**)
- `ffn_down`: Upstream MMVQ $74.50\ \mu\text{s}$ vs Custom 2-Warp $77.70\ \mu\text{s}$ (**$0.959\times$ vs U**) vs Custom 1-Warp $78.40\ \mu\text{s}$ (**$0.950\times$ vs U**)

---

Current meaningful benchmark: Q4_K x Q8_1 MMVQ decode microbenchmark suite across Qwen3-4B transformer shapes.
Current blocker: None for M3/M3R1 implementation.
Next required action: Technical-lead M3 final audit and authorization (DO NOT PROCEED TO M4).
