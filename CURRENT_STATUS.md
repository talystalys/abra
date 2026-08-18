# CURRENT_STATUS.md

Current milestone: M1 (Awaiting Technical-Lead Audit & Authorization — DO NOT PROCEED TO M2)
Current branch: m1/existing-baseline
Accepted M0 base commit: a2e207f0b2dfaa8ac40f3a1d638e0aef5524d4f2
(Git repository HEAD is authoritative for current commit state)

Execution host: BigBoy

Pascal qualification GPU: NVIDIA GeForce GTX 1060 6GB
Pascal PCI/UUID: PCI 0000:03:00.0 / UUID GPU-5078520c-bf92-4143-0d48-b02fd9d250dd
Pascal compute capability: 6.1 (NUMA node 0)

Control GPU: NVIDIA GeForce GTX 1660 SUPER
Control PCI/UUID: PCI 0000:81:00.0 / UUID GPU-8a043791-359d-1df4-9353-38c5bc98e5dd
Control compute capability: 7.5 (NUMA node 1)

NVIDIA driver: 550.163.01
CUDA toolkit: CUDA 12.9.2 (/usr/local/cuda-12.9)
NVCC: 12.9.86

---

## Milestone M1 Baseline Findings

- **Pinned llama.cpp runtime**: `ggml-org/llama.cpp` @ commit `169e4a7ff201e666c2325bcd35afb64573e39d09`
- **Pinned Model**: `Qwen/Qwen3-4B-GGUF` (`Qwen3-4B-Q4_K_M.gguf`, SHA-256 `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`)
- **Full GPU Offload**: 100% layer and weight residence verified in GPU VRAM (`offloaded 37/37 layers to GPU`, peak observed VRAM: 2841–3129 MiB out of 6065 MiB usable)

### Primary Qualification GPU (NVIDIA GeForce GTX 1060 6GB — Pascal SM 6.1):
- **Pascal Default Baseline (MMQ DP4A active)**:
  - `pp512 @ depth 0`: **662.970 median tokens/s** (Mean: 663.170, Min: 660.939, Max: 665.691, Sample StdDev: 1.341 t/s, 10 reps)
  - `tg128 @ depth 0`: **38.133 median tokens/s** (Mean: 38.144, Min: 38.018, Max: 38.261, Sample StdDev: 0.095 t/s, 10 reps)
  - `pp512 @ depth 2048`: **454.218 median tokens/s** (Mean: 454.108, Min: 452.156, Max: 455.680, Sample StdDev: 0.902 t/s, 10 reps)
  - `tg128 @ depth 2048`: **35.320 median tokens/s** (Mean: 35.319, Min: 35.296, Max: 35.331, Sample StdDev: 0.010 t/s, 10 reps)
- **Forced-cuBLAS Diagnostic**:
  - `pp512 @ depth 0`: **380.736 median tokens/s** (42.6% slower than MMQ default)
  - `tg128 @ depth 0`: **37.883 median tokens/s** (0.7% slower than MMQ default)

### Secondary Control GPU (NVIDIA GeForce GTX 1660 SUPER — Turing SM 7.5):
- **Turing Default Baseline**:
  - `pp512 @ depth 0`: **252.917 median tokens/s** (Mean: 252.761, Min: 251.347, Max: 252.946, Sample StdDev: 0.497 t/s, 10 reps)
  - `tg128 @ depth 0`: **78.947 median tokens/s** (Mean: 79.046, Min: 78.818, Max: 79.332, Sample StdDev: 0.208 t/s, 10 reps)
  - `pp512 @ depth 2048`: **192.135 median tokens/s** (Mean: 192.042, Min: 191.230, Max: 192.145, Sample StdDev: 0.286 t/s, 10 reps)
  - `tg128 @ depth 2048`: **64.629 median tokens/s** (Mean: 64.681, Min: 64.555, Max: 65.014, Sample StdDev: 0.148 t/s, 10 reps)

### Telemetry & Hardware Safety:
- Sustained SM clocks: 1695–1885 MHz on Pascal; 1800–1875 MHz on Turing
- Peak observed temperatures: 74.0 °C (Pascal) and 74.0 °C (Turing)
- No obvious thermal throttling was apparent from the recorded temperature/clock behavior, while explicit throttle-reason counters were not sampled.

---

Current meaningful benchmark: `llama.cpp` upstream default MMQ baseline on `Qwen3-4B-Q4_K_M.gguf`.
Current blocker: None for M1 implementation.
Next required action: Technical-lead M1 audit and authorization (DO NOT PROCEED TO M2).
