# Benchmark Methodology & Experimental Integrity Protocol

This document establishes the experimental protocol and benchmarking standards required for all performance investigations across the project.

---

## 1. Core Principles

1. **Reproducibility Over Speed**: A measurement is worthless unless any engineer or automated harness can reproduce it on the identical hardware state.
2. **No Cherry-Picking**: Selecting the single fastest run out of many noisy runs is strictly prohibited. Benchmarks must report distribution statistics (median, min, max, spread, standard deviation) alongside raw measurement logs.
3. **Full Hardware Identification**: Every benchmark output must explicitly record the executing GPU's runtime index, marketing name, compute capability, PCI bus ID, and hardware UUID.
4. **Control Boundaries**: Results measured on the GTX 1660 SUPER (`sm_75`) must never be described or substituted as Pascal (`sm_61`) results.

---

## 2. Experimental Controls & Variables to Hold Constant

All performance comparisons must explicitly specify and hold constant the following parameters:

### 2.1 Hardware & Environment State
- **Target GPU**: Exact physical card identified by PCI Bus ID and UUID.
- **Driver Version**: Locked NVIDIA driver release (e.g. `550.163.01`).
- **CUDA Toolkit**: Exact toolkit and compiler version (e.g. `CUDA 12.9.2` / `nvcc 12.9.86`).
- **Clocks & Power**: GPU core clock state, memory clock state, and power limit (`nvidia-smi -q -d CLOCK,POWER`).
- **Thermal State**: Adequate settling time between runs to prevent thermal throttling skew.
- **Host NUMA Affinity**: Process pinning (`numactl`) relative to the socket controlling the GPU's PCIe root complex.

### 2.2 Workload & Model Configuration
- **Model Identity**: Exact architecture, parameter count, and cryptographic hash of weights.
- **Quantization Scheme**: Quantization type (e.g. FP32, FP16, INT8, Q4_K, Q5_K, Q8_0), block sizes, and scaling formats.
- **Inference Phases**: In model-level evaluation, **prefill (prompt processing)** must be measured and reported separately from **autoregressive decode (token generation)**.
- **Batch & Context**: Exact batch size, sequence context length, prompt token count, and output token count.
- **Host-Device Offload**: Layer offloading ratio, pinned host memory usage, and transfer overheads.

### 2.3 Compilation & Kernel Parameters
- **Compiler Flags**: Exact `nvcc` flags (e.g. optimization levels, architecture flags, fast-math settings).
- **Launch Configurations**: Grid dimensions, thread block dimensions, dynamic shared memory allocation, and register spill counts (`--resource-usage`).

---

## 3. Timing Harness Requirements

Any benchmark harness used in this project must implement:

1. **Warm-Up Phase**:
   - Minimum of 5–10 warm-up iterations executed prior to measurement to prime caches, initialize driver state, and lock clock states.
2. **High-Precision Timing Mechanism**:
   - For GPU kernel micro-benchmarks: CUDA Events (`cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`).
   - For end-to-end inference: Monotonic high-resolution host timers combined with strict stream synchronization (`cudaStreamSynchronize`).
3. **Multiple Measurement Trials**:
   - Minimum of 20–100 measured iterations depending on workload duration.
4. **Statistical Reporting**:
   - Reporting must include:
     - Number of trials ($N$)
     - Median latency (primary metric)
     - Minimum latency
     - Maximum latency
     - Spread ($\text{Max} - \text{Min}$)
     - Standard deviation ($\sigma$)
     - Complete raw trial array
5. **Continuous Correctness Validation**:
   - The test harness must verify numerical results against CPU or mathematical oracle outputs. Never disable correctness checking to inflate benchmark throughput.

---

## 4. Benchmark Result Labeling

- Pre-baseline harness validation benchmarks (such as Milestone M0 smoke tests) must be prominently labeled:
  ```text
  HARNESS VALIDATION ONLY — NOT A PERFORMANCE RESULT
  ```
- Genuine performance baselines (starting in M1) must be accompanied by full raw output logs and exact reproduction commands.
