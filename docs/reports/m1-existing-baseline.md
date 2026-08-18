# Milestone M1 Report: llama.cpp Existing Inference Baseline

This report documents the empirical baseline established for modern quantized LLM inference on legacy NVIDIA Pascal hardware (`sm_61`) using an unmodified upstream inference runtime (`llama.cpp`), alongside a control baseline on NVIDIA Turing hardware (`sm_75`).

---

## 1. Scope

The purpose of Milestone M1 is to measure the best practical existing software baseline honestly and reproducibly under controlled experimental conditions on host `BigBoy`.

In accordance with the repository contract:
- No custom project kernels, DP4A research kernels, Q4 layouts, autotuning, or runtime optimizations are introduced.
- Upstream `llama.cpp` CUDA kernels are executed unmodified as pinned.
- Baseline measurements reflect the current upstream default dispatch behavior.

---

## 2. Pinned Software & Toolchain

### 2.1 Runtime
- **Repository**: `https://github.com/ggml-org/llama.cpp.git`
- **Pinned Commit**: `169e4a7ff201e666c2325bcd35afb64573e39d09`
- **Tracked Source State**: Completely clean (verified via `git diff-index --quiet HEAD --`)
- **Build Configurations**:
  1. **Default Upstream Baseline (Default MMQ Dispatch)**:
     ```bash
     CUDAFLAGS="-I.cuda_compat/include -Wno-deprecated-gpu-targets" \
     cmake -S .deps/llama.cpp -B .deps/llama.cpp/build-default \
         -DGGML_CUDA=ON \
         -DCMAKE_CUDA_ARCHITECTURES="61;75" \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
         -DCMAKE_BUILD_TYPE=Release \
         -DGGML_CUDA_FORCE_MMQ=OFF \
         -DGGML_CUDA_FORCE_CUBLAS=OFF
     ```
  2. **Forced-cuBLAS Diagnostic Build**:
     ```bash
     CUDAFLAGS="-I.cuda_compat/include -Wno-deprecated-gpu-targets" \
     cmake -S .deps/llama.cpp -B .deps/llama.cpp/build-cublas \
         -DGGML_CUDA=ON \
         -DCMAKE_CUDA_ARCHITECTURES="61;75" \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
         -DCMAKE_BUILD_TYPE=Release \
         -DGGML_CUDA_FORCE_CUBLAS=ON
     ```

### 2.2 System Toolchain
- **Host**: `BigBoy`
- **OS**: Debian GNU/Linux 13 (trixie)
- **Kernel**: Linux 6.12.100+deb13-amd64 x86_64
- **NVIDIA Driver**: `550.163.01`
- **CUDA Toolkit**: CUDA 12.9.2 (`/usr/local/cuda-12.9`)
- **NVCC**: 12.9.86 (`Cuda compilation tools, release 12.9, V12.9.86`)
- **Host Compiler**: GCC 14.2.0 (`gcc (Debian 14.2.0-19) 14.2.0`)
- **C Library**: glibc 2.41 (`ldd (Debian GLIBC 2.41-12+deb13u3) 2.41`)

---

## 3. Pinned Model

- **Repository**: `Qwen/Qwen3-4B-GGUF`
- **File**: `Qwen3-4B-Q4_K_M.gguf`
- **Quantization**: `Q4_K_M`
- **Exact File Size**: 2,497,280,256 bytes (2,381.6 MiB)
- **Parameters**: 4.02 Billion (`4,022,468,096`)
- **Expected SHA-256**: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`
- **Verified SHA-256**: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` (Exact match)

---

## 4. Hardware Demarcation

### 4.1 Primary Qualification GPU (Pascal SM 6.1)
- **Device**: NVIDIA GeForce GTX 1060 6GB
- **Core Architecture**: Pascal (GP104-A, CC 6.1)
- **PCI Bus ID**: `0000:03:00.0`
- **UUID**: `GPU-5078520c-bf92-4143-0d48-b02fd9d250dd`
- **Usable Global VRAM**: 6065 MiB GDDR5 (192-bit bus, ~192 GB/s peak theoretical bandwidth)
- **Power Limit**: 120.00 W
- **Host NUMA Node**: 0

### 4.2 Secondary Control GPU (Turing SM 7.5)
- **Device**: NVIDIA GeForce GTX 1660 SUPER
- **Core Architecture**: Turing (TU116-A, CC 7.5)
- **PCI Bus ID**: `0000:81:00.0`
- **UUID**: `GPU-8a043791-359d-1df4-9353-38c5bc98e5dd`
- **Usable Global VRAM**: 5929 MiB GDDR6 (192-bit bus, ~336 GB/s peak theoretical bandwidth)
- **Power Limit**: 125.00 W
- **Host NUMA Node**: 1

---

## 5. Methodology & Controlled Conditions

1. **GPU Isolation**:
   - Each benchmark invocation exposes strictly a single physical GPU by its immutable UUID (`CUDA_VISIBLE_DEVICES=${UUID}`).
   - The runtime confirms single-device enumeration before allocation.
2. **NUMA Affinity**:
   - Pascal processes are pinned to NUMA Node 0 via `numactl --cpunodebind=0 --membind=0`.
   - Turing processes are pinned to NUMA Node 1 via `numactl --cpunodebind=1 --membind=1`.
3. **Full GPU Offload**:
   - Full model offload requested (`-ngl 99`).
   - Startup logs confirm complete layer and tensor offloading:
     - `load_tensors: offloaded 37/37 layers to GPU`
     - `CUDA0 model buffer size = 2375.91 MiB`, `CPU_Mapped model buffer size = 304.28 MiB`
     - `llama_kv_cache: 36 layers: dev = CUDA0`
   - Complete runtime diagnostic logs are committed in [pascal-offload-proof.txt](../../bench/results/m1/pascal-offload-proof.txt) and [turing-offload-proof.txt](../../bench/results/m1/turing-offload-proof.txt).
4. **Canonical Workload**:
   - Prompt Processing: `512` tokens (`-p 512`)
   - Token Generation: `128` tokens (`-n 128`)
   - Context Depths: `0` and `2048` (`-d 0` and `-d 2048`)
   - Batch Size: `2048` (`-b 2048`)
   - Microbatch Size: `512` (`-ub 512`)
   - KV Cache Types: `F16 K`, `F16 V` (`-ctk f16 -ctv f16`)
   - Flash Attention: Upstream default (`auto`)
   - Repetitions: `10` measured iterations per test configuration.
5. **Statistical Metrics**:
   - Conventional median, arithmetic mean, minimum, maximum, spread, and sample standard deviation ($N - 1$ degrees of freedom) calculated from raw `samples_ns` / `samples_ts` arrays.
6. **Synchronous Telemetry**:
   - A dedicated telemetry sampler collected device temperature, power draw, SM clock, memory clock, utilization, and VRAM occupancy every 250 ms during all runs.

---

## 6. Primary Results

### 6.1 Pascal Default Baseline (GTX 1060 6GB)
- **Isolation**: `CUDA_VISIBLE_DEVICES=GPU-5078520c-bf92-4143-0d48-b02fd9d250dd`
- **NUMA**: Node 0 (`numactl --cpunodebind=0 --membind=0`)
- **Runtime Path**: Default Upstream (`build-default`, MMQ DP4A active)

| Workload | Depth | Median (t/s) | Mean (t/s) | Min (t/s) | Max (t/s) | Sample StdDev (t/s) | Spread (t/s) | Reps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **pp512** | `0` | **662.970** | 663.170 | 660.939 | 665.691 | 1.341 | 4.752 | 10 |
| **tg128** | `0` | **38.133** | 38.144 | 38.018 | 38.261 | 0.095 | 0.242 | 10 |
| **pp512** | `2048` | **454.218** | 454.108 | 452.156 | 455.680 | 0.902 | 3.524 | 10 |
| **tg128** | `2048` | **35.320** | 35.319 | 35.296 | 35.331 | 0.010 | 0.035 | 10 |

#### Individual Raw Samples (tokens/sec)
- **pp512 @ depth 0**: `[660.939, 661.555, 662.757, 662.919, 662.940, 663.000, 663.801, 664.036, 664.059, 665.691]`
- **tg128 @ depth 0**: `[38.018, 38.044, 38.070, 38.070, 38.093, 38.173, 38.221, 38.228, 38.260, 38.261]`
- **pp512 @ depth 2048**: `[452.156, 453.503, 453.811, 453.949, 454.186, 454.251, 454.338, 454.601, 454.602, 455.680]`
- **tg128 @ depth 2048**: `[35.296, 35.313, 35.316, 35.319, 35.320, 35.321, 35.322, 35.324, 35.328, 35.331]`

---

### 6.2 Turing Control Baseline (GTX 1660 SUPER)
- **Isolation**: `CUDA_VISIBLE_DEVICES=GPU-8a043791-359d-1df4-9353-38c5bc98e5dd`
- **NUMA**: Node 1 (`numactl --cpunodebind=1 --membind=1`)
- **Runtime Path**: Default Upstream (`build-default`, CC 7.5 MMQ selection)

| Workload | Depth | Median (t/s) | Mean (t/s) | Min (t/s) | Max (t/s) | Sample StdDev (t/s) | Spread (t/s) | Reps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **pp512** | `0` | **252.917** | 252.761 | 251.347 | 252.946 | 0.497 | 1.599 | 10 |
| **tg128** | `0` | **78.947** | 79.046 | 78.818 | 79.332 | 0.208 | 0.513 | 10 |
| **pp512** | `2048` | **192.135** | 192.042 | 191.230 | 192.145 | 0.286 | 0.915 | 10 |
| **tg128** | `2048` | **64.629** | 64.681 | 64.555 | 65.014 | 0.148 | 0.459 | 10 |

#### Individual Raw Samples (tokens/sec)
- **pp512 @ depth 0**: `[251.347, 252.880, 252.894, 252.915, 252.916, 252.918, 252.924, 252.933, 252.933, 252.946]`
- **tg128 @ depth 0**: `[78.818, 78.855, 78.875, 78.902, 78.909, 78.985, 79.222, 79.240, 79.323, 79.332]`
- **pp512 @ depth 2048**: `[191.230, 192.104, 192.122, 192.130, 192.135, 192.135, 192.135, 192.139, 192.144, 192.145]`
- **tg128 @ depth 2048**: `[64.555, 64.560, 64.569, 64.582, 64.594, 64.664, 64.687, 64.779, 64.808, 65.014]`

---

### 6.3 Forced-cuBLAS Diagnostic (GTX 1060 6GB)
- **Isolation**: `CUDA_VISIBLE_DEVICES=GPU-5078520c-bf92-4143-0d48-b02fd9d250dd`
- **NUMA**: Node 0 (`numactl --cpunodebind=0 --membind=0`)
- **Runtime Path**: Forced-cuBLAS (`-DGGML_CUDA_FORCE_CUBLAS=ON`)

| Workload | Depth | Median (t/s) | Mean (t/s) | Min (t/s) | Max (t/s) | Sample StdDev (t/s) | Spread (t/s) | Reps | Comparison to Default MMQ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **pp512** | `0` | **380.736** | 380.755 | 380.009 | 381.981 | 0.523 | 1.972 | 10 | **42.6% slower** (380.7 vs 663.0 t/s) |
| **tg128** | `0` | **37.883** | 37.885 | 37.866 | 37.901 | 0.012 | 0.035 | 10 | **0.7% slower** (37.88 vs 38.13 t/s) |

#### Individual Raw Samples (tokens/sec)
- **pp512 @ depth 0**: `[380.009, 380.438, 380.445, 380.453, 380.731, 380.742, 380.763, 380.991, 380.993, 381.981]`
- **tg128 @ depth 0**: `[37.866, 37.875, 37.875, 37.876, 37.883, 37.884, 37.890, 37.897, 37.899, 37.902]`

---

### 6.4 CPU Context Baseline (Context Only)
- **NUMA**: Node 0
- **Configuration**: `-ngl 0`, 32 host threads, 3 repetitions

| Workload | Depth | Median (t/s) | Mean (t/s) | Min (t/s) | Max (t/s) | Sample StdDev (t/s) | Reps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **pp512** | `0` | **211.369** | 210.166 | 207.735 | 211.393 | 2.105 | 3 |
| **tg128** | `0` | **11.217** | 11.192 | 11.136 | 11.224 | 0.049 | 3 |

---

## 7. Telemetry & Thermal Analysis

Telemetry was captured at 250 ms intervals during physical benchmark runs.

| Run Configuration | Samples | Temp Min–Max (°C) | Power Min–Max (W) | SM Clock Min–Max (MHz) | Mem Clock Min–Max (MHz) | Peak VRAM (MiB) | Thermal Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pascal Default Depth 0** | 151 | 48.0 – 70.0 | 7.4 – 127.3 | 139 – 1885 | 405 – 5003 | 2841 | Stable boost clocks |
| **Pascal Default Depth 2048** | 187 | 61.0 – 73.0 | 38.8 – 128.5 | 1695 – 1873 | 4799 – 5003 | 3129 | Stable boost clocks |
| **Pascal Forced-cuBLAS Depth 0** | 173 | 63.0 – 74.0 | 39.4 – 124.1 | 1695 – 1873 | 4799 – 5003 | 2929 | Stable boost clocks |
| **Turing Default Depth 0** | 142 | 44.0 – 68.0 | 9.4 – 126.4 | 300 – 1875 | 405 – 7000 | 2842 | Stable boost clocks |
| **Turing Default Depth 2048** | 210 | 58.0 – 74.0 | 28.6 – 128.1 | 495 – 1845 | 6801 – 7000 | 3130 | Stable boost clocks |

Both GPUs operated below their rated thermal ceilings (maximum observed 74.0 °C vs 83 °C thermal threshold) and sustained boost clocks throughout all measurement runs. No obvious thermal throttling is apparent from the recorded temperature and clock behavior, while explicit NVIDIA throttle-reason counters were not sampled in M1.

---

## 8. Source Inspection: Existing llama.cpp Pascal Path

Inspection of the pinned upstream source (`ggml-org/llama.cpp` at commit `169e4a7ff201e666c2325bcd35afb64573e39d09`) documents the exact architectural paths used for Pascal SM 6.1:

1. **Representation of Compute Capability 6.1**:
   - `ggml/src/ggml-cuda/common.cuh:50-51`:
     ```c++
     #define GGML_CUDA_CC_PASCAL 600
     #define GGML_CUDA_CC_DP4A   610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
     ```
2. **DP4A Capability & Intrinsics**:
   - `ggml/src/ggml-cuda/common.cuh:733-740`:
     ```c++
     #if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
         return __dp4a(a, b, c);
     #else
         const int8_t * a8 = (const int8_t *) &a;
         const int8_t * b8 = (const int8_t *) &b;
         return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
     #endif
     ```
3. **Quantized Type Support in MMQ**:
   - `ggml/src/ggml-cuda/mmq.cu:270-294`:
     Explicitly supports `GGML_TYPE_Q4_K` along with `Q2_K`, `Q3_K`, `Q5_K`, `Q6_K`, `IQ` variants, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`.
4. **MMQ vs cuBLAS Selection Criteria**:
   - `ggml/src/ggml-cuda/mmq.cu:324-326`:
     ```c++
     if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
         return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
     }
     ```
   - On Pascal (`cc == 610`), `fp16_mma_hardware_available(cc)` is `false` (requires `cc >= 700` Volta).
   - Therefore `!fp16_mma_hardware_available(cc)` evaluates to `true`, and upstream `llama.cpp` selects the MMQ path by default for all batch sizes.
5. **Vector-Dot Implementation for Q4_K × Q8_1**:
   - `ggml/src/ggml-cuda/mmq-vec-dot.cuh:913`:
     ```c++
     template <ggml_type type, int J, bool fallback>
     static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a(
         const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00)
     ```
6. **SASS Disassembly Verification**:
   - Inspection of `mmq-instance-q4_k.cu.o` using `cuobjdump -sass` confirms dense generation of hardware `IDP.4A.S8.S8` instructions (Pascal DP4A byte dot-product with INT32 accumulation) targeting SM 6.1 (2,912 matching instructions recorded in [m1-pascal-q4k-sass.txt](../evidence/m1-pascal-q4k-sass.txt)).

---

## 9. End-to-End Sanity Check (`llama-cli`)

A deterministic single-turn generation was executed on both GPUs with the exact prompt:
> `"Explain in one short paragraph why memory bandwidth matters for neural-network inference."`
- **Seed**: `42`, **Temperature**: `0.0`, **GPU Layers**: `99` (Full offload)

### Pascal Execution Output
```text
> Explain in one short paragraph why memory bandwidth matters for neural-network inference.

[Start thinking]

Okay, the user is asking why memory bandwidth matters for neural-network inference. Let me start by recalling what memory bandwidth is. It's the rate at which data can be read from or written to memory. Now, neural networks, especially deep ones, require a lot of data movement. During inference, the model processes input data through multiple layers, each layer needing to access weights and activations.

So, if the memory bandwidth is low, the system can't keep up with the data transfer rate. That would cause bottlenecks. For example, if the GPU or CPU is fast at processing but the memory is slow, the data

[ Prompt: 337.6 t/s | Generation: 37.5 t/s ]
```

### Turing Execution Output
```text
> Explain in one short paragraph why memory bandwidth matters for neural-network inference.

[Start thinking]

Okay, the user is asking why memory bandwidth matters for neural-network inference. Let me start by recalling what memory bandwidth is. It's the rate at which data can be read from or written to memory. Now, neural networks, especially deep ones, require a lot of data movement. During inference, the model processes input data through multiple layers, each layer needing to access weights and activations.

So, if the memory bandwidth is low, the system can't keep up with the data transfer rate. That would cause bottlenecks. For example, if the GPU or CPU is fast at processing but the memory is slow, the data

[ Prompt: 179.9 t/s | Generation: 75.7 t/s ]
```

**Observation**: Text generation output across Pascal and Turing is identical word-for-word, confirming numerical alignment and full deterministic model functionality.

---

## 10. Interpretation & Findings

1. **Current llama.cpp Baseline**:
   - Current `llama.cpp` achieved **662.970 median pp512 tokens/s** and **38.133 median tg128 tokens/s** on the physical GTX 1060 6GB under default dispatch at context depth 0.
   - At context depth 2048, prompt processing drops to **454.218 median tokens/s** (a 31.5% reduction, consistent with increased KV cache attention computation overhead), while token generation slightly reduces to **35.320 median tokens/s** (a 7.4% reduction).
2. **Forced-cuBLAS vs MMQ DP4A**:
   - Forcing dequantization + cuBLAS on Pascal degrades prompt processing by **42.6%** (down to 380.736 tokens/s), while token generation remains essentially unchanged (37.883 tokens/s).
   - This indicates that upstream's default MMQ selection is an active, highly effective existing optimization on Pascal.
3. **Turing Control Path & Cross-Architecture Observations**:
   - The GTX 1660 SUPER default run enters `llama.cpp`'s MMQ selection according to the pinned CC 7.5 dispatch logic (`turing_mma_available(cc)`). The build contains `sm_75`, so CC 7.5 satisfies that architecture-level MMQ selection condition. The GTX 1660 SUPER is a TU116 GTX product without physical Tensor Cores. M1 does not establish the exact instruction-level behavior of that MMQ path on TU116.
   - In token generation (`tg128`), the GTX 1660 SUPER achieves **78.947 tokens/s** compared to Pascal's **38.133 tokens/s**, which is consistent with the higher theoretical memory bandwidth of GDDR6 (336 GB/s) over GDDR5 (192 GB/s) and Turing memory subsystem differences.
   - In prompt processing (`pp512`), Pascal default MMQ achieves **662.970 tokens/s**, while the Turing default run achieves **252.917 tokens/s**.
   - M1 does not yet establish the limiting hardware resource for these execution paths; explicit bottleneck proof is deferred to M2.
4. **Baseline Characterization**:
   - Upstream `llama.cpp` already possesses a mature DP4A quantized execution path for Pascal CC 6.1. Future research in this repository must acknowledge this established high watermark.

---

## 11. Known Limitations

- In M1, profiling with hardware performance counters was not performed due to Pascal SM 6.x architecture profiling constraints under recent toolchain releases; detailed hardware counter profiling is deferred to M2.
- Explicit NVIDIA GPU throttle-reason bitmasks were not captured in the 250 ms telemetry sampling stream.

---

## 12. M1 Acceptance Evidence Files

All raw benchmark JSON outputs, telemetry logs, offload proofs, disassembly evidence, and machine-readable manifests are preserved in the repository:
- Manifest: [manifest.json](../../bench/results/m1/manifest.json)
- Statistical Analysis: [analysis-summary.json](../../bench/results/m1/analysis-summary.json)
- Pascal Offload Proof: [pascal-offload-proof.txt](../../bench/results/m1/pascal-offload-proof.txt)
- Turing Offload Proof: [turing-offload-proof.txt](../../bench/results/m1/turing-offload-proof.txt)
- Pascal SASS Disassembly Evidence: [m1-pascal-q4k-sass.txt](../evidence/m1-pascal-q4k-sass.txt)
- Pascal Default Depth 0: [pascal-default-depth0.json](../../bench/results/m1/pascal-default-depth0.json)
- Pascal Default Depth 2048: [pascal-default-depth2048.json](../../bench/results/m1/pascal-default-depth2048.json)
- Pascal Default Depth 0 Telemetry: [pascal-default-depth0-telemetry.csv](../../bench/results/m1/pascal-default-depth0-telemetry.csv)
- Pascal Default Depth 2048 Telemetry: [pascal-default-depth2048-telemetry.csv](../../bench/results/m1/pascal-default-depth2048-telemetry.csv)
- Pascal Forced cuBLAS Depth 0: [pascal-cublas-depth0.json](../../bench/results/m1/pascal-cublas-depth0.json)
- Pascal Forced cuBLAS Depth 0 Telemetry: [pascal-cublas-depth0-telemetry.csv](../../bench/results/m1/pascal-cublas-depth0-telemetry.csv)
- Turing Default Depth 0: [turing-default-depth0.json](../../bench/results/m1/turing-default-depth0.json)
- Turing Default Depth 2048: [turing-default-depth2048.json](../../bench/results/m1/turing-default-depth2048.json)
- Turing Default Depth 0 Telemetry: [turing-default-depth0-telemetry.csv](../../bench/results/m1/turing-default-depth0-telemetry.csv)
- Turing Default Depth 2048 Telemetry: [turing-default-depth2048-telemetry.csv](../../bench/results/m1/turing-default-depth2048-telemetry.csv)
- CPU Context Depth 0: [cpu-context-depth0.json](../../bench/results/m1/cpu-context-depth0.json)
- Pascal Sanity Log: [pascal-cli-sanity.txt](../../bench/results/m1/pascal-cli-sanity.txt)
- Turing Sanity Log: [turing-cli-sanity.txt](../../bench/results/m1/turing-cli-sanity.txt)
