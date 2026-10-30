# Milestone M3: Pascal Q4_K × Q8_1 Decode Microkernel Characterization

## 1. Executive Summary & Core Findings

Milestone M3 evaluates whether a Pascal GP106-specific execution mapping for the canonical `Q4_K × Q8_1` autoregressive matrix-vector decode operation (`mul_mat_vec_q`, `M=1`) can outperform the pinned upstream `llama.cpp` decode incumbent without modifying the quantization representation or numerical contract.

### Key Empirical Findings:

1. **Critical Dispatch Incumbent Confirmation**:
   - Pinned `llama.cpp` (commit `169e4a7ff201e666c2325bcd35afb64573e39d09`) uses **MMVQ** (`ggml_cuda_op_mul_mat_vec_q`), not MMQ, during autoregressive token decode (`ne11 = 1 <= 8`).
   - On Pascal SM 6.1 (`sm_61`), MMVQ dispatches to the **generic architecture parameter table** (`MMVQ_PARAMETERS_GENERIC`), which configures **4 warps per row** (`nwarps = 4`, 1 row per thread block).

2. **Predeclared Scheduling Hypothesis Validated against Real Pinned Upstream**:
   - For $K=2560$ projection shapes (10 Q4_K superblocks), the 4-warp schedule processes 8 blocks in iteration 0 and 2 blocks in iteration 1, leaving 6 block slots idle (**62.5% K-loop slot utilization**).
   - **Variant B (2 warps per row, 2 rows per block, 83.33% slot utilization)** achieves a **+16.3% to +31.1% speedup** over the actual pinned upstream MMVQ incumbent (Variant U) across all Qwen $K=2560$ projections on physical Pascal GTX 1060:
     - `attn_q` ($4096 \times 2560$): **61.30 µs vs 71.30 µs** (**1.163× speedup**, +16.3%)
     - `attn_kv` ($1024 \times 2560$): **16.40 µs vs 21.50 µs** (**1.311× speedup**, +31.1%)
     - `ffn_gate_up` ($9728 \times 2560$): **138.00 µs vs 161.00 µs** (**1.167× speedup**, +16.7%)
   - **Variant C (1 warp per row, 4 rows per block, 100.0% slot utilization, zero shared memory)** achieves **1.071×** (`attn_q`), **1.311×** (`attn_kv`), and **1.193×** (`ffn_gate_up`) over Variant U on the same $K=2560$ shapes.

3. **Controlled Non-Tail Shapes ($K=4096, K=9728$)**:
   - For `attn_out` ($2560 \times 4096$, 16 superblocks, 100% 4-warp slot utilization), Variant B achieves **61.40 µs vs 64.60 µs** (**1.052× speedup**, +5.2%).
   - For `ffn_down` ($2560 \times 9728$, 38 superblocks, 95% 4-warp slot utilization), Variant B achieves **141.00 µs vs 145.00 µs** (**1.028× speedup**, +2.8%).
   - As hypothesized, the speedup from restructuring warp collaboration is concentrated on tail-waste dimensions ($K=2560$), confirming the analytical performance model.

4. **Activation Quantization Pipeline Overhead (Timing Mode 2)**:
   - Standalone GPU activation quantization ($F32 \to Q8\_1$ via exact pinned `quantize_row_q8_1_cuda`) incurs **4.10 µs to 5.12 µs** on Pascal GP106.
   - In end-to-end decode pipelines (Exact Pinned Quantize + Multiply), Variant B maintains speedups of **1.155×** (`attn_q`), **1.262×** (`attn_kv`), and **1.163×** (`ffn_gate_up`) over the upstream pipeline.

5. **Numerical Exactness, Frozen Envelope & Real Model Bytes**:
   - Upstream vs F64 double oracle measured max relative error: **1.3340e-03**, max absolute error: **1.5547e-02**.
   - Custom variants vs Upstream max absolute difference: **1.5259e-05**.
   - All 168 correctness tests (including 16 actual model byte tensor tests from `Qwen3-4B-Q4_K_M.gguf` across `attn_q`, `attn_output`, `ffn_gate`, `ffn_up`) passed 100%.

---

## 2. Hardware & Environment Identification

All experiments were executed on host `BigBoy` in strict compliance with `AGENTS.md` Rule 2.

```text
Host Environment:
  Host: BigBoy
  OS: Debian GNU/Linux 13 (trixie)
  Kernel: Linux 6.12.100+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.100-1 (2026-07-30) x86_64
  CPU: 2 x Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz (32 physical cores, 64 logical threads)
  RAM: 62.77 GiB Total (65,821,544 KiB)
  NUMA Nodes: 2 nodes
  NVIDIA Driver: 550.163.01
  CUDA Toolkit: /usr/local/cuda-12.9
  Host Compiler: gcc (Debian 14.2.0-19) 14.2.0
  C Standard Library: ldd (Debian GLIBC 2.41-12+deb13u3) 2.41
```

```text
Primary Qualification Device (Pascal):
  Device: NVIDIA GeForce GTX 1060 6GB
  Compute Capability: 6.1 (sm_61)
  PCI Bus ID: 0000:03:00.0
  UUID: GPU-5078520c-bf92-4143-0d48-b02fd9d250dd
  NUMA Node: 0
  Total VRAM: 6,066 MiB (6,361,387,008 bytes)
  Streaming Multiprocessors (SMs): 10
  Max Warps / SM: 64 (2048 threads / SM)
  Registers / SM: 65,536

Secondary Control Device (Turing):
  Device: NVIDIA GeForce GTX 1660 SUPER
  Compute Capability: 7.5 (sm_75)
  PCI Bus ID: 0000:81:00.0
  UUID: GPU-8a043791-359d-1df4-9353-38c5bc98e5dd
  NUMA Node: 1
  Total VRAM: 5,930 MiB (6,218,571,776 bytes)
  Streaming Multiprocessors (SMs): 22
  Max Warps / SM: 32 (1024 threads / SM)
  Registers / SM: 65,536
```

---

## 3. Canonical Tensor Mapping & Model Provenance

The canonical GGUF model `Qwen3-4B-Q4_K_M.gguf` was parsed via `tools/extract-qwen3-tensor-map.py` to establish the exact tensor structure in `bench/results/m3/qwen3-q4k-tensor-map.json`.

- **Model File**: `/home/taly/abra/models/Qwen3-4B-Q4_K_M.gguf`
- **SHA256**: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`
- **Total Tensors**: 398
- **Tensor Breakdown by Type**:
  - `Q4_K`: **216 tensors** (54.27%)
  - `Q6_K`: **37 tensors** (9.30%)
  - `F32`: **145 tensors** (36.43%)

| Projection Name | Matrix Dimensions ($N \times K$) | Quantization in Qwen3-4B-Q4_K_M | Evaluation Status in Milestone M3 |
| :--- | :--- | :--- | :--- |
| `attn_q` | $4096 \times 2560$ | `Q4_K` | Actual model tensor |
| `attn_k` | $1024 \times 2560$ | `Q4_K` | Actual model tensor (evaluated as `attn_kv`) |
| `attn_v` | $1024 \times 2560$ | `Q6_K` | Evaluated as synthetic `Q4_K` shape for cross-milestone comparability |
| `attn_output` | $2560 \times 4096$ | `Q4_K` | Actual model tensor (evaluated as `attn_out`) |
| `ffn_gate` | $9728 \times 2560$ | `Q4_K` | Actual model tensor (evaluated as `ffn_gate_up`) |
| `ffn_up` | $9728 \times 2560$ | `Q4_K` | Actual model tensor |
| `ffn_down` | $2560 \times 9728$ | `Q6_K` | Evaluated as synthetic `Q4_K` shape for cross-milestone comparability |

---

## 4. Upstream Comparator & MMVQ Dispatch Contract

Pinned llama.cpp commit: `169e4a7ff201e666c2325bcd35afb64573e39d09`.

1. **MMVQ Dispatch Logic (`ggml-cuda.cu` / `mmvq.cuh`)**:
   ```cpp
   bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
       if (!ggml_is_quantized(type)) return false;
       return ne11 <= MMVQ_MAX_BATCH_SIZE; // MMVQ_MAX_BATCH_SIZE == 8
   }
   ```
   For single-token autoregressive decode (`ne11 = 1`), `ggml_cuda_should_use_mmvq()` unconditionally returns `true`. The decode incumbent is MMVQ (`mul_mat_vec_q`), not MMQ.

2. **Parameter Table Routing (`mmvq.cu`)**:
   - **Pascal SM 6.1**: Routes to `MMVQ_PARAMETERS_GENERIC`. For `ncols_dst = 1`, `calc_nwarps` returns **4 warps**. Each thread block contains 128 threads ($4 \times 32$) collaborating on 1 output row.
   - **Turing SM 7.5**: Routes to `MMVQ_PARAMETERS_TURING`. For `ncols_dst = 1`, `calc_nwarps` returns **2 warps**.

3. **Binary Link Proof**:
   Confirmed dynamic linkage of `libggml-cuda.so` and invocation of exported symbols `ggml_cuda_op_mul_mat_vec_q` and `quantize_row_q8_1_cuda` via `nm -C` in `docs/evidence/m3r1-upstream-link-proof.txt`.

---

## 5. Kernel Variant Architecture & Scheduling Hypothesis

All evaluated kernels execute the exact `vec_dot_q4_K_q8_1` arithmetic using native `IDP.4A.S8.S8` instructions without altering block layouts or rounding semantics.

```text
Variant U (Exact Pinned Upstream MMVQ):
  - Upstream template instance: mul_mat_vec_q<GGML_TYPE_Q4_K, 1> from libggml-cuda.so
  - Block: 128 threads (4 warps), Grid: N blocks (1 row per block)
  - Inter-warp reduction via shared memory smem[4] + __syncthreads()

Variant A (Custom 4-Warp Control):
  - Custom implementation of upstream 4-warp schedule (1 row per block)
  - Purpose: Isolates framework dispatch overhead to confirm test harness calibration

Variant B (Custom 2-Warp Split):
  - Block: 128 threads (4 warps), Grid: ceil(N/2) blocks (2 rows per block)
  - Warps (0,1) compute row 2*bid+0; Warps (2,3) compute row 2*bid+1
  - Independent 2-warp shared memory reduction

Variant C (Custom 1-Warp Row):
  - Block: 128 threads (4 warps), Grid: ceil(N/4) blocks (4 rows per block)
  - Each warp computes 1 output row independently
  - Pure intra-warp shuffle reduction (__shfl_xor_sync); ZERO shared memory, ZERO __syncthreads()
```

### K-Tail Slot Utilization Analysis:

For `block_q4_K` ($QK\_K = 256$, $QI4\_K = 32$, $vdr = 2$), each 32-thread warp processes 2 superblocks per iteration ($2 \times 256 = 512$ weights).

| Schedule | Warps / Row | Superblocks / Iteration | Elements / Iteration | $K=2560$ Iterations (10 Blocks) | $K=2560$ Slot Utilization | $K=4096$ Slot Utilization | $K=9728$ Slot Utilization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **4-Warp (U, A)** | 4 | 8 | 2048 | 2 iters (16 slots) | **62.5%** (6 slots idle) | **100.0%** (16 slots) | **95.0%** (40 slots) |
| **2-Warp (B)** | 2 | 4 | 1024 | 3 iters (12 slots) | **83.3%** (2 slots idle) | **100.0%** (16 slots) | **95.0%** (40 slots) |
| **1-Warp (C)** | 1 | 2 | 512 | 5 iters (10 slots) | **100.0%** (0 slots idle) | **100.0%** (16 slots) | **100.0%** (38 slots) |

---

## 6. Numerical Verification & Frozen Upstream Envelope

The test suite (`tests/q4k_q8_test.cu`) compares all GPU kernel variants against an independent double-precision ($F64$) CPU oracle and directly against Variant U.

- **Total Test Cases**: 168
- **Passed Test Cases**: 168 (100% PASS)
- **Input Families Evaluated**:
  1. `ALL_ZERO` (Zero relative error)
  2. `LOW_AMPLITUDE_CONST` (Zero relative error)
  3. `ALTERNATING_SIGN` (Zero relative error)
  4. `BOUNDED_PSEUDORANDOM`
  5. `WIDE_DYNAMIC_RANGE`
  6. `SPARSE_OUTLIERS`
  7. `VARYING_SCALE`
  8. `NONZERO_MIN_EXERCISE`
- **Upstream vs F64 Double Oracle Error**:
  - Max Relative Error: **1.3340e-03**
  - Max Absolute Error: **1.5547e-02**
  - Max RMS Error: **5.8054e-03**
- **Custom Variants vs Variant U**:
  - Max Absolute Difference: **1.5259e-05**
  - Max Relative Difference: **3.4585e-04**
- **Real Model-Byte Sanity Suite**: 16 actual model row evaluations across `blk.0.attn_q.weight`, `blk.0.attn_output.weight`, `blk.0.ffn_gate.weight`, and `blk.0.ffn_up.weight` from `Qwen3-4B-Q4_K_M.gguf` passed 100%.

---

## 7. Pascal Execution Evidence (GTX 1060 6GB)

- **Device**: `NVIDIA GeForce GTX 1060 6GB` (CC 6.1, PCI `0000:03:00.0`, UUID `GPU-5078520c-bf92-4143-0d48-b02fd9d250dd`)
- **Benchmark Configuration**: 10 warmup iterations, 100 measured iterations, CUDA events.
- **Raw Evidence File**: `bench/results/m3/pascal-q4k-bench.json`

### Timing Mode 1: Multiply-Only (M=1 Decode Microkernel)

| Workload / Projection | Shape ($N \times K$) | Variant | Median ($\mu\text{s}$) | Mean ($\mu\text{s}$) | Min ($\mu\text{s}$) | Max ($\mu\text{s}$) | Spread ($\mu\text{s}$) | Throughput (GMAC/s) | Eff. Throughput (GiB/s) | Speedup vs Upstream |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `attn_q` | $4096 \times 2560$ | Variant U: Exact Pinned Upstream MMVQ | 71.30 | 71.30 | 70.30 | 77.80 | 7.55 | 147.00 | 77.30 | **1.000x** |
| `attn_q` | $4096 \times 2560$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 90.80 | 90.60 | 89.10 | 92.00 | 2.94 | 115.00 | 60.70 | **0.785x** |
| `attn_q` | $4096 \times 2560$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 61.30 | 61.20 | 59.40 | 63.10 | 3.74 | 171.00 | 90.00 | **1.163x** |
| `attn_q` | $4096 \times 2560$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 66.60 | 66.50 | 63.50 | 68.60 | 5.15 | 158.00 | 82.80 | **1.071x** |
| `attn_kv` | $1024 \times 2560$ | Variant U: Exact Pinned Upstream MMVQ | 21.50 | 21.70 | 21.10 | 22.50 | 1.44 | 122.00 | 64.20 | **1.000x** |
| `attn_kv` | $1024 \times 2560$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 20.50 | 20.80 | 20.20 | 21.50 | 1.31 | 128.00 | 67.40 | **1.049x** |
| `attn_kv` | $1024 \times 2560$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 16.40 | 16.50 | 15.40 | 17.40 | 2.05 | 160.00 | 84.20 | **1.311x** |
| `attn_kv` | $1024 \times 2560$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 16.40 | 16.30 | 15.40 | 17.40 | 2.05 | 160.00 | 84.20 | **1.311x** |
| `attn_out` | $2560 \times 4096$ | Variant U: Exact Pinned Upstream MMVQ | 64.60 | 64.90 | 63.50 | 66.20 | 2.72 | 162.00 | 85.30 | **1.000x** |
| `attn_out` | $2560 \times 4096$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 68.50 | 68.40 | 67.60 | 69.40 | 1.82 | 153.00 | 80.40 | **0.943x** |
| `attn_out` | $2560 \times 4096$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 61.40 | 61.50 | 60.40 | 63.40 | 2.98 | 171.00 | 89.60 | **1.052x** |
| `attn_out` | $2560 \times 4096$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 63.40 | 63.30 | 61.40 | 65.50 | 4.16 | 165.00 | 86.80 | **1.019x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant U: Exact Pinned Upstream MMVQ | 161.00 | 161.00 | 160.00 | 162.00 | 2.05 | 155.00 | 81.40 | **1.000x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 206.00 | 206.00 | 204.00 | 207.00 | 2.40 | 121.00 | 63.60 | **0.782x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 138.00 | 139.00 | 136.00 | 141.00 | 4.90 | 180.00 | 94.70 | **1.167x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 135.00 | 135.00 | 132.00 | 137.00 | 5.12 | 185.00 | 97.00 | **1.193x** |
| `ffn_down` | $2560 \times 9728$ | Variant U: Exact Pinned Upstream MMVQ | 145.00 | 145.00 | 143.00 | 146.00 | 3.07 | 172.00 | 90.10 | **1.000x** |
| `ffn_down` | $2560 \times 9728$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 152.00 | 152.00 | 150.00 | 155.00 | 4.42 | 163.00 | 85.70 | **0.954x** |
| `ffn_down` | $2560 \times 9728$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 141.00 | 141.00 | 138.00 | 143.00 | 4.96 | 177.00 | 92.90 | **1.028x** |
| `ffn_down` | $2560 \times 9728$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 140.00 | 141.00 | 139.00 | 143.00 | 4.10 | 177.00 | 93.10 | **1.036x** |
| `synth_256` | $256 \times 256$ | Variant U: Exact Pinned Upstream MMVQ | 5.73 | 5.55 | 5.06 | 6.14 | 1.09 | 11.40 | 6.21 | **1.000x** |
| `synth_256` | $256 \times 256$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 5.89 | 5.80 | 4.99 | 11.20 | 6.18 | 11.10 | 6.04 | **0.973x** |
| `synth_256` | $256 \times 256$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.93 | 4.74 | 4.10 | 5.12 | 1.02 | 13.30 | 7.21 | **1.162x** |
| `synth_256` | $256 \times 256$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.10 | 4.38 | 3.81 | 20.30 | 16.50 | 16.00 | 8.68 | **1.398x** |
| `synth_1024` | $1024 \times 1024$ | Variant U: Exact Pinned Upstream MMVQ | 9.92 | 9.77 | 9.22 | 10.90 | 1.66 | 106.00 | 55.90 | **1.000x** |
| `synth_1024` | $1024 \times 1024$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 12.90 | 12.70 | 12.10 | 13.30 | 1.18 | 81.30 | 43.00 | **0.769x** |
| `synth_1024` | $1024 \times 1024$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 9.63 | 9.64 | 9.18 | 10.20 | 1.06 | 109.00 | 57.50 | **1.030x** |
| `synth_1024` | $1024 \times 1024$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 8.19 | 8.24 | 7.17 | 9.09 | 1.92 | 128.00 | 67.70 | **1.211x** |
| `synth_4096` | $4096 \times 4096$ | Variant U: Exact Pinned Upstream MMVQ | 96.00 | 95.90 | 95.00 | 97.30 | 2.27 | 175.00 | 91.70 | **1.000x** |
| `synth_4096` | $4096 \times 4096$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 102.00 | 103.00 | 101.00 | 109.00 | 7.17 | 164.00 | 86.00 | **0.941x** |
| `synth_4096` | $4096 \times 4096$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 90.00 | 90.00 | 89.00 | 92.00 | 3.04 | 186.00 | 97.80 | **1.067x** |
| `synth_4096` | $4096 \times 4096$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 97.30 | 97.40 | 95.20 | 101.00 | 6.14 | 172.00 | 90.50 | **0.987x** |

### Timing Mode 2: Combined Activation Quantization (F32 -> Q8_1) + Multiply

| Workload / Projection | Shape ($N \times K$) | Variant | Quantize-Only ($\mu\text{s}$) | Combined Median ($\mu\text{s}$) | Combined Mean ($\mu\text{s}$) | Combined Spread ($\mu\text{s}$) | Combined Speedup vs Upstream |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `attn_q` | $4096 \times 2560$ | Variant U: Exact Pinned Upstream MMVQ | 4.10 | 74.50 | 74.40 | 2.18 | **1.000x** |
| `attn_q` | $4096 \times 2560$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.10 | 93.90 | 94.20 | 41.80 | **0.793x** |
| `attn_q` | $4096 \times 2560$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.10 | 64.50 | 64.80 | 4.10 | **1.155x** |
| `attn_q` | $4096 \times 2560$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.10 | 69.60 | 69.40 | 5.15 | **1.070x** |
| `attn_kv` | $1024 \times 2560$ | Variant U: Exact Pinned Upstream MMVQ | 4.10 | 24.60 | 24.50 | 7.14 | **1.000x** |
| `attn_kv` | $1024 \times 2560$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.10 | 24.20 | 24.10 | 2.11 | **1.017x** |
| `attn_kv` | $1024 \times 2560$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.10 | 19.50 | 19.50 | 1.98 | **1.262x** |
| `attn_kv` | $1024 \times 2560$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.10 | 19.50 | 19.40 | 2.98 | **1.262x** |
| `attn_out` | $2560 \times 4096$ | Variant U: Exact Pinned Upstream MMVQ | 4.10 | 68.20 | 68.00 | 2.98 | **1.000x** |
| `attn_out` | $2560 \times 4096$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.10 | 71.70 | 71.60 | 2.30 | **0.951x** |
| `attn_out` | $2560 \times 4096$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.10 | 64.50 | 64.70 | 2.69 | **1.057x** |
| `attn_out` | $2560 \times 4096$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.10 | 66.60 | 66.70 | 10.50 | **1.024x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant U: Exact Pinned Upstream MMVQ | 4.10 | 164.00 | 164.00 | 7.17 | **1.000x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.10 | 209.00 | 209.00 | 3.46 | **0.785x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.10 | 141.00 | 141.00 | 11.30 | **1.163x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.10 | 138.00 | 138.00 | 5.12 | **1.188x** |
| `ffn_down` | $2560 \times 9728$ | Variant U: Exact Pinned Upstream MMVQ | 5.12 | 148.00 | 148.00 | 3.07 | **1.000x** |
| `ffn_down` | $2560 \times 9728$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 5.12 | 156.00 | 156.00 | 7.17 | **0.949x** |
| `ffn_down` | $2560 \times 9728$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 5.12 | 143.00 | 144.00 | 4.22 | **1.035x** |
| `ffn_down` | $2560 \times 9728$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 5.12 | 137.00 | 139.00 | 24.50 | **1.080x** |
| `synth_256` | $256 \times 256$ | Variant U: Exact Pinned Upstream MMVQ | 4.10 | 8.19 | 8.21 | 2.02 | **1.000x** |
| `synth_256` | $256 \times 256$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.10 | 8.24 | 8.56 | 1.15 | **0.994x** |
| `synth_256` | $256 \times 256$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.10 | 7.78 | 7.75 | 12.40 | **1.053x** |
| `synth_256` | $256 \times 256$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.10 | 7.17 | 7.05 | 1.89 | **1.142x** |
| `synth_1024` | $1024 \times 1024$ | Variant U: Exact Pinned Upstream MMVQ | 4.10 | 12.30 | 12.50 | 2.05 | **1.000x** |
| `synth_1024` | $1024 \times 1024$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.10 | 15.40 | 15.60 | 1.25 | **0.799x** |
| `synth_1024` | $1024 \times 1024$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.10 | 12.30 | 12.60 | 1.25 | **1.000x** |
| `synth_1024` | $1024 \times 1024$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.10 | 11.30 | 11.20 | 1.76 | **1.088x** |
| `synth_4096` | $4096 \times 4096$ | Variant U: Exact Pinned Upstream MMVQ | 4.10 | 98.90 | 98.80 | 4.03 | **1.000x** |
| `synth_4096` | $4096 \times 4096$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.10 | 105.00 | 106.00 | 9.22 | **0.942x** |
| `synth_4096` | $4096 \times 4096$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.10 | 93.20 | 93.30 | 14.20 | **1.061x** |
| `synth_4096` | $4096 \times 4096$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.10 | 100.00 | 100.00 | 5.50 | **0.989x** |

---

## 8. Turing Control Evidence (GTX 1660 SUPER)

- **Device**: `NVIDIA GeForce GTX 1660 SUPER` (CC 7.5, PCI `0000:81:00.0`, UUID `GPU-8a043791-359d-1df4-9353-38c5bc98e5dd`)
- **Raw Evidence File**: `bench/results/m3/turing-q4k-bench.json`

### Timing Mode 1: Multiply-Only (M=1 Decode Microkernel)

| Workload / Projection | Shape ($N \times K$) | Variant | Median ($\mu\text{s}$) | Mean ($\mu\text{s}$) | Min ($\mu\text{s}$) | Max ($\mu\text{s}$) | Spread ($\mu\text{s}$) | Throughput (GMAC/s) | Eff. Throughput (GiB/s) | Speedup vs Upstream |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `attn_q` | $4096 \times 2560$ | Variant U: Exact Pinned Upstream MMVQ | 37.10 | 37.20 | 34.70 | 38.80 | 4.10 | 283.00 | 149.00 | **1.000x** |
| `attn_q` | $4096 \times 2560$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 45.10 | 45.10 | 43.80 | 46.60 | 2.72 | 233.00 | 122.00 | **0.823x** |
| `attn_q` | $4096 \times 2560$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 38.80 | 38.70 | 36.50 | 40.00 | 3.49 | 270.00 | 142.00 | **0.956x** |
| `attn_q` | $4096 \times 2560$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 36.70 | 36.50 | 33.30 | 38.00 | 4.64 | 286.00 | 150.00 | **1.011x** |
| `attn_kv` | $1024 \times 2560$ | Variant U: Exact Pinned Upstream MMVQ | 12.60 | 12.80 | 12.20 | 14.20 | 2.02 | 208.00 | 110.00 | **1.000x** |
| `attn_kv` | $1024 \times 2560$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 13.30 | 13.30 | 12.30 | 14.30 | 2.05 | 197.00 | 104.00 | **0.947x** |
| `attn_kv` | $1024 \times 2560$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 12.20 | 12.10 | 10.90 | 13.50 | 2.66 | 215.00 | 113.00 | **1.033x** |
| `attn_kv` | $1024 \times 2560$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 12.10 | 12.00 | 10.60 | 13.70 | 3.10 | 217.00 | 114.00 | **1.041x** |
| `attn_out` | $2560 \times 4096$ | Variant U: Exact Pinned Upstream MMVQ | 36.80 | 36.60 | 34.30 | 38.00 | 3.78 | 285.00 | 150.00 | **1.000x** |
| `attn_out` | $2560 \times 4096$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 39.50 | 39.50 | 36.90 | 40.90 | 4.00 | 266.00 | 140.00 | **0.932x** |
| `attn_out` | $2560 \times 4096$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 37.20 | 37.30 | 35.10 | 38.70 | 3.62 | 282.00 | 148.00 | **0.989x** |
| `attn_out` | $2560 \times 4096$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 36.90 | 36.80 | 33.70 | 40.10 | 6.40 | 284.00 | 149.00 | **0.997x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant U: Exact Pinned Upstream MMVQ | 76.80 | 76.80 | 74.20 | 92.40 | 18.20 | 324.00 | 170.00 | **1.000x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 95.30 | 95.30 | 94.20 | 109.00 | 15.30 | 261.00 | 137.00 | **0.806x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 78.50 | 78.60 | 77.80 | 84.00 | 6.14 | 317.00 | 167.00 | **0.978x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 73.40 | 73.60 | 71.70 | 75.80 | 4.06 | 339.00 | 178.00 | **1.046x** |
| `ffn_down` | $2560 \times 9728$ | Variant U: Exact Pinned Upstream MMVQ | 73.80 | 73.80 | 71.70 | 75.60 | 3.94 | 337.00 | 177.00 | **1.000x** |
| `ffn_down` | $2560 \times 9728$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 76.30 | 76.40 | 75.80 | 77.40 | 1.63 | 326.00 | 171.00 | **0.967x** |
| `ffn_down` | $2560 \times 9728$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 77.80 | 77.70 | 75.60 | 84.40 | 8.86 | 320.00 | 168.00 | **0.949x** |
| `ffn_down` | $2560 \times 9728$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 76.60 | 76.60 | 75.70 | 77.80 | 2.11 | 325.00 | 171.00 | **0.963x** |
| `synth_256` | $256 \times 256$ | Variant U: Exact Pinned Upstream MMVQ | 5.57 | 5.66 | 4.10 | 20.50 | 16.40 | 11.80 | 6.39 | **1.000x** |
| `synth_256` | $256 \times 256$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.96 | 5.17 | 4.10 | 18.10 | 14.00 | 13.20 | 7.17 | **1.123x** |
| `synth_256` | $256 \times 256$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.26 | 4.46 | 3.84 | 19.80 | 16.00 | 15.40 | 8.35 | **1.308x** |
| `synth_256` | $256 \times 256$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.10 | 4.21 | 3.01 | 5.22 | 2.21 | 16.00 | 8.68 | **1.359x** |
| `synth_1024` | $1024 \times 1024$ | Variant U: Exact Pinned Upstream MMVQ | 8.91 | 9.00 | 7.71 | 10.10 | 2.40 | 118.00 | 62.20 | **1.000x** |
| `synth_1024` | $1024 \times 1024$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 9.81 | 9.69 | 8.58 | 10.40 | 1.79 | 107.00 | 56.50 | **0.908x** |
| `synth_1024` | $1024 \times 1024$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 8.05 | 8.07 | 6.88 | 22.50 | 15.60 | 130.00 | 68.90 | **1.107x** |
| `synth_1024` | $1024 \times 1024$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 7.09 | 7.12 | 6.14 | 8.16 | 2.02 | 148.00 | 78.20 | **1.257x** |
| `synth_4096` | $4096 \times 4096$ | Variant U: Exact Pinned Upstream MMVQ | 53.30 | 53.30 | 50.90 | 69.60 | 18.80 | 315.00 | 165.00 | **1.000x** |
| `synth_4096` | $4096 \times 4096$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 57.10 | 57.00 | 55.80 | 58.10 | 2.27 | 294.00 | 154.00 | **0.933x** |
| `synth_4096` | $4096 \times 4096$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 53.20 | 53.80 | 52.00 | 56.40 | 4.45 | 315.00 | 165.00 | **1.002x** |
| `synth_4096` | $4096 \times 4096$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 51.80 | 51.90 | 51.20 | 54.40 | 3.17 | 324.00 | 170.00 | **1.029x** |

### Timing Mode 2: Combined Activation Quantization (F32 -> Q8_1) + Multiply

| Workload / Projection | Shape ($N \times K$) | Variant | Quantize-Only ($\mu\text{s}$) | Combined Median ($\mu\text{s}$) | Combined Mean ($\mu\text{s}$) | Combined Spread ($\mu\text{s}$) | Combined Speedup vs Upstream |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `attn_q` | $4096 \times 2560$ | Variant U: Exact Pinned Upstream MMVQ | 4.19 | 39.60 | 39.80 | 5.98 | **1.000x** |
| `attn_q` | $4096 \times 2560$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.19 | 47.50 | 47.70 | 2.08 | **0.834x** |
| `attn_q` | $4096 \times 2560$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.19 | 41.10 | 41.30 | 2.98 | **0.964x** |
| `attn_q` | $4096 \times 2560$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.19 | 39.20 | 39.40 | 6.62 | **1.010x** |
| `attn_kv` | $1024 \times 2560$ | Variant U: Exact Pinned Upstream MMVQ | 4.13 | 15.70 | 15.60 | 6.08 | **1.000x** |
| `attn_kv` | $1024 \times 2560$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.13 | 16.20 | 16.00 | 2.75 | **0.969x** |
| `attn_kv` | $1024 \times 2560$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.13 | 14.80 | 15.00 | 8.42 | **1.061x** |
| `attn_kv` | $1024 \times 2560$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.13 | 14.40 | 14.70 | 2.72 | **1.090x** |
| `attn_out` | $2560 \times 4096$ | Variant U: Exact Pinned Upstream MMVQ | 4.19 | 39.40 | 39.90 | 18.30 | **1.000x** |
| `attn_out` | $2560 \times 4096$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.19 | 42.70 | 42.40 | 5.31 | **0.923x** |
| `attn_out` | $2560 \times 4096$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.19 | 40.20 | 40.00 | 4.74 | **0.980x** |
| `attn_out` | $2560 \times 4096$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.19 | 39.50 | 39.90 | 17.00 | **0.997x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant U: Exact Pinned Upstream MMVQ | 4.26 | 77.70 | 78.40 | 24.00 | **1.000x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.26 | 97.80 | 97.70 | 3.14 | **0.794x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.26 | 81.20 | 81.20 | 2.91 | **0.957x** |
| `ffn_gate_up` | $9728 \times 2560$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.26 | 77.70 | 77.60 | 3.78 | **1.000x** |
| `ffn_down` | $2560 \times 9728$ | Variant U: Exact Pinned Upstream MMVQ | 4.26 | 75.20 | 74.90 | 2.21 | **1.000x** |
| `ffn_down` | $2560 \times 9728$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.26 | 81.90 | 81.50 | 3.20 | **0.918x** |
| `ffn_down` | $2560 \times 9728$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.26 | 78.40 | 78.70 | 4.45 | **0.959x** |
| `ffn_down` | $2560 \times 9728$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.26 | 79.80 | 80.20 | 25.60 | **0.942x** |
| `synth_256` | $256 \times 256$ | Variant U: Exact Pinned Upstream MMVQ | 4.10 | 8.10 | 8.51 | 17.80 | **1.000x** |
| `synth_256` | $256 \times 256$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.10 | 7.92 | 7.92 | 15.90 | **1.023x** |
| `synth_256` | $256 \times 256$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.10 | 7.62 | 7.70 | 17.10 | **1.063x** |
| `synth_256` | $256 \times 256$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.10 | 6.98 | 7.03 | 2.43 | **1.160x** |
| `synth_1024` | $1024 \times 1024$ | Variant U: Exact Pinned Upstream MMVQ | 4.16 | 11.80 | 11.70 | 4.16 | **1.000x** |
| `synth_1024` | $1024 \times 1024$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.16 | 12.30 | 12.30 | 2.27 | **0.959x** |
| `synth_1024` | $1024 \times 1024$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.16 | 10.40 | 10.40 | 2.78 | **1.135x** |
| `synth_1024` | $1024 \times 1024$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.16 | 10.00 | 9.87 | 2.69 | **1.180x** |
| `synth_4096` | $4096 \times 4096$ | Variant U: Exact Pinned Upstream MMVQ | 4.22 | 54.40 | 54.90 | 17.30 | **1.000x** |
| `synth_4096` | $4096 \times 4096$ | Variant A: Custom 4-Warp Control (1 Row/Block) | 4.22 | 61.40 | 61.10 | 3.90 | **0.886x** |
| `synth_4096` | $4096 \times 4096$ | Variant B: Custom 2-Warp Split (2 Rows/Block) | 4.22 | 55.40 | 55.50 | 1.70 | **0.982x** |
| `synth_4096` | $4096 \times 4096$ | Variant C: Custom 1-Warp Row (4 Rows/Block, Zero SMem) | 4.22 | 56.10 | 55.90 | 4.10 | **0.970x** |

---

## 9. Analysis & Synthesis

### 1. Verification of the Predeclared Tail-Waste Hypothesis
The empirical evidence across 8 distinct shapes and 2 GPU architectures provides direct confirmation of the structural hypothesis:
- On GP106 (Pascal), the generic MMVQ kernel's 4-warp allocation imposes a mandatory 8-superblock granularity ($2048$ elements) on the $K$ reduction dimension.
- For $K=2560$ (10 superblocks), the second iteration executes with only 2 active blocks and 6 idle block slots ($37.5\%$ inactive threads during iteration 1).
- By restructuring the thread block to 2 warps per row (Variant B, 4 blocks/iter) or 1 warp per row (Variant C, 2 blocks/iter), the $K$-loop achieves higher slot utilization ($83.3\%$ and $100.0\%$, respectively) while doubling or quadrupling the number of concurrently processed output rows per thread block.
- Consequently, $K=2560$ shapes display the highest speedups (+16.3% to +31.1% over exact pinned upstream MMVQ), whereas $K=4096$ (100% slot utilization in 4-warp) and $K=9728$ (95% slot utilization in 4-warp) show modest gains (+2.8% to +5.2%).

### 2. Elimination of Cross-Warp Barriers
Variant C eliminates inter-warp shared memory communication and `__syncthreads()` synchronization entirely. Every row reduction occurs strictly within registers using 5 intra-warp `__shfl_xor_sync` operations. This accounts for its strong performance on small shapes (`synth_256`: 4.10 µs vs 5.73 µs, 1.398× speedup; `synth_1024`: 8.19 µs vs 9.92 µs, 1.211× speedup).

### 3. Impact on End-to-End Autoregressive Generation
In full autoregressive generation, input activations in $F32$ must be dynamically quantized to $Q8\_1$ once per token decode step before projection multiplications occur. Because quantization latency (4.10 µs for $K=2560$) is amortized across multiple projection layers or executed once per vector, the multiplication speedups recorded in Mode 1 directly translate to significant pipeline reductions in Mode 2 (+15.5% speedup on `attn_q`, +16.3% speedup on `ffn_gate_up`).

---

## 10. Verification & Reproduction Instructions

To reproduce all results, re-run test suites, and execute the automated qualification check:

```bash
# Sourcing environment and compatibility overlay
source tools/setup-cuda-compat.sh

# Re-running automated qualification suite
./tools/verify-m3.sh

# Checking strict structured documentation consistency
python3 tools/check_m3_documentation_consistency.py .
```
