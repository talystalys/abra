# CURRENT_STATUS.md

## Further speed work — Qwen3-4B

The user authorized implementation and tests after the research-only round.
The default two-card launcher now splits Qwen3-4B's duplicated tied output
projection by vocabulary rows and reuses unchanged Q8_1 activation bytes
within a graph. Model weights, F16 KV and upstream arithmetic remain unchanged.
`--unoptimized` preserves the previous tensor mode; single-card and exact modes
remain available. Placement/NCCL tuning and the new tensor kernel variants
were not selected after measured negative results.

The corrected comparison against frozen libraries measured generation at
105.42 to 115.37 tokens/s at empty context (+9.4%), 94.23 to 102.49 at 2K
(+8.8%), and 80.60 to 86.91 at 8K (+7.8%). Each mode/depth has 20 measured
samples and five excluded warmups per forward/reverse block. Prompt medians
changed +0.04%, +0.06% and +2.27%. The primary reached 91 C; dynamic clocks
and sustained thermal slowdown remain. The first timing attempt loaded the
candidate fallback via a copied executable's absolute RUNPATH and is
superseded. The harness now verifies actual resolved library paths/hashes.

All 14,585,856 selected-candidate logits across 0/2K/8K were bit-identical to
the frozen tensor/FA reference. Both cards completed 7542 total oracle and
activation regression cases. Fixed-corpus perplexity was 12.4064 in both
modes, with identical probe answers and 8/10 scores. Complete code copy/edit,
prose and reasoning matched all IDs across three repetitions. Each runtime
completed a 100-request / 25,600-token reset soak with identical outputs and
zero sampled swap. Allocation fell 152 MiB per card in the measured workloads.
All eight actual launcher smokes passed; context-4096 probes also matched
answers and retained 8/10. The two new installers were byte-idempotent.
[Full measurements, limitations and reproduction](docs/research/1080ti-speed-results.md).

## Previous optimization round — Qwen3-4B

The user authorized further research, implementation and tests in
[the next-round plan](docs/1080ti-next-plan.md). Default inference now uses
both GTX 1080 Ti GPUs with NCCL FP32, tensor splitting and F16 Flash Attention.
Balanced 20-sample decode comparisons: 72.18 → 94.10 tokens/s at 2K (+30.4%),
60.79 → 82.53 at 8K (+35.8%). Perplexity changed +0.194% with no lost fixed
task successes (8/10). NCCL matched the original tensor logits bit for bit
at 0/2K/8K; the strict single-versus-tensor failure remains recorded.
CPU KV/layers and quantized KV saved VRAM but slowed the screens. Prompt
lookup accelerated repeated code prefixes but failed general token equality
and remains explicitly experimental. A balanced complete 24-function copy
comparison measured 72.04 single, 90.00 tensor and 293.02 tensor lookup
tokens/s (20 samples each, identical complete text/IDs), a workload-specific
4.067× gain over the single-card baseline. The 100-request NCCL soak completed
25,600 tokens with identical outputs and flat sampled allocation; all 5838
original oracle cases and corrected launcher smokes completed. The 4K launcher
context probes retained 8/10 with no lost successes. [Full evidence and limitations](docs/research/1080ti-next-results.md).

## First port round — Qwen3-4B only

- Branch: `port/1080ti-runtime`, derived from M4 `815e3ff`.
- BigBoy checkout: `/home/taly/legacy-ai-1080ti-port`; public-key SSH to
  `taly@BigBoy` resolves to `<experimental-host-ip>` and works.
- Hardware: two GTX 1080 Ti cards, each SM 6.1 and 11 GiB; identities and
  current host specs are in [the inventory](docs/research/2026-10-12-bigboy-refresh.md).
- P0: the unchanged M4 oracle suite completed 756/756 cases on each card.
- P2: graph replay, exact-order Q4_K/Q6_K kernels and metadata prepacking
  passed full-vocabulary, 32-step validation at depths 0 and 2048.
- P3: thread-layout and activation-sum regressions are recorded. Custom QK/value
  kernels passed bit-identical full-runtime comparisons on both GTX 1080 Ti
  GPUs. Balanced final timing comparisons completed with 20 measured samples
  per configuration/depth and five excluded full-workload warmups per block.
- At depth 2048, Flash Attention improved primary decode 46.87 → 71.15
  tokens/s (+51.8%) and secondary decode 51.23 → 72.66 (+41.8%). Exact custom
  QK/value kernels reached 57.80 on the primary (+23.3%, bit-identical logits).
  The primary card showed thermal slowdown; the secondary was the first-round
  launcher default because it delivered higher sustained Flash Attention throughput.
- The user delegated the attention quality policy. Upstream Flash Attention
  passed the separately declared quality criterion: WikiText-2 perplexity
  12.3786 → 12.3824 (+0.031%) and 8/10 exact-format task successes in both modes.
  Its frozen strict-logit failure remains recorded. The first-round launcher selected Flash Attention; `--runtime exact` preserves validated exact output.
- [Milestones and acceptance criteria](docs/1080ti-port-plan.md). No historical
  milestone verdict is changed by the port.
- [Full results, limitations and reproduction](docs/research/1080ti-port-results.md).

## Historical M4 status


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
