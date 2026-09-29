# Project Milestones Roadmap

This document outlines the research and engineering roadmap for exploring modern quantized neural-network inference on legacy NVIDIA architectures (specifically NVIDIA Pascal SM 6.1).

> **Current authorization (2026-10-12):** execute P0–P4 in
> [the GTX 1080 Ti port plan](docs/1080ti-port-plan.md), for Qwen3-4B only.
> The prior M5 authorization wait below is historical and does not block
> this explicitly requested port.

> **Next-round authorization (2026-10-12):** the user requested further
> research, implementation and testing, including RAM placement. Follow
> [the next-round plan](docs/1080ti-next-plan.md); measurements and remaining
> limitations are in [the results](docs/research/1080ti-next-results.md).
> This does not change historical milestone verdicts.

> **Historical M4 status**: Milestone M4 is currently **completed and awaiting technical lead audit**. Future milestones require review, audit of evidence, and explicit authorization from the technical lead before implementation begins.

---

## Roadmap

- **M0: Reproducible Environment** *(COMPLETED — PASS)*
  - Establish trustworthy experimental host baseline on BigBoy.
  - Verify CUDA 12.9.2 toolchain and glibc 2.41 compatibility.
  - Verify dual code generation for `sm_61` (Pascal) and `sm_75` (Turing).
  - Physical device identification and separate correctness execution on GTX 1060 and GTX 1660 SUPER.
  - Validate CUDA event timing harness infrastructure.

- **M1: Existing Baseline** *(COMPLETED — PASS)*
  - Quantified baseline inference performance on legacy hardware using upstream `llama.cpp` MMQ on `Qwen3-4B-Q4_K_M.gguf`.
  - Established hardware offload proofs and telemetry on Pascal (GTX 1060) and Turing (GTX 1660 SUPER).

- **M2: DP4A Microkernel** *(COMPLETED — PASS)*
  - Implemented signed INT8 $\times$ signed INT8 $\to$ INT32 matrix-vector arithmetic microkernels for Pascal SM 6.1 (`IDP.4A.S8.S8`).
  - Verified 100% bit-exact correctness against CPU 64-bit accumulation reference oracle across 156 test cases on both Pascal and Turing GPUs.
  - SASS disassembly analysis proving 0 `IDP.4A` in scalar baseline and active `IDP.4A.S8.S8` in DP4A variants.
  - Characterized $2.60\times\text{--}3.64\times$ speedup on Pascal over compiler-verified scalar baseline, achieving up to $176.13\text{ GiB/s}$ effective requested throughput.

- **M3: Pascal Q4_K × Q8_1 Decode Microkernel Characterization** *(COMPLETED — PASS)*
  - Identified actual upstream decode incumbent as MMVQ (`ne11 <= 8`).
  - Linked directly against pinned `libggml-cuda.so` (`ggml_cuda_op_mul_mat_vec_q`, `quantize_row_q8_1_cuda`) and `libggml-base.so` (`quantize_row_q4_K_ref`).
  - Extracted canonical model tensor map (`Qwen3-4B-Q4_K_M.gguf`, 398 tensors: 216 Q4_K, 37 Q6_K, 145 F32).
  - Implemented custom Q4_K × Q8_1 MMVQ microkernel variants (Variant A: 4-warp control, Variant B: 2-warp split, Variant C: 1-warp row).
  - Validated 168/168 tests against double-precision reference oracle and pinned upstream comparator.
  - Characterized $+16.3\%\text{--}+31.1\%$ speedup on Pascal $K=2560$ shapes over actual pinned upstream MMVQ.

- **M4: Pascal Autotuned Kernel Family** *(COMPLETED — AWAITING TECHNICAL LEAD AUDIT)*
  - Parameterized kernel family `q4k_q8_mmvq_family_kernel<WARPS_PER_ROW, WARPS_PER_BLOCK, UNROLL>` spanning 17 candidate variants.
  - Verified deadlock-free predication and zero register spilling across all candidates.
  - Characterized theoretical occupancy (100% for $WPR=1$, 75% for $WPR=2,4$).
  - Built automated autotuner orchestrator and deterministic shape-aware tuning table.
  - Validated 756/756 correctness tests across synthetic distributions, boundary edges, and real model bytes.
  - Qualified $1.101\times$ aggregate decode speedup (+9.2% latency reduction) on Pascal GP106 across all 216 canonical $Q4\_K$ model tensors in `Qwen3-4B-Q4_K_M.gguf`.

- **M5: Runtime Integration** *(PLANNED — NOT STARTED / NOT AUTHORIZED)*
  - Integrate quantized execution kernels into a lightweight inference execution pipeline.

- **M6: Bottleneck Reassessment** *(PLANNED)*
  - Re-evaluate system bottlenecks across compute, memory bandwidth, latency, and host-device transfer.

- **M7: Pascal-Specific Backend** *(PLANNED)*
  - Architect execution backend optimized specifically around Pascal SM 6.1 memory hierarchy and compute pipelines.

- **M8: Architecture-Aware Compiler/Generator** *(PLANNED)*
  - Automated code generation and specialized kernel synthesis for legacy targets.

- **M9: Hardware/Model Co-Design (Optional)** *(PLANNED)*
  - Co-design quantization representations and architecture parameters tailored to legacy GPU limits.
