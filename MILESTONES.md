# Project Milestones Roadmap

This document outlines the research and engineering roadmap for exploring modern quantized neural-network inference on legacy NVIDIA architectures (specifically NVIDIA Pascal SM 6.1).

> **Important**: Milestone M3 characterization is currently **completed and awaiting technical lead audit**. Future milestones require review, audit of evidence, and explicit authorization from the technical lead before implementation begins.

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

- **M3: Pascal Q4_K × Q8_1 Decode Microkernel Characterization** *(COMPLETED — AWAITING TECHNICAL LEAD AUDIT)*
  - Identified actual upstream decode incumbent as MMVQ (`ne11 <= 8`).
  - Linked directly against pinned `libggml-cuda.so` (`ggml_cuda_op_mul_mat_vec_q`, `quantize_row_q8_1_cuda`) and `libggml-base.so` (`quantize_row_q4_K_ref`).
  - Extracted canonical model tensor map (`Qwen3-4B-Q4_K_M.gguf`, 398 tensors: 216 Q4_K, 37 Q6_K, 145 F32).
  - Implemented custom Q4_K × Q8_1 MMVQ microkernel variants (Variant A: 4-warp control, Variant B: 2-warp split, Variant C: 1-warp row).
  - Validated 168/168 tests (including real model bytes from `Qwen3-4B-Q4_K_M.gguf`) against double-precision reference oracle and pinned upstream comparator.
  - Characterized $+16.3\%\text{--}+31.1\%$ speedup on Pascal $K=2560$ shapes over actual pinned upstream MMVQ by resolving 4-warp $K$-tail idle slot waste.

- **M4: Autotuned Kernel Family** *(PLANNED — NOT STARTED / NOT AUTHORIZED)*
  - Develop family of parameter-tuned kernels (tile size, warp layout, memory access patterns) for Pascal hardware characteristics.

- **M5: Runtime Integration** *(PLANNED)*
  - Integrate quantized execution kernels into a lightweight inference execution pipeline.

- **M6: Bottleneck Reassessment** *(PLANNED)*
  - Re-evaluate system bottlenecks across compute, memory bandwidth, latency, and host-device transfer.

- **M7: Pascal-Specific Backend** *(PLANNED)*
  - Architect execution backend optimized specifically around Pascal SM 6.1 memory hierarchy and compute pipelines.

- **M8: Architecture-Aware Compiler/Generator** *(PLANNED)*
  - Automated code generation and specialized kernel synthesis for legacy targets.

- **M9: Hardware/Model Co-Design (Optional)** *(PLANNED)*
  - Co-design quantization representations and architecture parameters tailored to legacy GPU limits.
