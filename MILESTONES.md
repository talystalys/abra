# Project Milestones Roadmap

This document outlines the research and engineering roadmap for exploring modern quantized neural-network inference on legacy NVIDIA architectures (specifically NVIDIA Pascal SM 6.1).

> **Important**: Milestone M0 is the **only active milestone**. Future milestones require review, audit of evidence, and explicit authorization from the technical lead before implementation begins.

---

## Roadmap

- **M0: Reproducible Environment** *(ACTIVE)*
  - Establish trustworthy experimental host baseline on BigBoy.
  - Verify CUDA 12.9.2 toolchain and glibc 2.41 compatibility.
  - Verify dual code generation for `sm_61` (Pascal) and `sm_75` (Turing).
  - Physical device identification and separate correctness execution on GTX 1060 and GTX 1660 SUPER.
  - Validate CUDA event timing harness infrastructure.

- **M1: Existing Baseline** *(PLANNED)*
  - Quantify current baseline inference performance on legacy hardware with standard kernels / libraries.

- **M2: DP4A Microkernel** *(PLANNED)*
  - Implement and evaluate 8-bit vector dot product (`__dp4a`) arithmetic primitives on Pascal SM 6.1.
  - Address legacy profiler tooling constraints (Nsight Compute SM 6.x compatibility).

- **M3: Quantized Weight Microkernel** *(PLANNED)*
  - Implement specialized quantized weight kernels (Q4/Q5/Q8) with on-the-fly dequantization and INT32 accumulation.

- **M4: Autotuned Kernel Family** *(PLANNED)*
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
