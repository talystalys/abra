# Target Hardware Architectures & System Topology

This document details the architectural characteristics, compute capabilities, memory hierarchies, and project roles of the GPU architectures present on experimental host `BigBoy`.

---

## 1. Primary Qualification GPU: NVIDIA Pascal (`sm_61`)

### 1.1 Device Identity
- **Marketing Name**: NVIDIA GeForce GTX 1060 6GB
- **Core Architecture**: Pascal
- **Silicon Implementation**: GP106 / GP104 variant (GP104 [10de:1b83] rev a1 on BigBoy)
- **Compute Capability**: 6.1 (`sm_61`)
- **PCI Bus ID**: `0000:03:00.0`
- **Device UUID**: `GPU-5078520c-bf92-4143-0d48-b02fd9d250dd`
- **Total VRAM**: 6144 MiB GDDR5 (192-bit bus, ~192 GB/s theoretical bandwidth)
- **Power Limit**: 120.00 W

### 1.2 Microarchitectural Structure
- **Streaming Multiprocessors (SMs)**: 10 SMs (1280 FP32 CUDA cores total, 128 cores per SM).
- **Sub-core Partitioning**: Each Pascal SM is split into two processing blocks (64 cores each), with warp schedulers issuing up to 2 instructions per warp per cycle.
- **Register File**: 64K 32-bit registers per SM (256 KiB register file per SM).
- **Shared Memory / L1 Cache**: 48 KiB dedicated shared memory per SM + separate L1 cache.
- **L2 Cache**: 1536 KiB dedicated L2 cache.
- **Tensor Cores**: None (Tensor Cores were introduced in Volta SM 7.0 and Turing SM 7.5).
- **Integer Arithmetic & DP4A**:
  - Pascal SM 6.1 introduced native hardware acceleration for 8-bit integer vector dot products via the `DP4A` instruction (`__dp4a`).
  - `DP4A` calculates: $\text{acc} + (a_0 b_0 + a_1 b_1 + a_2 b_2 + a_3 b_3)$ where inputs are packed 4-element 8-bit integers and accumulation occurs in 32-bit integer registers.
  - Theoretical INT8 compute throughput with DP4A is 4× higher than single-precision FP32 ALU throughput.

### 1.3 Project Role
The GTX 1060 6GB is the **sole qualification GPU** for the research project. All qualification gates, Pascal microkernels, DP4A primitives, quantized weight kernels, and legacy execution strategies must physically execute and be verified on this hardware.

---

## 2. Secondary / Control GPU: NVIDIA Turing (`sm_75`)

### 2.1 Device Identity
- **Marketing Name**: NVIDIA GeForce GTX 1660 SUPER
- **Core Architecture**: Turing
- **Silicon Implementation**: TU116 [10de:21c4] (rev a1 on BigBoy)
- **Compute Capability**: 7.5 (`sm_75`)
- **PCI Bus ID**: `0000:81:00.0`
- **Device UUID**: `GPU-8a043791-359d-1df4-9353-38c5bc98e5dd`
- **Total VRAM**: 6144 MiB GDDR6 (192-bit bus, ~336 GB/s theoretical bandwidth)
- **Power Limit**: 125.00 W

### 2.2 Microarchitectural Differences vs Pascal
- **Concurrent FP32 and INT32 Pipelines**: Turing allows simultaneous execution of integer addressing/control instructions and floating-point math instructions.
- **Unified L1 Data Cache & Shared Memory**: Configurable up to 64 KiB shared memory per SM combined with high-bandwidth unified L1 caching.
- **Tensor Cores**: Turing architecture introduces Tensor Cores (though TU116 focuses primarily on concurrent integer/FP pipelines without RT cores).
- **Instruction Issue & Warp Scheduling**: 1 warp scheduler per sub-core partition (4 per SM) with single-instruction dual-dispatch capabilities.

### 2.3 Project Role
The GTX 1660 SUPER serves as an **architectural control**. It is utilized for:
1. Validating multi-architecture fatbinary compilation (`sm_61` + `sm_75`).
2. Ensuring test suites identify and execute across non-homogeneous GPU topologies.
3. Catching unintended architecture-specific bugs or hardcoded assumptions.
4. **Strict Boundary**: Performance numbers from the GTX 1660 SUPER are never considered Pascal performance results.

---

## 3. Host System Topology & Interconnect

- **Host Processor**: Dual-socket Intel Xeon E5-2683 v4 @ 2.10GHz (Broadwell-EP, 14nm).
- **Total Logical Cores**: 64 (32 physical cores across 2 sockets, 2 threads per core).
- **NUMA Topology**: 2 NUMA nodes (Node 0: 32 GB, Node 1: 32 GB).
- **PCIe Interconnect**:
  - GPU 0 (GTX 1660 SUPER): Attached to NUMA Node 1 PCIe root complex at `0000:81:00.0`.
  - GPU 1 (GTX 1060 6GB): Attached to NUMA Node 0 PCIe root complex at `0000:03:00.0`.
- **System Memory**: 62 GiB DDR4 ECC.
