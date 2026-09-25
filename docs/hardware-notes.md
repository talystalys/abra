# Hardware Notes: Experimental Host `BigBoy`

Current inventory (2026-10-12): BigBoy now has two GTX 1080 Ti cards, each
with 11 GiB physical VRAM and compute capability 6.1. The host has 32 CPU
cores / 64 threads, 62.77 GiB RAM and kernel `6.12.107+deb13-amd64`.
See the [current inventory and raw evidence](research/2026-10-12-bigboy-refresh.md)
for device UUIDs, NUMA topology, toolchain and project checkout state.

The sections below preserve the historical M0 hardware inventory. The GTX
1060 and GTX 1660 SUPER listed there are no longer installed on BigBoy.

---

## 1. Host System Summary

| Component | Specification |
| :--- | :--- |
| **Hostname** | `BigBoy` |
| **OS** | Debian GNU/Linux 13 (trixie) / Debian 13.6 |
| **Kernel** | `6.12.100+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.100-1 (2026-07-30) x86_64` |
| **Processors** | 2 × Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz (Broadwell-EP) |
| **Physical Sockets** | 2 Sockets |
| **Cores / Threads** | 16 physical cores per socket (32 cores total, 64 logical threads with SMT) |
| **L1d Cache** | 1 MiB (32 instances × 32 KiB) |
| **L1i Cache** | 1 MiB (32 instances × 32 KiB) |
| **L2 Cache** | 8 MiB (32 instances × 256 KiB) |
| **L3 Cache** | 80 MiB (2 instances × 40 MiB shared per socket) |
| **System RAM** | 62 GiB DDR4 ECC (~64 GB physical) |
| **Swap Space** | 22 GiB |
| **NVIDIA Driver** | `550.163.01` |
| **CUDA Toolkit** | `CUDA 12.9.2` (Release 12.9, V12.9.86, compiler build `36037853_0`) |
| **Host Compiler** | `GCC 14.2.0` / `G++ 14.2.0` (Debian 14.2.0-19) |
| **C Standard Library** | `glibc 2.41` (`Debian GLIBC 2.41-12+deb13u3`) |

---

## 2. NUMA Topology

```text
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
node 0 size: 32072 MB (31.3 GiB)
node 1 cpus: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
node 1 size: 32206 MB (31.4 GiB)
node distances:
node   0   1
  0:  10  21
  1:  21  10
```

---

## 3. Physical GPU Devices

### 3.1 Primary Qualification GPU (Pascal SM 6.1)
- **CUDA Runtime Index**: `1` (Note: Runtime indices are not static contracts)
- **Device Name**: `NVIDIA GeForce GTX 1060 6GB`
- **Architecture**: Pascal
- **Compute Capability**: `6.1`
- **PCI Bus ID**: `0000:03:00.0`
- **PCI Device ID**: `10de:1b83` (GP104 variant, rev a1)
- **UUID**: `GPU-5078520c-bf92-4143-0d48-b02fd9d250dd`
- **Total Global VRAM**: 6144 MiB GDDR5 (reported ~6066 MiB usable by CUDA runtime)
- **Driver Version**: `550.163.01`
- **Power Limit**: 120.00 W
- **PCI Root Complex Affinity**: NUMA Node 0

### 3.2 Secondary Control GPU (Turing SM 7.5)
- **CUDA Runtime Index**: `0`
- **Device Name**: `NVIDIA GeForce GTX 1660 SUPER`
- **Architecture**: Turing
- **Compute Capability**: `7.5`
- **PCI Bus ID**: `0000:81:00.0`
- **PCI Device ID**: `10de:21c4` (TU116, rev a1)
- **UUID**: `GPU-8a043791-359d-1df4-9353-38c5bc98e5dd`
- **Total Global VRAM**: 6144 MiB GDDR6 (reported ~5930 MiB usable by CUDA runtime)
- **Driver Version**: `550.163.01`
- **Power Limit**: 125.00 W
- **PCI Root Complex Affinity**: NUMA Node 1

---

## 4. Profiler & Tooling Constraints

- **Nsight Compute Limitation**:
  Modern releases of NVIDIA Nsight Compute (e.g. 2024.x/2025.x shipped with CUDA 12.x) have deprecated and removed kernel profiling support for Pascal SM 6.x architectures. Attempting to profile SM 6.1 kernels with modern `ncu` results in unsupported architecture errors.
- **Milestone Handling**:
  This is a known tooling limitation. Per M0 guidelines, no historical profiler versions are installed during M0. Profiling methodology for Pascal will be formally addressed during Milestone M2 (DP4A Microkernel).
