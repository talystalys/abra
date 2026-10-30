# Milestone M0 Report: Reproducible Experimental Environment

This report documents the verification of Milestone M0 for the legacy-NVIDIA-GPU AI inference research project on host `BigBoy`.

---

## 1. Repository Information

- **Starting Branch**: `m0/reproducible-baseline`
- **Starting Commit**: `unborn` (clean repository initialization)
- **Verified Implementation Commit**: `9938befd49316b6ba9548cedf3346e3da2ed0502`
- **Git Authority**: Git repository HEAD is authoritative for current commit state.

---

## 2. Host System Identity

- **Hostname**: `BigBoy`
- **Operating System**: Debian GNU/Linux 13 (trixie) / Debian 13.6 (`ID=debian`, `VERSION_ID="13"`)
- **Kernel Release**: Linux BigBoy 6.12.100+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.100-1 (2026-07-30) x86_64
- **Processors (CPU)**: 2 × Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz (32 physical cores, 64 logical threads)
- **System Memory (RAM)**: 62 GiB DDR4 (64,278 MiB physical, 22 GiB Swap)
- **NUMA Configuration**: 2 NUMA Nodes (Node 0: 32,072 MB, Node 1: 32,206 MB)

---

## 3. GPU Hardware Devices

### 3.1 Primary Qualification GPU (Pascal SM 6.1)
- **Device Name**: NVIDIA GeForce GTX 1060 6GB
- **Core Architecture**: Pascal
- **Compute Capability**: 6.1 (`sm_61`)
- **Hardware UUID**: `GPU-5078520c-bf92-4143-0d48-b02fd9d250dd`
- **PCI Bus ID**: `0000:03:00.0` (GP104 [10de:1b83] rev a1)
- **Total Global VRAM**: 6144 MiB GDDR5 (6066 MiB runtime usable)
- **Power Limit**: 120.00 W
- **Driver Version**: `550.163.01`

### 3.2 Secondary Control GPU (Turing SM 7.5)
- **Device Name**: NVIDIA GeForce GTX 1660 SUPER
- **Core Architecture**: Turing
- **Compute Capability**: 7.5 (`sm_75`)
- **Hardware UUID**: `GPU-8a043791-359d-1df4-9353-38c5bc98e5dd`
- **PCI Bus ID**: `0000:81:00.0` (TU116 [10de:21c4] rev a1)
- **Total Global VRAM**: 6144 MiB GDDR6 (5930 MiB runtime usable)
- **Power Limit**: 125.00 W
- **Driver Version**: `550.163.01`

---

## 4. CUDA Toolchain & Host Compiler

- **CUDA Toolkit Release**: CUDA 12.9.2
- **NVCC Compiler Version**: `Cuda compilation tools, release 12.9, V12.9.86` (Build `cuda_12.9.r12.9/compiler.36037853_0`)
- **Installation Path**: `/usr/local/cuda-12.9` (Symlinked as `/usr/local/cuda`)
- **Host Compiler**: GCC 14.2.0 / G++ 14.2.0 (Debian 14.2.0-19)
- **C Standard Library**: glibc 2.41 (`Debian GLIBC 2.41-12+deb13u3`)
- **Build System**: CMake 3.31.6, Ninja 1.12.1

---

## 5. Debian 13 / glibc 2.41 Compatibility Workaround

### 5.1 Observed Issue
When compiling CUDA code on Debian 13 with GCC 14.2.0 and glibc 2.41, standard glibc `<bits/mathcalls.h>` declarations for `sinpi`, `cospi`, `sinpif`, and `cospif` include `noexcept(true)` exception specifications:
```c
extern double cospi (double __x) noexcept (true);
extern double sinpi (double __x) noexcept (true);
extern float cospif (float __x) noexcept (true);
extern float sinpif (float __x) noexcept (true);
```
Unmodified CUDA 12.9.2 headers in `/usr/local/cuda-12.9/targets/x86_64-linux/include/crt/math_functions.h` declare these without exception specifications (`throw()`), triggering compiler errors during the `cudafe++` preprocessing phase.

### 5.2 Raw Initial Failure
```text
/usr/include/x86_64-linux-gnu/bits/mathcalls.h(79): error: exception specification is incompatible with that of previous function "cospi" (declared at line 2601 of /usr/local/cuda-12.9/bin/../targets/x86_64-linux/include/crt/math_functions.h)
   extern double cospi (double __x) noexcept (true); extern double __cospi (double __x) noexcept (true);
                                    ^

/usr/include/x86_64-linux-gnu/bits/mathcalls.h(81): error: exception specification is incompatible with that of previous function "sinpi" (declared at line 2556 of /usr/local/cuda-12.9/bin/../targets/x86_64-linux/include/crt/math_functions.h)
   extern double sinpi (double __x) noexcept (true); extern double __sinpi (double __x) noexcept (true);
                                    ^

/usr/include/x86_64-linux-gnu/bits/mathcalls.h(79): error: exception specification is incompatible with that of previous function "cospif" (declared at line 2623 of /usr/local/cuda-12.9/bin/../targets/x86_64-linux/include/crt/math_functions.h)
   extern float cospif (float __x) noexcept (true); extern float __cospif (float __x) noexcept (true);
                                   ^

/usr/include/x86_64-linux-gnu/bits/mathcalls.h(81): error: exception specification is incompatible with that of previous function "sinpif" (declared at line 2579 of /usr/local/cuda-12.9/bin/../targets/x86_64-linux/include/crt/math_functions.h)
   extern float sinpif (float __x) noexcept (true); extern float __sinpif (float __x) noexcept (true);
                                   ^

4 errors detected in compilation. Exit code: 2.
```

### 5.3 Canonical Patch Application
1. A minimal compatibility overlay is established in `.cuda_compat/include` via `tools/setup-cuda-compat.sh`.
2. System CUDA headers at `/usr/local/cuda-12.9` are never modified in place.
3. Pristine header copies are populated into `.cuda_compat/include/crt/` and the canonical committed patch file `patches/cuda-12.9-glibc-2.41-math-functions.patch` is applied via the `patch` tool (`patch -p1`).
4. CMake configuration prioritizes `.cuda_compat/include` via `-I${PROJECT_SOURCE_DIR}/.cuda_compat/include`.
5. Verification: NVCC compilation exits with code `0`, and all downstream binary builds and executions succeed cleanly.

---

## 6. Build Configuration & Binary Architecture Inspection

### 6.1 Configuration & Build Commands
```bash
# Configure
CUDAFLAGS="-I/home/taly/abra/.cuda_compat/include -Wno-deprecated-gpu-targets" \
cmake -S /home/taly/abra -B /home/taly/abra/build \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build /home/taly/abra/build --parallel
```

### 6.2 Binary Inspection Evidence (`cuobjdump`)
Command:
```bash
/usr/local/cuda-12.9/bin/cuobjdump -lelf build/smoke_test
```
Output:
```text
ELF file    1: smoke_test.1.sm_61.cubin
ELF file    2: smoke_test.2.sm_75.cubin
ELF file    3: smoke_test.3.sm_61.cubin
ELF file    4: smoke_test.4.sm_75.cubin
```
- `sm_61` present: **YES**
- `sm_75` present: **YES**

---

## 7. Numerical Correctness & Dual-GPU Execution (Pinned Identity)

### 7.1 Test Run A: NVIDIA GeForce GTX 1060 6GB (Pascal SM 6.1)
- **Targeting Method**: Immutable UUID `GPU-5078520c-bf92-4143-0d48-b02fd9d250dd`
- **Identity Assertions**:
  - Name: `NVIDIA GeForce GTX 1060 6GB` (Verified)
  - Compute Capability: `6.1` (Verified)
  - PCI Bus ID: `0000:03:00.0` (Verified)
  - UUID: `GPU-5078520c-bf92-4143-0d48-b02fd9d250dd` (Verified)
- **Results Across Test Dimensions**:

| Dimension | Divisible by Block Size (256) | Mismatches | Status |
| :--- | :--- | :--- | :--- |
| `1` | NO | 0 | **PASS** |
| `31` | NO | 0 | **PASS** |
| `32` | NO | 0 | **PASS** |
| `33` | NO | 0 | **PASS** |
| `255` | NO | 0 | **PASS** |
| `256` | YES | 0 | **PASS** |
| `257` | NO | 0 | **PASS** |
| `1024` | YES | 0 | **PASS** |
| `65536` | YES | 0 | **PASS** |

**GTX 1060 Result**: **PASS**

### 7.2 Test Run B: NVIDIA GeForce GTX 1660 SUPER (Turing SM 7.5 Control)
- **Targeting Method**: Immutable UUID `GPU-8a043791-359d-1df4-9353-38c5bc98e5dd`
- **Identity Assertions**:
  - Name: `NVIDIA GeForce GTX 1660 SUPER` (Verified)
  - Compute Capability: `7.5` (Verified)
  - PCI Bus ID: `0000:81:00.0` (Verified)
  - UUID: `GPU-8a043791-359d-1df4-9353-38c5bc98e5dd` (Verified)
- **Results Across Test Dimensions**:

| Dimension | Divisible by Block Size (256) | Mismatches | Status |
| :--- | :--- | :--- | :--- |
| `1` | NO | 0 | **PASS** |
| `31` | NO | 0 | **PASS** |
| `32` | NO | 0 | **PASS** |
| `33` | NO | 0 | **PASS** |
| `255` | NO | 0 | **PASS** |
| `256` | YES | 0 | **PASS** |
| `257` | NO | 0 | **PASS** |
| `1024` | YES | 0 | **PASS** |
| `65536` | YES | 0 | **PASS** |

**GTX 1660 SUPER Result**: **PASS**

---

## 8. Timing Harness Infrastructure Validation

> **Notice**: These timings are infrastructure validation and are not performance claims. No performance baseline or optimization claims exist in Milestone M0.

- **Command**: `./build/timer_smoke --uuid GPU-5078520c-bf92-4143-0d48-b02fd9d250dd --assert-uuid GPU-5078520c-bf92-4143-0d48-b02fd9d250dd --assert-pci 0000:03:00.0 --assert-name "GeForce GTX 1060 6GB" --assert-cc 6.1 --iterations 20 --warmup 5`
- **Selected GPU**: NVIDIA GeForce GTX 1060 6GB (PCI `0000:03:00.0`, CC `6.1`, UUID `GPU-5078520c-bf92-4143-0d48-b02fd9d250dd`)
- **Workload**: 1,048,576 int32 elements (12 MiB active data)
- **Warm-up**: 5 iterations
- **Measured Trials**: 20 iterations
- **Median Formula**: Conventional statistical median (mean of two middle values for even sample count $N=20$)
- **Numerical Mismatches**: `0`

---

## 9. Pascal Qualification Gate

All qualification criteria evaluated on host `BigBoy`:
1. Physical GTX 1060 detected: **YES**
2. Compute capability verified as 6.1: **YES**
3. `sm_61` binary artifact verified via `cuobjdump`: **YES**
4. Smoke kernel physically executed on GTX 1060: **YES**
5. CPU oracle and GPU results agree across all dimensions: **YES**

```text
Pascal physical execution verified: YES
```

---

## 10. Raw Verification Artifacts

The complete unedited raw hardware inventory and verification suite logs are preserved in the repository at:
- `docs/evidence/hardware-inventory-raw.txt`
- `docs/evidence/m0-verification-raw.txt`
