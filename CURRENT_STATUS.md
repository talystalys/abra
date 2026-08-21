# CURRENT_STATUS.md

Current milestone: M0
Current branch: m0/reproducible-baseline
Verified implementation commit: 9938befd49316b6ba9548cedf3346e3da2ed0502
(Git repository HEAD is authoritative for current commit state)

Execution host: BigBoy

Pascal qualification GPU: NVIDIA GeForce GTX 1060 6GB
Pascal PCI/UUID: PCI 0000:03:00.0 / UUID GPU-5078520c-bf92-4143-0d48-b02fd9d250dd
Pascal compute capability: 6.1

Control GPU: NVIDIA GeForce GTX 1660 SUPER
Control PCI/UUID: PCI 0000:81:00.0 / UUID GPU-8a043791-359d-1df4-9353-38c5bc98e5dd
Control compute capability: 7.5

NVIDIA driver: 550.163.01
CUDA toolkit: CUDA 12.9.2 (/usr/local/cuda-12.9)

What works:
- Live physical detection and device enumeration of both GTX 1060 6GB (Pascal SM 6.1) and GTX 1660 SUPER (Turing SM 7.5) on BigBoy
- Dual architecture fatbinary code generation targeting sm_61 and sm_75 simultaneously
- Canonical patch file applied cleanly via `patch` into `.cuda_compat` overlay without in-place modification of system headers
- Physical device pinning in `verify-m0.sh` selecting GTX 1060 and GTX 1660 SUPER by immutable UUID / PCI bus ID with strict identity assertions
- CPU oracle correctness verification across test dimensions {1, 31, 32, 33, 255, 256, 257, 1024, 65536} executed independently on both GPUs
- Micro-benchmark timing harness validation using CUDA events, conventional median calculation, and input validation
- Single-command automated verification flow via tools/verify-m0.sh including automated raw hardware and verification evidence generation

What does not work:
- Pascal SM 6.x profiling with modern Nsight Compute (known toolchain limitation on older architectures; to be addressed in M2)

Pascal physical execution verified: YES

Current meaningful benchmark: NONE — M0 contains no performance baseline.
Current blocker: None for M0.
Next required experiment: M1 baseline establishment (pending technical lead review and authorization).
