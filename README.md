# Legacy GPU AI Inference Research

Research project investigating recovery of modern quantized neural-network inference performance on legacy NVIDIA GPU architectures (specifically **NVIDIA Pascal compute capability 6.1 / `sm_61`**).

---

## Current work: Qwen3-4B on two GTX 1080 Ti GPUs

The authorized port runs in `/home/taly/legacy-ai-1080ti-port` on BigBoy,
branch `port/1080ti-runtime`. Both installed GPUs are Pascal SM 6.1 with
11 GiB physical VRAM. The user selected Qwen3-4B until extreme optimization.
See [the port milestones](docs/1080ti-port-plan.md),
[current inventory](docs/research/2026-10-12-bigboy-refresh.md), and
[current status](CURRENT_STATUS.md).

Reproduction tooling:

```bash
# Requires the existing, clean pinned control checkout and CUDA/glibc overlay.
# Fetch once: extract pinned NVIDIA packages locally, without apt/system changes.
python3 tools/fetch-nccl.py
python3 tools/build-port.py --profile --nccl-prefix .deps/nccl-2.27.7-cuda12
python3 tools/qwen3-4b.py -- --reasoning off --single-turn -p 'Explain memory bandwidth. /no_think' -n 128
# Original reduction order with exact-output QK/value kernels:
python3 tools/qwen3-4b.py --runtime exact --gpu secondary
```

The default launcher uses both identified GTX 1080 Ti GPUs, equal tensor
splitting, NCCL FP32 and Flash Attention, with unchanged Q4_K_M weights and
F16 KV. Balanced measurements show 72.18 → 94.10 tokens/s at 2K context
(+30.4%) and 60.79 → 82.53 at 8K (+35.8%), compared with single-GPU Flash
Attention. The declared quality comparison passed (+0.194% perplexity,
8/10 fixed tasks in both modes). Tensor logits differ from single-GPU logits;
NCCL matches the pinned tensor implementation bit for bit. Experimental
prompt lookup reached 293.02 tokens/s versus 72.04 on a complete repetitive
code-copy task (4.067×, identical tokens, 20 samples); general prose token
equality failed, so this stays separate from normal chat.

```bash
# Single-card fallback:
python3 tools/qwen3-4b.py --runtime flash --gpu secondary
# Experimental greedy lookup for repetitive code; prose equality failed:
python3 tools/qwen3-4b-lookup.py --gpu both --allreduce nccl --file prompt.txt
```

`--runtime exact` uses custom QK/value kernels that matched the original
single-GPU logits bit for bit. Diagnostic profiling is compiled into the
measured build but disabled by timing/inference tools. Raw results live in
`bench/results/1080ti-port/`. See [the new results and RAM tradeoffs](docs/research/1080ti-next-results.md),
[the next-round milestones](docs/1080ti-next-plan.md), and
[the original port report](docs/research/1080ti-port-results.md).

## Historical M0 setup

The following hardware roles and verification commands document the old
GTX 1060 / GTX 1660 SUPER qualification environment.

## Target Hardware & Roles (Host: `BigBoy`)

- **Primary Qualification GPU**: **NVIDIA GeForce GTX 1060 6GB**
  - Architecture: Pascal
  - Compute Capability: 6.1 (`sm_61`)
  - PCI Bus ID: `0000:03:00.0`
  - UUID: `GPU-5078520c-bf92-4143-0d48-b02fd9d250dd`
  - *Role*: Primary qualification target. Only physical execution on this GPU qualifies as Pascal execution evidence.

- **Secondary / Control GPU**: **NVIDIA GeForce GTX 1660 SUPER**
  - Architecture: Turing
  - Compute Capability: 7.5 (`sm_75`)
  - PCI Bus ID: `0000:81:00.0`
  - UUID: `GPU-8a043791-359d-1df4-9353-38c5bc98e5dd`
  - *Role*: Architecture control, cross-architecture correctness verification, and detecting accidental architecture assumptions.

---

## Toolchain & Environment

- **Host**: BigBoy (Debian GNU/Linux 13 / trixie, Kernel 6.12.100+deb13-amd64)
- **CPU**: 2 × Intel Xeon E5-2683 v4 @ 2.10GHz (64 logical CPUs, 2 NUMA nodes)
- **RAM**: 62 GiB
- **NVIDIA Driver**: `550.163.01`
- **CUDA Toolkit**: `CUDA 12.9.2` (`/usr/local/cuda-12.9`)
- **Host Compiler**: `GCC / G++ 14.2.0` (glibc 2.41)
- **Build System**: `CMake 3.31.6` / `Ninja 1.12.1`

---

## Quick Start & Verification

To run the complete automated Milestone M0 verification suite:

```bash
./tools/verify-m0.sh
```

To run individual tools:

```bash
# 1. Enumerate CUDA devices
./build/list_cuda_devices

# 2. Run CPU oracle correctness test on GTX 1060 (Pascal SM 6.1)
./build/smoke_test --target-arch sm_61

# 3. Run CPU oracle correctness test on GTX 1660 SUPER (Turing SM 7.5)
./build/smoke_test --target-arch sm_75

# 4. Run timing harness validation (Infrastructure validation only)
./build/timer_smoke --target-arch sm_61
```

---

## Directory Structure

```text
/
├── CMakeLists.txt              # Top-level CMake configuration
├── README.md                   # Project overview & quickstart
├── AGENTS.md                   # Strict agent & engineering contract
├── CURRENT_STATUS.md           # Live environment state & active blockers
├── MILESTONES.md               # Project roadmap (M0-M9)
├── .gitignore                  # Git exclusions
│
├── cmake/
│   └── LegacyGpuArch.cmake     # Architecture flags & CUDA compat overlay
│
├── patches/
│   └── cuda-12.9-glibc-2.41-math-functions.patch  # glibc 2.41 / CUDA 12.9 patch diff
│
├── docs/
│   ├── architecture.md         # Target GPU architecture overview
│   ├── benchmark-methodology.md# Strict benchmarking guidelines & protocols
│   ├── hardware-notes.md       # Live BigBoy hardware details & topology
│   └── reports/
│       └── m0-reproducible-environment.md  # Milestone M0 complete audit report
│
├── kernels/
│   └── smoke/
│       ├── smoke.h             # Deterministic smoke kernel declarations
│       └── smoke.cu            # Vector compute kernel & GPU execution logic
│
├── bench/
│   └── micro/
│       └── timer_smoke.cu      # CUDA event timing harness validation
│
├── tests/
│   └── smoke_test.cu           # CPU oracle correctness test suite
│
└── tools/
    ├── collect-hardware.sh     # Non-destructive hardware inventory script
    ├── list_cuda_devices.cu    # CUDA device query & enumeration utility
    ├── setup-cuda-compat.sh    # Deterministic glibc 2.41 compatibility setup
    └── verify-m0.sh            # End-to-end automated M0 verification flow
```
