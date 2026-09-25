# BigBoy connection, hardware and repository audit — 2026-10-12

Collected through public-key SSH to `taly@BigBoy` at `82.128.141.4`.
This inventory records the current machine and checkout. Historical M0–M4
measurements retain their original device identities and benchmark conditions.

## Project locations and source state

| Location | Observed state |
| --- | --- |
| GitHub | `https://github.com/talystalys/legacy-ai`; default branch `m0/reproducible-baseline`; published branches M0–M3 |
| Local working copy | `/home/taly/legacy-ai`; `m4/autotuned-kernel-family`, HEAD `815e3ff338b72e31c4a2ace6815776762469c3f3`; existing uncommitted September research |
| Local older copy | `/home/taly/legacy-ai-bigboy`; separate older checkout |
| BigBoy working copy | `/home/taly/legacy-ai`; clean tracked worktree on `exp/qwen38-27b`, HEAD `fb1d5f26162cf659ddc47b5752c257dff0bfa43a` (M3) |
| BigBoy M4 branch | `m4/autotuned-kernel-family` exists at `815e3ff` but is not checked out |

GitHub's branch listing did not contain M4 or `exp/qwen38-27b` at collection.
The host alias and direct IP use `~/.ssh/id_ed25519_bigboy` with
`IdentitiesOnly yes`; authentication succeeded in noninteractive batch mode.

## Current physical machine

| Component | Observed specification |
| --- | --- |
| Host / motherboard | `BigBoy` / `JINGS-X99-P8` |
| OS / kernel | Debian GNU/Linux 13 (trixie), `6.12.107+deb13-amd64` |
| CPU | 2 × Intel Xeon E5-2683 v4; 32 physical cores, 64 threads; 2.10 GHz nominal, 3.00 GHz reported maximum |
| NUMA | Two CPU/memory nodes; each GPU attached to a different node; inter-GPU topology `SYS` |
| RAM / swap | 62.77 GiB RAM / 22.33 GiB swap |
| Storage | PNY CS3030 1TB NVMe SSD, 931.5 GiB device; root filesystem approximately 893 GiB with 581 GiB available at capture |
| GPU at `0000:03:00.0` | GTX 1080 Ti, CC 6.1; UUID `GPU-3007f159-e0cf-9303-0a81-41ea8e945315`; 11264 MiB physical VRAM; NUMA 0; 250 W configured limit |
| GPU at `0000:81:00.0` | GTX 1080 Ti, CC 6.1; UUID `GPU-00279cb0-a4dc-86ec-7aa2-223d35376e6c`; 11264 MiB physical VRAM; NUMA 1; 280 W configured limit |
| PCIe | Both cards support Gen3 x16; idle query reported Gen1 x16 |
| NVIDIA driver | `550.163.01` |
| CUDA toolkit | `/usr/local/cuda-12.9`, nvcc release 12.9, V12.9.86 |
| Host tools | GCC 14.2.0, glibc 2.41, CMake 3.31.6, Ninja 1.12.1 |

Each card has 11 GiB physical VRAM. CUDA's usable memory is recorded separately
in the driver-property JSON. Both devices report CC 6.1, 28 multiprocessors,
a 352-bit memory bus and 2.75 MiB L2. The two memory allocations belong to
separate devices; their sum does not describe one GPU's memory capacity.

The original GTX 1060 and GTX 1660 SUPER are absent. Both current cards are
Pascal devices, so neither supplies the historical Turing control. Existing
device-specific qualification and tuning results need fresh correctness and
performance measurements before being applied to these cards.

## Existing build and model files

The pinned upstream dependency at `.deps/llama.cpp` remains at
`169e4a7ff201e666c2325bcd35afb64573e39d09`. Its existing benchmark binaries
are in `build-default/bin/llama-bench` and `build-cublas/bin/llama-bench`.
There is also `.deps/llama.cpp-qwen38/build/bin/llama-bench`.
The standard `.deps/llama.cpp/build/bin/llama-bench` path is absent.
Binary presence establishes availability only; their correctness and runtime
linkage were not evaluated in this inventory.

BigBoy's `models/` contains `Qwen3-4B-Q4_K_M.gguf` (2,497,280,256 bytes)
and `Qwen3.8-27B-UD-Q4_K_M.gguf` (16,464,440,224 bytes). These are observed
filenames and sizes; model metadata and file hashes were not checked.

M4 remains awaiting technical-lead review under the repository contract.
The next implementation decision must account for the remote experimental
branch and the replacement GPUs before selecting a benchmark or integration
task. This audit establishes no new performance result or milestone verdict.

## Evidence and reproduction

Raw outputs are saved under
[`docs/evidence/bigboy-refresh-2026-10-12`](../evidence/bigboy-refresh-2026-10-12/):

- `inventory.txt`: exact SSH command, fastfetch, hardware, tools and remote repository state.
- `cuda-device-properties.json`: CUDA driver property queries, without kernel launches.
- `repository-state.txt`: local state and live GitHub API responses.

The collection filename follows the session date (2026-10-12). The raw
inventory records the actual collection clocks: both local and remote report
2026-10-11 UTC. Preserve those timestamps when correlating run logs.

To repeat the driver-property inventory from the local project directory:

```bash
ssh -o BatchMode=yes taly@BigBoy 'python3 -' < tools/research-device-inventory.py
```
