# Further Qwen3-4B speed work on BigBoy

The user authorized implementation and testing on 2026-10-12 after the
research-only round. This campaign compares new changes against an immutable
snapshot of the previously selected two-card runtime, rather than comparing
results from different thermal conditions or multiplying earlier gains.

## Implementation

**Split the tied output projection.** Qwen3-4B has no separate `output.weight`
in this GGUF. Its duplicated vocabulary projection retains the name
`token_embd.weight`. The pinned tensor placement callback recognizes output
weights by name and therefore mirrors this projection on both cards. The
new opt-in patch recognizes the original output tensor by pointer identity
and uses the existing vocabulary-row split. It leaves the lookup embedding
placement unchanged. Guards require Qwen3, 36 layers, two devices, Q6_K and
the canonical 2560 x 151936 shape. No dot-product arithmetic changes.

**Reuse activation quantization within a graph.** Independent projections
of the same unchanged F32 input can consume the same upstream Q8_1 bytes.
The cache only accepts contiguous, single-token, 2560-element inputs and
Q4_K/Q6_K model weights on SM 6.1. It uses tensor and storage identity,
invalidates overlapping writes, covers fused destinations, and ends at each
graph boundary. It falls back during CUDA graph capture, concurrent branches,
other streams, other shapes or other devices. Quantization and weight
dot-product implementations remain upstream. This changes neither model
quantization nor KV precision.

Source installers are [prepare-output-split.py](../../tools/prepare-output-split.py)
and [prepare-q8-reuse.py](../../tools/prepare-q8-reuse.py). The host cache is in
[abra-q8-reuse.cuh](../../kernels/q4k/abra-q8-reuse.cuh).

## Hardware and workload

All execution was on BigBoy, Debian 13, kernel 6.12.107+, dual Xeon E5-2683 v4,
32 physical cores / 64 threads, approximately 62.77 GiB RAM. Driver
550.163.01; CUDA 12.9, nvcc 12.9.86; SM 6.1 build. No clock, power, fan,
driver or service settings changed. Clocks remain dynamic; thermal variation
limits extrapolation to other machines or cooling conditions.

| Role | Device | CC | PCI | UUID | Existing power limit |
| --- | --- | --- | --- | --- | --- |
| Primary | GTX 1080 Ti, 11 GiB | 6.1 | 0000:03:00.0 | GPU-3007f159-e0cf-9303-0a81-41ea8e945315 | 250 W |
| Secondary | GTX 1080 Ti, 11 GiB | 6.1 | 0000:81:00.0 | GPU-00279cb0-a4dc-86ec-7aa2-223d35376e6c | 280 W |

Unchanged model: `/home/taly/abra/models/Qwen3-4B-Q4_K_M.gguf`,
2,497,280,256 bytes, SHA256
`7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
Upstream revision: `169e4a7ff201e666c2325bcd35afb64573e39d09`.
Frozen starting project revision: `80040d28149ee1788c7dbb450a7ee1692ff456e8`.
Both runtimes use equal tensor splitting, all 37 model/output layers on GPU,
NCCL 2.27.7 FP32, upstream Flash Attention and F16 KV; four CPU/batch threads,
batch 2048, microbatch 512, CPU and memory on NUMA node 0.
Custom Q4_K/Q6_K, prepacking and CUDA graphs remain disabled in the selected
candidate. No CPU weight/KV offloading or speculative draft model is used.

The immutable control worktree is `.deps/llama.cpp-speed-control`.
[Its snapshot](../../bench/results/1080ti-speed/control-snapshot/manifest.json)
records original executable/library/header hashes and the original source
diff. Each benchmark records its own model, runtime, source, NCCL and header
hashes, command, explicit environment, hardware, toolkit and telemetry.

## Final balanced measurements

Each configuration/depth has 20 measured samples in forward/reverse blocks
of ten, with five excluded full-workload warmups per block. The workloads are
tg128 generation and pp512 prompt processing, measured separately, at prefilled
depths 0/2048/8192. These are llama-bench token rates, not network streaming
latency or a continuously growing conversation. All 12 rows contain both
blocks, one immutable model hash, and explicit resolved-library checks.

### Generation

| Depth | Runtime | Median tokens/s | Min | Max | Spread | Change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Frozen control | 105.423 | 103.145 | 106.690 | 3.545 | reference |
| 0 | Output split + reuse | 115.374 | 113.346 | 115.867 | 2.521 | +9.44% |
| 2048 | Frozen control | 94.228 | 89.628 | 98.090 | 8.462 | reference |
| 2048 | Output split + reuse | 102.488 | 99.273 | 105.772 | 6.500 | +8.77% |
| 8192 | Frozen control | 80.602 | 79.555 | 82.177 | 2.623 | reference |
| 8192 | Output split + reuse | 86.905 | 85.557 | 90.037 | 4.480 | +7.82% |

### Prompt processing

| Depth | Runtime | Median tokens/s | Min | Max | Spread | Change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Frozen control | 2362.183 | 2350.427 | 2373.944 | 23.517 | reference |
| 0 | Output split + reuse | 2363.157 | 2353.037 | 2375.897 | 22.860 | +0.04% |
| 2048 | Frozen control | 1806.550 | 1765.115 | 1815.390 | 50.275 | reference |
| 2048 | Output split + reuse | 1807.709 | 1757.667 | 1821.716 | 64.049 | +0.06% |
| 8192 | Frozen control | 1021.547 | 583.452 | 1076.837 | 493.385 | reference |
| 8192 | Output split + reuse | 1044.704 | 738.102 | 1088.234 | 350.132 | +2.27% |

Generation improved 9.44%, 8.77% and 7.82% under these recorded conditions.
Prompt medians changed +0.04%, +0.06% and +2.27%. The main gain is generation;
these results do not imply the same gain for batched service workloads.
No samples were discarded beyond the declared warmups.

The primary's sampled temperatures were 59-91 C and SM clocks 734-1936 MHz;
the secondary's were 53-81 C and 1531-1923 MHz. Extrema include initialization,
excluded warmups and both workloads. Dynamic clocks and thermal slowdown
remain material limits; a balanced order does not reproduce fixed clocks.
This corrected matrix started after launcher/quality/profile checks and an
idle interval, rather than immediately after the long soak. Its absolute
rates must not be compared against the hotter first attempt as a new gain.

At depth 0, each card's sampled peak allocation fell from 1905 to 1753 MiB (152 MiB saved).
At depth 2048, each card's sampled peak allocation fell from 2049 to 1897 MiB (152 MiB saved).
At depth 8192, each card's sampled peak allocation fell from 2481 to 2329 MiB (152 MiB saved).
These are allocation peaks for the recorded workloads, not maximum context
capacity measurements.

[Corrected report with all samples, telemetry ranges and directories](../../bench/results/1080ti-speed/final-libraries/report.json).
Each block contains raw.json, manifest.json, telemetry.csv, proof logs and
runtime-dependencies.txt. The manifests record actual resolved library paths
and hashes; llama/ggml libraries outside the selected snapshot are rejected.

The first attempt remains under `final/`. Its copied control executable
retained an absolute build RUNPATH and loaded candidate libraries with both
new features disabled. It is a candidate-fallback comparison, not evidence
against the frozen original runtime, and its speed figures are superseded.
See the [library selection audit](../../bench/results/1080ti-speed/library-selection-audit.json).
Quality/logit/soak/profile runners already selected their libraries explicitly
and are unaffected. No cross-campaign ratio is used to claim this improvement.

## Correctness and quality

At context depths 0, 2048 and 8192, each candidate was compared against the
frozen tensor/FA runtime for 32 greedy steps over all 151,936 logits.
Output splitting alone, activation reuse alone and their combination were
bit-identical, with identical token IDs and prompt lengths. The selected
combination checks 14,585,856 logits across these depths. Frozen tolerance
remains `abs(error) <= 0.001 + 0.0001 * abs(reference)`; bit equality is an
additional result, not a relaxed criterion. See
[comparisons](../../bench/results/1080ti-speed/output-logits/summary.json).

Both cards completed 756 original Q4_K oracle cases, 2835 extended Q4_K/Q6_K
runtime cases, 90 attention cases, 48 value cases and 42 new activation reuse
output checks: 7542 cases across both cards. The activation suite tests mixed
Q4/Q6 consumers, independent writes, views, partial/full aliases, different
descriptors, graph resets and capture/concurrency fallback, with bit equality
to upstream. Existing double-oracle envelopes were unchanged. See
[commands and identities](../../bench/results/1080ti-speed/final-oracles/commands.json).

The fixed 20-chunk WikiText-2 comparison at context 2048 measured
**12.4064 +/- 0.29175 in both runtimes**, zero reported relative change.
All ten probe answers matched; the exact-format scorer retained **8/10**
in both modes and lost no baseline successes. The two existing failures
remain recorded. See [quality results](../../bench/results/1080ti-speed/final-quality/quality.json).
These fixtures do not establish quality equivalence on every task.

The actual launcher context capacity of 4096 was also tested with all ten
fixed probes: identical answers, 8/10 in both modes, no lost successes. The
chat/completion launcher suite passed all eight checks, including optimized
chat, optimized completion, `--unoptimized`, single-card Flash Attention,
the existing experimental lookup paths and invalid arguments/output paths.
See [4K probes](../../bench/results/1080ti-speed/context4096-quality/quality.json)
and [launcher smokes](../../bench/results/1080ti-speed/launcher-smokes/report.json).
The default is optimized tensor mode; `--unoptimized` disables both new flags.
The launcher checks that the required features exist in the built libraries
and gives a rebuild instruction for old binaries. Other modes set both new
flags to zero.

Longer greedy decoding also matched all IDs and repeat outputs: complete
531-token code copy, complete 531-token code edit, 256-token prose and
256-token reasoning, three repetitions per case and runtime. The driver
checks every vocabulary logit for finiteness and grows the context during
each request. No draft tokens were proposed. See
[decoding results](../../bench/results/1080ti-speed/full-decoding/report.json).
These runs are correctness checks with no excluded warmups; their timings
are not the final performance comparison.

The reset soak completed **100 requests / 25,600 generated tokens per
runtime**, checking finite full-vocabulary logits and matching all token IDs
across runtimes and repetitions. Sampled swap was zero. Sampled RSS did not
increase across the run; post-initialization maxima were 1,145,560 KiB for
the control and 1,145,568 KiB for the candidate. Each card's sampled peak
allocation was 1907 MiB versus 1755 MiB, a 152 MiB reduction per card for this
specific workload. See [soak audit](../../bench/results/1080ti-speed/soak/audit.json).
This is a sequential reliability test, not a balanced sustained speed claim:
its medians were 86.078 and 80.384 tokens/s, respectively. The primary reached
91 C in both runs; its candidate-run sampled maximum clock was 1556 MHz
versus 1847 MHz during the control run. The candidate ran second, after more
heat accumulated. No cooling or clock settings were changed. The final
balanced comparison below is the performance evidence.

## Screens and negative results

The timing screens using copied control binaries also loaded the candidate
with new features disabled; they are exploratory comparisons against that
fallback, not the frozen original. Placement/protocol screens used the current
build directly and are unaffected. The incorrect copied-control library
metadata remains preserved; see the [library audit](../../bench/results/1080ti-speed/library-selection-audit.json).

Screens used two excluded full-workload warmups and three measured samples
per forward/reverse block: six samples per configuration, tg128 at depth
2048. They are exploratory selection evidence. Medians, min/max/spread,
raw samples and explicit commands are recorded in each linked report.

- [Placement/NCCL screen](../../bench/results/1080ti-speed/communication-screen/report.json):
  control 98.844 tokens/s; both CPU sockets 98.002; interleaved RAM 98.608;
  ignore affinity 98.756; LL one channel 98.737; LL two channels 98.504;
  Simple one channel 87.794. No placement/protocol change was selected.
  Simple with SHM CUDA copy mode stalled twice during initialization. One
  proof timed out after 300 seconds; the repeated stalled experiment was
  stopped explicitly, with its exact PID/command/reason recorded. No GPU
  reset or service change was used.
- [Output screen](../../bench/results/1080ti-speed/output-screen/report.json):
  matched baseline 98.742, output split 105.478, output plus reuse 106.495
  tokens/s. These six-sample results motivated the final measurement.
- [Reuse screen](../../bench/results/1080ti-speed/reuse-screen-v3/report.json):
  baseline 99.247 versus reuse 99.988 tokens/s. Its benefit alone was small.
- [Actual tensor-shape kernels](../../bench/results/1080ti-speed/tensor-screen/report.json):
  output plus reuse was faster than every tested remapped Q4/Q6 variant.
  WPR 2 / WPB 8 passed strict full-logit checks, and the CPU oracle shape set
  gained K=2048/4864 while retaining previous cases. WPR 1 variants were
  screened but were not promoted or claimed to have full-runtime qualification.
- CUDA graphs passed the initial bit-equality checks but showed no material
  gain in the initial screen. They remain disabled. That screen was later
  interrupted by a logging assertion, so it is not a final graph benchmark.

Earlier failed test/screen attempts are retained. The first activation test
read a nonblocking CUDA stream through an unsynchronized default-stream host
copy; the test was corrected to synchronize its own stream, without changing
production math. The first cache guard required WEIGHTS buffer usage, while
the meta backend's model shards retain ANY usage. It consequently did not
activate in the full model. Corrected runs require an actual reuse dispatch
proof. Another screen incorrectly expected info logs from quiet timed
llama-bench runs; activation is now checked in a separate untimed verbose
proof. None of those inactive or failed runs qualifies the selected cache.

CUDA profiles are diagnostics only: profiling overhead is not a throughput
result. The first trace started only the current CUDA context; the corrected
[control trace](../../bench/results/1080ti-speed/production-control-v2/manifest.json)
starts profiling in both device contexts. The directory named `shape-profile`
contains a timeline run, as recorded in its manifest, rather than a
synchronous shape diagnostic.

The optimized diagnostic trace confirms eight 75,968-row output launches
per device, versus eight 151,936-row launches per device in the frozen
control. No full-width output launch remains in the candidate trace. See
[control projection audit](../../bench/results/1080ti-speed/control-output-profile.json)
and [candidate projection audit](../../bench/results/1080ti-speed/optimized-output-profile.json).
These audits read the CSV's declared duration units; nvprof selected ms for
the control and us for the candidate. Their timings are diagnostic only.

Rerunning both new source installers left all five affected files
byte-identical; see [idempotence evidence](../../bench/results/1080ti-speed/installer-idempotence.json).
Python compilation and `git diff --check` completed without errors.
The [final evidence audit](../../bench/results/1080ti-speed/final-audit.json)
checks actual library paths/hashes, all 12 matrix rows, frozen control identity,
strict logits, quality, repeated requests, oracle logs and launcher results.
Its runnable source is preserved beside the audit.

## Reproduction

Run from `/home/taly/abra-1080ti-port` on BigBoy. The original clean pinned
checkout, CUDA/glibc overlay, canonical model and extracted NCCL package are
required. Preserve/build the original runtime from starting project revision
80040d2 before building the candidate; the recorded control source diff and
header texts describe that snapshot.

```bash
python3 tools/build-port.py --reuse-q8 --split-tied-output --profile \
  --nccl-prefix .deps/nccl-2.27.7-cuda12 --jobs 8
python3 tools/run-port-correctness.py --reuse-q8 --out NEW_ORACLE_DIRECTORY
GGML_CUDA_ALLREDUCE=nccl python3 tools/validate-port-runtime.py \
  --out NEW_LOGIT_DIRECTORY --runtime-lib .deps/llama.cpp-port/build-port/bin \
  --baseline-lib .deps/llama.cpp-speed-control/build-port/bin \
  --gpu both --reference-split-mode tensor --reference-flash-attn \
  --reuse-q8 --split-tied-output --no-graphs --depths 0 2048 8192 --steps 32
python3 tools/qualify-flash-attention.py --out NEW_QUALITY_DIRECTORY \
  --gpu secondary --source .deps/llama.cpp-port \
  --baseline-runtime-bin .deps/llama.cpp-speed-control/build-port/bin \
  --baseline-split tensor --baseline-flash on --candidate-split tensor \
  --allreduce nccl --reuse-q8 --split-tied-output
python3 tools/lookup-experiments.py --out NEW_DECODING_DIRECTORY \
  --stage quality --configs tensor-control,tensor-output-reuse \
  --baseline-runtime-bin .deps/llama.cpp-speed-control/build-port/bin \
  --allreduce nccl --complete-code --cases code-copy,code-edit,prose,reasoning \
  --reps 3 --require-identical
python3 tools/lookup-experiments.py --out NEW_SOAK_DIRECTORY \
  --stage quality --configs tensor-control,tensor-output-reuse \
  --baseline-runtime-bin .deps/llama.cpp-speed-control/build-port/bin \
  --allreduce nccl --cases prose --reps 100 --require-identical
python3 tools/measure-speed.py --out NEW_MEASUREMENT_DIRECTORY \
  --final --variants split-output-reuse
python3 tools/smoke-next-launchers.py --out NEW_LAUNCHER_DIRECTORY
python3 tools/qualify-flash-attention.py --out NEW_4K_PROBE_DIRECTORY \
  --gpu secondary --source .deps/llama.cpp-port \
  --baseline-runtime-bin .deps/llama.cpp-speed-control/build-port/bin \
  --baseline-split tensor --baseline-flash on --candidate-split tensor \
  --allreduce nccl --reuse-q8 --split-tied-output --context 4096 --probes-only
```

Use fresh output directories. Complete per-run commands and environments,
including earlier screens, are under `bench/results/1080ti-speed/`.
The benchmark/qualification runner versions used before launcher integration
are preserved in `runner-sources/`, with their hashes in the run manifests.
Implementation and results are present in the local port checkout and on
BigBoy; this turn did not commit, push, or open a pull request.
