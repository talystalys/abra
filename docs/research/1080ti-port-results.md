# Qwen3-4B port to BigBoy's GTX 1080 Ti GPUs

Session date: 2026-10-12. The raw commands and host clocks retain their actual
UTC timestamps. Branch: `port/1080ti-runtime`, based on M4 `815e3ff`.

## Hardware, source and workload

BigBoy has two GTX 1080 Ti cards, each CC 6.1 and 11 GiB physical VRAM:

| Role | PCI | UUID | NUMA | Existing power limit |
| --- | --- | --- | --- | --- |
| Primary | `0000:03:00.0` | `GPU-3007f159-e0cf-9303-0a81-41ea8e945315` | 0 | 250 W |
| Secondary | `0000:81:00.0` | `GPU-00279cb0-a4dc-86ec-7aa2-223d35376e6c` | 1 | 280 W |

The host has two Xeon E5-2683 v4 CPUs (32 cores, 64 threads), 62.77 GiB RAM,
Debian 13, driver 550.163.01 and CUDA 12.9 / nvcc V12.9.86. No clock, power,
driver or service settings were changed. Dynamic clocks and telemetry are
recorded in every final run. The cards are connected across CPU sockets;
their memory is separate. [Full hardware inventory](2026-10-12-bigboy-refresh.md).

Runtime source: pinned llama.cpp `169e4a7ff201e666c2325bcd35afb64573e39d09`.
The original checkout is the independent control. The modified checkout is
isolated at `/home/taly/legacy-ai-1080ti-port/.deps/llama.cpp-port` and built
for SM 6.1. The control's existing CUDA build includes SM 6.1 and SM 7.5.
The respective CMake caches, patches, custom headers and library hashes are
saved alongside measurements.

Canonical model: `Qwen3-4B-Q4_K_M.gguf`, 2,497,280,256 bytes, SHA256
`7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
Weights and precision remain unchanged. Final throughput conditions:
512 prompt tokens, 128 generated tokens, decode depths 0 and 2048, F16 K/V,
full 37/37 layer offload to one explicitly identified GPU, split mode none,
four CPU threads, batch 2048 and microbatch 512. Prompt and decode speeds are
separate measurements. These are controlled synthetic latency workloads,
not semantic quality evaluations.

Each final configuration uses two blocks in forward/reverse order, ten
measured samples per block, five excluded full-workload warmup samples per
block, and the upstream built-in warmup. Timed invocations omit verbose
logging; separate invocations prove full GPU offload. No benchmarks or builds
run concurrently with the final GPU timing work.

## Final primary-card throughput

Each row contains 20 measured samples; rates are tokens/s. These are aggregate
medians from balanced blocks with dynamic clocks, not a fixed-clock kernel
speedup. The primary card reported thermal slowdown during sustained runs;
that affects absolute throughput and the size of the measured relative gain.
Telemetry samples with at least 90% GPU utilization covered 66–90°C and
1556–1885 MHz SM clocks. The independent observation recorded active software
thermal slowdown. Small short-context gains are especially sensitive to this
temperature ramp; do not interpret the aggregate ratios as universal gains.

### Batch-one decode, 128 generated tokens

| Context depth | Configuration | Median | Min | Max | Spread | Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Baseline | 72.75 | 68.69 | 75.43 | 6.74 | 1.0000× |
| 0 | Exact QK/value | 76.61 | 72.81 | 78.34 | 5.53 | 1.0530× |
| 0 | Flash Attention | 79.03 | 76.74 | 79.90 | 3.16 | 1.0863× |
| 2048 | Baseline | 46.87 | 46.12 | 50.42 | 4.30 | 1.0000× |
| 2048 | Exact QK/value | 57.80 | 56.45 | 59.67 | 3.22 | 1.2333× |
| 2048 | Flash Attention | 71.15 | 68.28 | 72.75 | 4.47 | 1.5180× |

### Prompt processing, 512 prompt tokens

| Context depth | Configuration | Median | Min | Max | Spread | Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Baseline | 1634.47 | 1588.43 | 1655.70 | 67.27 | 1.0000× |
| 0 | Exact QK/value | 1630.99 | 1595.85 | 1640.55 | 44.70 | 0.9979× |
| 0 | Flash Attention | 1636.59 | 1632.02 | 1648.12 | 16.10 | 1.0013× |
| 2048 | Baseline | 1030.55 | 977.26 | 1051.38 | 74.11 | 1.0000× |
| 2048 | Exact QK/value | 1029.35 | 979.38 | 1046.75 | 67.37 | 0.9988× |
| 2048 | Flash Attention | 1154.79 | 1100.37 | 1162.31 | 61.94 | 1.1205× |

[Primary report and raw sample references](../../bench/results/1080ti-port/final-primary/report.json).

## Final secondary-card throughput and selected default

The secondary card delivered higher sustained Flash Attention throughput and
had narrower timing ranges. It is the default GPU in `qwen3-4b.py`; both cards
remain selectable by explicit UUID-backed role. Each row has 20 measured
samples under the same conditions as the primary run. Active telemetry covered
70–85°C and 1708–1898 MHz SM clocks. These card-to-card results reflect their
existing cooling and power limits, rather than a controlled hardware comparison.

| Context depth | Workload | Configuration | Median tok/s | Min | Max | Spread | Ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Decode 128 | Baseline | 75.41 | 73.81 | 75.70 | 1.88 | 1.0000× |
| 0 | Decode 128 | Flash Attention | 80.16 | 79.18 | 80.39 | 1.21 | 1.0630× |
| 2048 | Decode 128 | Baseline | 51.23 | 50.54 | 52.10 | 1.57 | 1.0000× |
| 2048 | Decode 128 | Flash Attention | 72.66 | 71.60 | 73.42 | 1.82 | 1.4185× |
| 0 | Prompt 512 | Baseline | 1675.22 | 1651.40 | 1694.65 | 43.25 | 1.0000× |
| 0 | Prompt 512 | Flash Attention | 1686.07 | 1672.54 | 1695.89 | 23.35 | 1.0065× |
| 2048 | Prompt 512 | Baseline | 1056.72 | 1053.19 | 1058.26 | 5.07 | 1.0000× |
| 2048 | Prompt 512 | Flash Attention | 1190.88 | 1187.54 | 1199.24 | 11.69 | 1.1270× |

[Secondary report and raw sample references](../../bench/results/1080ti-port/final-secondary/report.json).

The selected secondary-card configuration exceeds the declared >=25% decode
target at depth 2048: **+41.85%** with the quality qualification below. At short
context the measured improvement is **+6.30%**. Prompt processing is essentially
unchanged at depth 0 and improves **12.70%** at depth 2048. These results establish
the best tested configuration; further optimization remains possible.

## Correctness and quality

- The unchanged historical M4 oracle suite passed 756/756 cases on each
  current GTX 1080 Ti. This does not change the historical milestone verdict.
- The exact-order Q4_K/Q6_K, activation-sum and packed-metadata kernel family
  passed 2,025 independent cases on each GPU against pinned upstream and the
  existing F64 oracle. It is retained as experimental research; it is disabled
  in the default inference configuration.
- Final attention kernels passed 90 QK and 48 value cases per card, including
  padded strides, boundaries and large values. Outputs matched the original
  control bit for bit and passed the unchanged double oracle tolerance.
- QK/value runtime modes, combined and with graphs, produced bit-identical
  full-vocabulary logits at all 32 steps at depths 0 and 2048 on both cards.
  Each comparison checks 4,861,952 finite values plus exact greedy token IDs.
- With Flash Attention enabled in the reference as well, the patched fallback
  and Pascal graphs matched original upstream logits bit for bit at both
  depths on the primary GPU. This verifies that the disabled custom kernels
  do not change the selected upstream Flash Attention path.
- Upstream Flash Attention **failed** the frozen strict logit envelope in the
  first screen: maximum absolute difference 1.84059 despite matching the
  32-token greedy sequence. This failure remains recorded. No custom oracle
  tolerance was relaxed.

The user delegated the choice of attention policy for maximum performance
while maintaining quality. The separately declared quality criterion was
at most 1% mean perplexity regression on 20 WikiText-2 raw test chunks at
context 2048, plus no lost baseline successes on ten fixed deterministic
tasks. The corpus supplies 40,960 input tokens and 20,460 scored predictions
under this pinned perplexity implementation. The original corpus remains
outside Git; URL and SHA256 are in the result manifest.

Primary comparison: baseline PPL **12.3786 ± 0.29076**, Flash Attention
**12.3824 ± 0.29075**, a **+0.0307%** change. Both modes scored **8/10** on
strict task formatting, with no lost baseline success. The two common failures
were verbose math/code answers where the fixture required only an integer;
the code answer also reached the fixed 128-token limit. Individual outputs,
commands and failed checks are retained. The reported PPL uncertainty is the
tool's corpus uncertainty, not a confidence bound on the paired difference.
The secondary card reproduced the same PPL values and 8/10 scores, with no
lost baseline successes.
The rounded per-chunk cumulative PPL values give a paired mean log-loss change
of 0.0003069 and a standard error across chunks of 0.0011204; the derivation
is saved separately as `paired-chunk-analysis.json`.

An initial task run used an open assistant prefix and recorded 8/10 versus
7/10 because Flash Attention emitted an extra closing thought marker on an
otherwise correct extraction. Both modes were rerun with Qwen's official
`enable_thinking=false` prefix, retaining the same expected answers, scoring
and thresholds. [Official template](https://huggingface.co/Qwen/Qwen3-4B/blob/main/tokenizer_config.json).
The complete rerun is `flash-quality-primary-v3`; the earlier failure is
`flash-quality-primary-v2`. An earlier interrupted setup run is also retained:
a separate CLI smoke process used the model's large default context and caused
the first PPL allocation to fail. That process was stopped before rerunning.

This qualification supports the selected configuration on this corpus and
probe set. It does not prove equal quality for every task or longer context.
`--runtime exact` retains strict output equivalence for the validated workloads.

## Mechanisms and rejected experiments

The useful exact QK kernel preserves the two original warp reductions while
grouping four rows per CUDA block. Read-only restricted pointers let Pascal
emit `LDG.E.CI` loads and reuse queries and grouped K/V data through the read-only
cache. The original control's QK microbenchmark median was 0.162816 ms; the
four-row candidate was 0.074688 ms (50 measured event samples). End-to-end
throughput is measured separately. The value kernel also preserves upstream
accumulation/reduction order and groups two rows per block.

Recorded negative results:

- Direct M4 runtime transplant changed reduction order and failed the strict
  logits check (maximum difference 0.8658). The replacement preserves upstream
  per-lane partial-sum order before warp reduction.
- Nine exact Q4_K/Q6_K layout configurations and activation-sum variants
  regressed in screening. They are disabled by default.
- Lossless Q4_K metadata prepacking duplicated approximately 2 GB of weights
  for only about 0.7% exploratory improvement in the best screen. It is not
  selected. The cache research assumes immutable canonical weights; general
  writes through aliased views require additional invalidation qualification.
- Graph-only screening did not establish a gain. Pascal graph replay remains
  opt-in, preserving upstream compatibility/update guards.
- With Flash Attention already enabled, secondary-card screening also found
  no graph gain: depth 0 medians 80.740 → 80.532 tokens/s and depth 2048
  74.241 → 74.159 tokens/s, three exploratory samples per mode. Graphs remain
  disabled in the default configuration.
- QK grouping without read-only restricted loads regressed at long context.
  One-warp and address-only variants are archived with their source and logs.
- Row splitting failed model load because this pinned CUDA backend did not
  support split buffers for the selected model. No two-GPU throughput claim
  is made.

## Reproduce on BigBoy

The existing clean control checkout, canonical model and CUDA/glibc include
overlay are prerequisites. Setup does not delete or rebuild that checkout.

```bash
cd /home/taly/legacy-ai-1080ti-port
python3 tools/build-port.py --profile
python3 tools/run-port-correctness.py --out bench/results/new-oracles
python3 tools/validate-port-runtime.py \
  --out bench/results/new-exact-validation \
  --runtime-lib .deps/llama.cpp-port/build-port/bin --attention --attention-v
python3 tools/qualify-flash-attention.py --out bench/results/new-quality
python3 tools/run-port-experiments.py --stage final --attention-v --include-flash \
  --out bench/results/new-final
python3 tools/analyze-port-results.py bench/results/new-final \
  --out bench/results/new-final/report.json
python3 tools/qwen3-4b.py
python3 tools/qwen3-4b.py --gpu primary
python3 tools/qwen3-4b.py --gpu secondary --runtime exact
```

All inference tools disable diagnostic `LEGACY_AI_PROFILE`; the measured build
includes the diagnostic hook with `--profile` but the hook remains inactive.
`--completion` selects the noninteractive completion tool. The interactive
`llama-cli` is a chat program in this pinned revision; `--single-turn` exits
after a supplied prompt. Choose `--reasoning off` explicitly for the same
non-thinking prefix used in the quality probes, and include `/no_think` in the
user message as those probes do. Sampling settings retain upstream defaults
unless specified; the task qualification used `--temp 0 --seed 1`.

The rebuilt runtime libraries matched the final benchmark hashes exactly,
and all oracle suites passed again after the reproduction build. Launcher
checks exercised both interactive chat and noninteractive exact completion.
They also exposed a baseline model limitation: without `/no_think`, the
hard-disabled-thinking math prompt returned the incorrect `371` in both
baseline and Flash Attention modes. With the qualified prompt including
`/no_think`, Flash Attention returned `391`; exact completion returned `Paris`
for the capital fixture. Those initial failures are retained and are not
included as successful quality probes. See `launcher-commands.json` and the
associated raw outputs.

Final raw evidence is in `bench/results/1080ti-port/`, including manifests,
sample nanoseconds, source patches, kernel snapshots, logs and telemetry.
GPU binaries, models, corpus text and temporary full-vocabulary dumps remain
excluded from Git.

For a fresh checkout on this same host, create the prerequisite links before
building (the commands intentionally fail if those paths already exist):

```bash
mkdir -p .deps
ln -s /home/taly/legacy-ai/.deps/llama.cpp .deps/llama.cpp
ln -s /home/taly/legacy-ai/.cuda_compat .cuda_compat
ln -s /home/taly/legacy-ai/models models
python3 tools/build-port.py --profile
```
