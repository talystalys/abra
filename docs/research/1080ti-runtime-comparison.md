# Qwen3-4B: Abra, Ollama and stock llama.cpp

Status: measurements in progress. No comparative speed conclusion yet.

## Scope

This campaign answers which runtime is useful on the existing BigBoy machine.
It compares identical Qwen3-4B Q4_K_M weights through local HTTP, with F16 KV,
matched raw prompts and deterministic greedy settings. The earlier 7.8–9.4%
gain was against the previous port; it was not an Ollama comparison.

[Authorized plan](../1080ti-comparison-plan.md) ·
[Raw campaign](../../bench/results/1080ti-comparison/campaign/) ·
[Installation provenance](../../bench/results/1080ti-comparison/setup/provenance.json) ·
[Preliminary and superseded runs](../../bench/results/1080ti-comparison/validation-history.json)

## Hardware and software

All inference executes on BigBoy, dual Xeon E5-2683 v4, 32 physical CPU cores,
64 threads, approximately 62.77 GiB RAM, two NUMA nodes. No driver, service,
clock, fan or power changes were made. Locally compiled llama.cpp/Legacy
builds target SM 6.1 with CUDA 12.9.86 and the existing local CUDA/glibc header
overlay. Official Ollama binaries retain their bundled libraries, identified
by release and loaded-file hashes. Driver: 550.163.01.

| GPU | PCI | UUID | CC | VRAM |
| --- | --- | --- | --- | --- |
| Primary | `0000:03:00.0` | `GPU-3007f159-e0cf-9303-0a81-41ea8e945315` | 6.1 | 11 GiB |
| Secondary | `0000:81:00.0` | `GPU-00279cb0-a4dc-86ec-7aa2-223d35376e6c` | 6.1 | 11 GiB |

The cards communicate across CPU sockets; there is no NVLink. Dynamic clocks
and primary-card thermal throttling are part of the measured environment.
A read-only diagnostic during the Legacy primary-card matrix block recorded
`SW Thermal Slowdown: Active` at 91 C, with hardware slowdown inactive.
[Snapshot and command](../../bench/results/1080ti-comparison/setup/thermal-snapshot-during-legacy-primary.txt).

| Runtime | Frozen version | Backend and settings |
| --- | --- | --- |
| Current official Ollama | 0.40.6 | Vulkan on this driver; defaults and tuned settings |
| Older compatible official Ollama | 0.11.10 | CUDA; defaults and tuned settings |
| Clean stock llama.cpp, current stable release | v0.7.0, `620d2225cc436ab0b001e3200e590e3b286ef9ab` | CUDA, Flash Attention, clean sources; NCCL screened for two GPUs |
| Clean stock llama.cpp, port revision | `169e4a7ff201e666c2325bcd35afb64573e39d09` | Same build/toolkit controls; NCCL screened for two GPUs |
| Abra | Same pinned base plus recorded port patches | CUDA, Flash Attention, NCCL FP32, tied-output splitting and Q8 activation reuse |
| Previous port settings | Same port binary with both new optimizations disabled | Attribution control for the two latest changes |

Current Ollama's CUDA-only attempt allocated no model on a GPU and was rejected
as CPU fallback. The official documentation requires driver 570+ for CC
5.0–6.2; the existing driver is 550. The current official Vulkan path did
load all model layers. These results do not establish current Ollama CUDA
performance on a newer driver. See [Ollama's hardware requirements](https://docs.ollama.com/gpu).

Both stock source trees remain clean. Stock NCCL retains upstream arithmetic;
Abra's prior Pascal patch uses FP32 collectives. Comparators are not
assumed to produce bit-identical logits. Fixed-corpus and task checks accompany
timing results; existing custom-kernel tolerances were not changed.

The unchanged GGUF is 2,497,280,256 bytes, SHA-256:

```text
7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5
```

Ollama imports that exact content-addressed file. No registry model with an
unverified quantization is substituted. No weights or executable binaries are
stored in the repository.

## Protocol

- One physical GPU at a time, then supported dual-GPU placements. Each process
  is bound to its local CPU/RAM NUMA node; dual-GPU processes use node 0.
- Context capacity 10,240 tokens per request slot. Main requests contain exactly
  512, 2,048 or 8,192 input tokens and request up to 128 output tokens; EOS is
  honored and actual counts are retained.
- Raw identical Qwen3 no-thinking prompts, temperature 0, seed 0 and repetition
  penalty 1. “Ollama defaults” means default runtime tuning under this matched
  workload, not an unchanged `ollama run` chat session or registry model tag.
- Tuned Ollama requests Flash Attention, F16 KV, four CPU threads and batch
  2,048; the dual-GPU tuned case requests scheduler spreading. Actual GPU
  allocation and backend logs are retained. Stock/Legacy use four CPU threads,
  batch 2,048 and microbatch 512. Stock retains default graph behavior; the
  previously qualified Legacy profile disables CUDA graphs.
- Each main condition has 20 measured requests: ten in each opposing-order
  block, with two excluded full-workload warmups per depth/block.
- Every fresh request has a unique early delimited document identifier.
  1,000 identifiers have the same 11-token length; the first 24 full prompts
  per depth were independently verified. Short common template prefixes can
  still be cached; exposed cache counts are recorded and long-prefix hits are
  rejected. [Tokenization proof](../../bench/results/1080ti-comparison/workloads-v3/tokenization-proof.json).
- Client TTFT means arrival of the first nonempty content chunk. Client total
  latency includes prompt processing, generation, HTTP and stream consumption.
  Streaming chunks are not assumed to represent individual token latencies.
- Backend generation rate is API output count divided by API generation
  duration. Native llama.cpp rates are retained separately because first-token
  accounting can differ. Current Ollama prefill rate subtracts exposed cached
  input tokens from its total input count. Client latency is independent of
  these backend timing conventions.
- Approximately 1 Hz telemetry records temperature, SM clocks, board power,
  GPU allocation, process-tree RSS/swap and host load. Request and telemetry
  monotonic timestamps permit alignment. Sampled extrema are not guarantees
  about unsampled instants.
- Ten fixed tasks are repeated twice for every main configuration, with three
  additional full-response tasks. Fixed WikiText perplexity covers 20 chunks
  for the selected llama.cpp arithmetic/placement configurations. Ollama has
  no equivalent corpus-PPL API in this suite.
- Finalists are chosen by the lowest sum of median full-request latencies at
  the three input lengths, separately for Ollama, current upstream and Legacy.
  Cached-history and concurrency-2/4 results are separate from fresh requests.
- Sustained tests use two opposing 15-minute blocks per finalist: 30 minutes
  total per runtime. These are not a single uninterrupted 30-minute session.
- Separate natural short-prompt tests measure complete coding, reasoning and
  prose responses with a 1,024-token budget, EOS honored, 20 measured requests
  per task/runtime, unique early reference IDs and balanced order. Different
  output lengths are reported rather than treated as equal work.

## Validation discoveries

The preliminary alternating-prompt screen exposed an important cache issue:
current Ollama retained both complete prefixes. Repeating either made almost
all 2,048 input tokens cache hits. Those timings are not final fresh-prefill
results. Unique identifiers replaced that protocol before the main matrix.

An initial numeric identifier design was rejected because token merges made
its lengths vary. Delimiting individual digits fixed the shape and passed the
recorded tokenization checks. Initial client validation also encountered an
occupied port; the harness now rejects occupied ports and holds an exclusive
campaign lock. Superseded evidence is preserved and labeled.

Both stock row-split attempts failed model loading with “device CUDA0 does
not support split buffers” under these exact builds. Layer and tensor modes
ran; stock NCCL candidates were additionally screened. These failures are
scoped to the recorded configurations, not all possible Pascal software stacks.

The first partial matrix was interrupted because its process had loaded the
runner before the disjoint input-length identifier ranges were installed.
Those timings are preserved under `matrix-superseded-shared-identifiers` and
excluded. A further partial stock-only attempt is preserved under
`matrix-superseded-logging-preflight`: final configuration review disabled
Ollama diagnostic logging before final Ollama timing samples were collected.
Final timing runs use `OLLAMA_DEBUG=0`; qualification used diagnostic logging. The completed quality and corpus checks were reused; all timing
conditions were restarted with the corrected runner.

## Quality checks

All 22 configurations scored 8/10 on the fixed task set in both repetitions.
The same eight question IDs succeeded in every configuration. This is ten
questions repeated twice, not twenty independent questions.
The Legacy reference's two retained failures are formatting: it computes 36
but includes an explanation, and writes `No.` with punctuation. The frozen
exact-format checks still count both as failures.

| Configuration class | WikiText PPL | Reported uncertainty |
| --- | ---: | ---: |
| Single GPU: clean pinned, clean current, Legacy | 12.3824 | 0.29075 |
| Two GPUs: clean pinned/current stock NCCL | 12.3871 | 0.29104 |
| Two GPUs: Legacy and previous-port control | 12.4064 | 0.29175 |

The two latest Legacy optimizations retain the previous port's corpus result.
The separate Ollama APIs were not evaluated with corpus perplexity here.
These small corpus/task checks do not establish universal quality equivalence
or bit-identical behavior across backends.

All 30 stock/Legacy full-response repeat pairs matched. Fifteen of 36 Ollama
pairs differed; all responses ended before the 1,024-token cap. The repeat
requests used each API's cache behavior: current Ollama's changed prose
response coincided with 30 of 31 input tokens cached, versus five on the first
request. Legacy/stock requests explicitly disabled prompt reuse. This is a
comparison across execution/cache paths, not proof of random output or a
quality regression. Fresh timing requests use unique IDs to prevent long
prefix reuse.

## Measurements

Pending completion of the campaign and evidence audit.

## Reproduction

Requires the previously prepared pinned port checkout, canonical GGUF, CUDA
12.9 header overlay and locally extracted NCCL 2.27.7 CUDA-12 package. All
commands run on BigBoy from `/home/taly/abra-1080ti-port` with no other
benchmark active. Each output directory must be new.

```bash
bash tools/provision-comparison.sh bench/results/1080ti-comparison-repro/setup
bash tools/build-comparison-nccl.sh bench/results/1080ti-comparison-repro/setup
python3 tools/record-comparison-setup.py \
  --out bench/results/1080ti-comparison-repro/setup/provenance.json
python3 tools/compare-runtimes.py --stage workloads --configs legacy-secondary \
  --out bench/results/1080ti-comparison-repro/workloads-v3 --blocks 1 --fail-fast
```

The first campaign's preliminary screen used alternating prompts; its Ollama
prefill timings are superseded. Reproduction uses the corrected unique-ID
protocol for the screen as well as the final matrix:

```bash
python3 tools/screen-runtime-placements.py \
  --out bench/results/1080ti-comparison-repro/placement-screen \
  --workloads bench/results/1080ti-comparison-repro/workloads-v3/workloads.json
```

Then run the final campaign and audit:

```bash
python3 tools/run-runtime-comparison.py \
  --out bench/results/1080ti-comparison-repro/campaign \
  --screen bench/results/1080ti-comparison-repro/placement-screen \
  --workloads bench/results/1080ti-comparison-repro/workloads-v3/workloads.json
python3 tools/analyze-runtime-comparison.py bench/results/1080ti-comparison-repro/campaign/matrix
python3 tools/audit-runtime-comparison.py bench/results/1080ti-comparison-repro/campaign
```

Optional figures use an isolated local plotting environment and the recorded
aggregate; they never substitute generated values for missing measurements:

```bash
python3 -m venv /tmp/abra-comparison-plots
/tmp/abra-comparison-plots/bin/pip install matplotlib==3.10.7
/tmp/abra-comparison-plots/bin/python tools/plot-runtime-comparison.py \
  bench/results/1080ti-comparison-repro/campaign/matrix \
  --out bench/results/1080ti-comparison-repro/campaign/plots
```

## Limits

These measurements cover one GGUF, two specific Pascal cards, one host and one
driver. Short fixed probes and 20 corpus chunks do not establish universal
quality equivalence. Synthetic context bodies isolate input length; natural
short-prompt tests and sustained windows answer different questions. API
backend timing conventions can differ; full client latencies are retained.
Two GPUs can improve one request while consuming more hardware and power;
independent per-GPU replica serving is not measured here. No vLLM,
TensorRT-LLM or different-quantization results are implied.
