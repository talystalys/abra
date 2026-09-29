# Next Qwen3-4B optimization round

Authorized by the user's request to research, implement and test further
speed improvements and RAM placement. Same BigBoy identities, driver,
toolchain and immutable Q4_K_M weights as the completed first port.

## Milestones

1. Research and screen: profile the selected Flash Attention runtime; compare
   F16/Q8/Q4 KV, CPU KV, partial CPU layers, thread counts and tensor splitting.
2. Implement: a reproducible prompt lookup decoding path using the same target
   model, explicit KV rollback, greedy verification and ordinary decoding
   fallback. No additional model or synthetic acceptance.
3. Qualify: retain frozen kernel/logit checks. Quantized cache candidates must
   pass the existing 20-chunk WikiText perplexity criterion (at most 1% relative
   regression) and lose no baseline successes on the ten fixed tasks. Prompt
   lookup must match baseline token IDs on fixed workloads, including ordinary
   prose and code editing/copying; disclose any numerical mismatch.
4. Measure and ship: compare successful candidates against the current F16
   Flash Attention baseline, with identical workloads, five excluded full
   workload warmups and 20 measured samples for final llama-bench claims.
   Record per-token or per-verification-block latency arrays and distinguish
   them explicitly, GPU peak memory, host RSS, and sustained runs. Promote
   only qualified improvements; preserve regressions and capacity tradeoffs.

## Hypotheses and boundaries

- Quantized KV may reduce attention traffic at long context, but adds
  quantization arithmetic and changes cache precision. Screen before adoption.
- CPU placement can save VRAM; repeated PCIe transfers and Xeon computation
  may reduce speed. Measure rather than assume a benefit. Automatic Unified
  Memory spilling is deferred unless explicit placement shows a useful need.
- Prompt lookup can batch target verification for repeated code/text with
  no extra weights. Acceptance and numerical differences may limit its benefit.
- Two GPU tensor splitting might parallelize dense operations, but the SYS
  topology across sockets may dominate latency. Independent replicas are a
  throughput option, not a single-request latency improvement.
- No clock, power, fan, driver or service changes. No reduced model precision,
  dropped attention history, or undocumented benchmark condition changes.

## Primary research

- [NVIDIA Pascal tuning guide](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html)
- [NVIDIA memory oversubscription study](https://developer.nvidia.com/blog/improving-gpu-memory-oversubscription-performance/)
- [FlexGen](https://arxiv.org/abs/2303.06865): throughput-oriented placement and batching.
- [Upstream speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md):
  n-gram methods require no draft model; trained EAGLE/DFlash/DSpark heads
  require additional weights and are outside this round's one-model scope.

Research checked 2026-10-12. Implementations remain pinned to
`169e4a7ff201e666c2325bcd35afb64573e39d09`; master documentation may change.

## Additional screened options

- NCCL 2.27.x with CUDA 12.9 is extracted beneath `.deps`, with NVIDIA
  package SHA256 checks and no system package or driver changes. Localize
  its compilation to the communication translation unit, then compare
  explicit NCCL versus the original butterfly reduction. Pascal reductions
  remain FP32 for every tensor size; upstream's large-tensor BF16 heuristic
  is not appropriate for a same-precision comparison. Full oracle and
  quality evidence remain required.
- [Pinned upstream multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/169e4a7ff201e666c2325bcd35afb64573e39d09/docs/multi-gpu.md)
  describes tensor splitting, NCCL, and its current restrictions.
- [Speculative decoding paper](https://arxiv.org/abs/2211.17192) motivates
  batching target verification while preserving the target distribution.
  This implementation is greedy only and tests actual token identities.
- [KIVI](https://arxiv.org/abs/2402.02750) motivates asymmetric KV layouts
  as future work. The current Q8/Q4 screen uses existing GGML cache types,
  not KIVI's algorithm; it does not validate that paper on Pascal/Qwen3.

Exploratory placement and lookup screens briefly ran on separate GPUs and
NUMA nodes concurrently. Their short measurements select candidates only.
Final throughput measurements run without other GPU work or compilation.

## Screen findings and decisions

- F16 single-GPU Flash Attention remains the comparison baseline.
- CPU KV and partial CPU layers reduced speed at both screened contexts;
  quantized KV reduced memory but also speed. Do not adopt them for latency.
- Tensor split improved both 2K and 8K screens. Its frozen single-GPU logit
  comparison failed (32 greedy tokens matched). Qualify this upstream
  placement numerically through the existing fixed quality criteria, while
  preserving the failed strict comparison and all custom kernel tolerances.
- Lookup matched both 512-token code prefixes but diverged on prose and
  reasoning. Under the predeclared exact-token criterion it remains an
  explicit research feature. Test full code completion before reporting
  a complete-task gain; do not call it universally exact or make it default.
- Backend sampling alone showed no material screen gain. The first combined
  lookup/backend run failed because upstream defaults to one backend sample
  per sequence; setting the actual verification-block output limit fixed it.

Final telemetry still shows primary thermal slowdown. After the equal-split
comparison, screen 4:5, 2:3 and 3:5 tensor proportions against a fresh 1:1
control at depth 2048. Hypothesis: moving some work to the cooler secondary
may improve sustained speed without changing clocks or model precision.
Promote only if the screen gain exceeds 5%, then repeat balanced final
measurements and full quality qualification for that explicit ratio.

## Current qualification evidence

NCCL FP32 matched pinned tensor logits bit for bit at depths 0/2048/8192
(32 greedy steps each), and both tensor collective variants completed the
20-chunk corpus and all ten probes. PPL 12.3824 to 12.4064 (+0.193823%),
8/10 successes in both modes, no lost baseline successes. The single-GPU
strict failure and general lookup token mismatches remain failures.

## Final workload refinements

The full 24-function copy and edit screen completed 531 generated tokens
rather than stopping at the old 512-token cap, and both matched baseline
IDs and requested complete text. A balanced final copy comparison covers
single-GPU, tensor and tensor-plus-lookup modes (five excluded runs per
block, 20 measured runs total per mode). Editing retains its explicitly
labeled three-sample exploratory result; no final editing speed claim is
made. General lookup still failed its predeclared token criterion.

Latency placement tests reuse one short prose prompt for eight requests per
mode. A separate 100-request tensor soak clears the context between requests;
it tests repeatability and allocation behavior, not continuously growing
context or production reliability. Host RSS/swap is sampled from `/proc`
at 250 ms alongside GPU telemetry. Frozen oracle suites and actual launcher
smokes run after serialized timing, without other GPU work or compilation.

The completed copy/probe comparisons show identical IDs on the tested code
and ten short fixtures, while the separate prose/reasoning failures remain.
The supplemental decoding analyzer retains literal task matching and also
reports the unchanged existing qualification scorer; the two scores are
labeled separately. An additional probes-only check at the launcher's actual
4096 context capacity completed after the timing/soak queue with 8/10 in
both modes and no lost successes. It supplements
the original 2048-capacity corpus qualification and does not replace its result.

## Completed scope

Research screens, explicit RAM/cache characterization, NCCL FP32 integration,
quality qualification, balanced tensor/code-copy timings, short repeated
soaks and original regression suites are recorded in the results. Tensor
NCCL is selected for ordinary inference; general lookup remains experimental
and RAM/cache candidates remain unqualified for promotion. Historical
milestone PASS verdicts are unchanged. Physical cooling, further NCCL tuning,
fused kernels and a RAM streaming engine remain future research.
