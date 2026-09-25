# GTX 1080 Ti port and inference performance plan

Authorization: on 2026-10-12 the user requested the hardware port, substantial
performance improvement, a milestone plan, and immediate execution. That
authorizes the sequence below, including runtime integration and testing.
Historical milestone verdicts and device evidence remain historical.

Working branch: `port/1080ti-runtime`, based on M4 commit `815e3ff`.
Local: `/home/taly/projects/legacy-ai-1080ti-port`.
BigBoy: `/home/taly/legacy-ai-1080ti-port`.
The earlier local research and BigBoy's 27B experimental checkout are retained.

## Milestones and acceptance criteria

| Step | Work | Required evidence |
| --- | --- | --- |
| P0 — Hardware port | Record both GTX 1080 Ti identities; create an isolated, reproducible SM 6.1 build; run the full existing M4 oracle suite on both cards | Model SHA256, exact toolchain, compiled device images, per-device correctness logs; no Turing-control claim |
| P1 — Fresh runtime baseline | Measure Qwen3-4B prompt processing and batch-one decode at depths 0 and 2048 | Raw repeated samples, UUID/NUMA binding, offload logs, telemetry, fixed precision and workload |
| P2 — Runtime integration | Diagnose launch overhead; integrate validated kernels into pinned llama.cpp with narrow dispatch conditions and upstream fallback | Same token stream and model; full-vocabulary logits comparison, exact greedy tokens, fallback checks; oracle suite preserved |
| P3 — Optimize measured costs | Evaluate CUDA Graph replay on Pascal, fresh kernel selection and Qwen3-4B placement | Alternating baseline/candidate runs, unchanged model/precision/context, recorded negative results; every shipped change independently justified |
| P4 — Deliver and reproduce | Provide runnable build/inference tools, source patches, matched measurements and updated status | Clean reproduction commands; raw logs; measured median/min/max/spread; all model/build provenance |

Target: at least **25% faster batch-one token generation** on a declared
workload against the fresh pinned upstream baseline on the same hardware.
This is an engineering target, not a promised result. Prompt processing must
be reported separately. Lower precision or a smaller model cannot count as
that speedup. Gains from configuration and from custom code are reported
individually. Measurements from the GTX 1060 cannot serve as the denominator.

The user selected **Qwen3-4B only until extreme optimizations are reached**.
The existing canonical Qwen3-4B-Q4_K_M permits direct oracle and
historical-source comparisons. All implementation and experiments in this
plan target that model. Subsequent optimization should address measured
launch gaps, quantization reuse, activation partial sums and weight layout,
with end-to-end validation and paired measurements for each mechanism.

## Initial experiments

1. Validate M4 on both actual devices with explicit name, CC, PCI and UUID.
2. Establish baseline with 512 prompt tokens, 128 decode tokens, F16 KV,
   unchanged Q4_K_M weights, GPU offload and explicit NUMA binding. Final
   performance claims require at least twenty measured samples plus five full-workload warmup samples per block and alternating runs.
3. Test Pascal CUDA Graph replay as an opt-in runtime change. Measure first
   capture and steady state separately and compare logits and token IDs.
4. Tune/integrate Q4_K and Q6_K only where independent measurements show a benefit.
   Unsupported types/shapes/batches retain upstream dispatch.
5. Compare Qwen3-4B placement and attention settings, then prioritize the
   largest remaining measured costs on the best validated configuration.

No GPU clock, power limit, driver or system service changes are part of this
plan. Existing correctness tolerances remain fixed. Scope follows observed
bottlenecks, and failed hypotheses are recorded as results.

## Execution notes

### User-delegated attention quality decision

The user requested maximum performance while maintaining quality and delegated
the choice of attention standard. Upstream Flash Attention may become the
default only after an explicit quality comparison; its strict logit failure
remains recorded. Custom kernels retain the frozen oracle and full-runtime
checks. Both attention modes use the same Q4_K_M weights and F16 KV cache.

Before measuring quality, define acceptance as no more than 1% relative
perplexity regression on the first 20 WikiText-2 raw test chunks at context
2048, and no lost baseline successes on ten fixed deterministic task probes.
Store corpus URL/hash, exact commands, raw outputs and individual scores.
This is a practical qualification for this workload, not proof of equivalent
quality on every task. The probe fixture and criterion must be frozen before
execution. A failure keeps the exact-output configuration as the default.

The first probe run used `/no_think` with an open assistant prefix and recorded
8/10 baseline versus 7/10 Flash Attention successes. The lost exact-format
success was an extra `</think>` before an otherwise correct extracted answer.
The corrected fixture applies Qwen's official `enable_thinking=false` prefix
(`<think>\n\n</think>\n\n`) to both modes. Answers, checks and thresholds remain
fixed; rerun the complete qualification and retain the initial failure.
Template source: [Qwen3-4B tokenizer configuration](https://huggingface.co/Qwen/Qwen3-4B/blob/main/tokenizer_config.json).

- P0: canonical model hash verified; an SM 6.1 image was freshly compiled;
  the unchanged M4 suite completed 756/756 tests on each GTX 1080 Ti.
- P2: graph replay and the final virtual-warp Q4_K/Q6_K integration produced
  bit-identical full-vocabulary logits and exact greedy token sequences for
  32 steps at both depths 0 and 2048 on the primary GPU.
- The first direct M4 transplant failed the frozen full-runtime logit
  envelope (maximum difference 0.8658). It reduced within each warp before
  combining warps; upstream combines per-lane warp partials first. Changed
  rounding can propagate through subsequent Q8 quantization. The replacement
  preserves the four upstream partial sums and reduction order, even when
  changing the number of physical warps.
- Initial graph screening did not establish a performance benefit. Final
  paired measurements exclude verbose per-token logging; verbose offload
  proof is collected in a separate untimed invocation. Early screen runs
  with different logging settings are exploratory and cannot support a
  matched performance claim.
- P1/P3: balanced 20-sample measurements completed on both cards, with five
  excluded full-workload warmups per block. At context depth 2048 the selected
  upstream Flash Attention configuration improved primary decode by 51.8%
  and secondary decode by 41.8% over each card's own fresh baseline. The
  primary showed a temperature ramp and thermal slowdown; report the raw
  telemetry and ranges with these results. The >=25% target was met for this
  declared workload. This does not establish a global performance optimum.
- P2/P3: exact custom QK/value kernels matched logits bit for bit on both
  cards, and improved primary long-context decode by 23.3% in the final run.
  All original and new oracle suites were rerun successfully on both cards.
- P4: isolated build, inference, quality, oracle, validation and benchmark
  tools plus raw results and source snapshots are supplied. Full findings:
  [port results](research/1080ti-port-results.md). Historical milestone PASS
  verdicts remain unchanged.
