# Further Qwen3-4B speed work

The user authorized implementation and testing on 2026-10-12 after the
research-only round. Scope: unchanged Qwen3-4B Q4_K_M weights, F16 KV, the
identified BigBoy GTX 1080 Ti pair and the pinned llama.cpp revision.

## Milestones

1. Add reproducible CPU/memory placement and NCCL protocol/channel screens.
   Compare each configuration in forward and reverse order, without other GPU
   work or compilation. Record negative outcomes and thermal variation.
2. Implement narrow, opt-in graph-local reuse of activation quantization.
   Reuse only an identical input across proven independent projections, retain
   upstream quantization and reduction math, and preserve the default fallback.
   Screen kernels on the actual tensor shard dimensions if evidence warrants it.
   The production trace also identified a tied-output name-classification
   candidate: use the duplicated output tensor's identity to apply existing
   vocabulary-row splitting, preserving the token lookup embedding placement.
3. Qualify candidates against an immutable snapshot of the current runtime:
   frozen full-vocabulary tolerance, greedy IDs, reset/growing-context checks,
   existing CPU kernel oracles and fixed quality fixtures as applicable. Never
   substitute task quality for a failed custom kernel correctness check.
4. Measure promising candidates with five excluded full-workload warmups and
   twenty measured samples per configuration, balanced forward/reverse blocks
   at context depths 0/2048/8192. Promote only reproducible, qualified gains;
   document limitations, failures, exact commands and raw evidence.

No clock, power, fan, driver or service changes. No precision reduction or
modified weights. Larger speculative-decoding changes require their own
implementation and correctness evidence; they are not assumed to yield gains
in this round. Estimated speed targets are not experimental results.

## Execution evidence

All four planned steps produced recorded evidence. Placement/protocol changes
and remapped tensor kernels were not selected. The implemented default uses
tied output row splitting plus graph-local activation reuse, with bit-identical
logits, unchanged fixed-corpus perplexity/task scores, completed reset soaks,
launcher checks and balanced final generation gains at all three depths.
Prompt processing changed little at 0/2K and +2.3% at 8K; thermal slowdown
remains. The copied-control library-path error was corrected and the complete
final matrix repeated with resolved-library assertions. See
[the measured report](research/1080ti-speed-results.md) for commands,
qualification, negative outcomes, raw samples and limitations.
