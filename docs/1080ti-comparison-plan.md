# Qwen3-4B runtime comparison on BigBoy

Authorized by the user on 2026-10-20. This measures alternatives to running
Legacy AI, including Ollama. No performance outcome is assumed.

## Milestones

1. **Inventory and installation:** verify the two authorized GTX 1080 Ti
   identities; record driver/toolchain, idle processes, upstream revisions,
   artifact hashes, build logs and runtime library mappings. Use isolated
   installations and localhost ports, without changing system services or
   drivers. Record unsupported configurations, including silent CPU fallback.
2. **HTTP comparison:** official Ollama defaults and tuned F16/Flash Attention,
   clean stock llama.cpp at the port's pin and a current release, and the
   optimized Legacy AI runtime. Run each physical GPU separately and supported
   dual-GPU placements. Keep the previous port available as an attribution
   control. Same SHA-256 checked Qwen3-4B Q4_K_M GGUF in every case.
3. **Quality and stability:** deterministic fixed task probes, full responses
   for representative coding/reasoning/prose prompts and repeat consistency.
   Record differences rather than requiring numerical identity between
   independently versioned backends. Existing strict custom kernel tolerances
   remain unchanged. No claim of universal quality equivalence.
4. **Serving:** strongest compatible Ollama, current upstream and Legacy AI
   configurations each undergo 30 minutes total of continuous requests in two
   opposing 15-minute blocks, plus a separate concurrency sweep. The block
   design balances order because primary-card thermal drift confounded earlier
   sequential soaks; it is not a single uninterrupted 30-minute run.
5. **Publication:** preserve raw requests/responses, commands, timing samples,
   telemetry, failures and a reproducible report with narrowly scoped claims.

## Main measurement protocol

- Raw, explicitly rendered identical Qwen3 no-thinking prompts through local
  HTTP; verify tokenization/counts. Fixed weights and F16 KV for the main table.
- Prompt lengths approximately 512, 2048 and 8192 tokens, with recorded actual
  counts; context capacity leaves room for generation. Generate up to 128
  tokens; EOS is honored consistently and actual output length is reported.
- At least 20 measured requests per main condition, in forward/reverse blocks,
  with excluded warmups. Use a unique early document identifier for every
  fresh request, including warmups; alternating two texts is insufficient
  because current Ollama caches multiple prefixes. Verify identifier token
  lengths and record residual short-prefix hits. Reject long-prefix reuse.
- Report client first-content latency, total response latency, backend prefill
  and decode rates separately. Streaming chunk intervals are not token latency.
  Loading and server startup are recorded separately from warm requests.
- Fresh-prefill and repeated-history/cache experiments are separate. Ollama
  exposed cached-token counts and llama.cpp prompt counts are retained. For
  current Ollama, prefill rate uses total input count minus cached input count.
  Backend decode rate uses output count / generation duration; retain native
  llama.cpp rates separately because first-token accounting can differ.
- Dynamic clocks; no fan, power, clock or thermal changes. Sample temperature,
  SM clocks, power, utilization, GPU memory, host RSS/swap and active process
  mappings. Balanced order mitigates drift; sustained tests establish behavior
  once hot. Results from different thermal states are not interchangeable.
- Report median, min, max, spread, p95 request latency, all raw samples,
  warmup count and failure/timeout counts. CPU fallback is not a CUDA result.
- Identical weights alone do not establish identical quality. Small existing
  fixed probes and actual responses accompany comparisons; full-vocabulary
  bitwise equivalence remains scoped to prior port qualification.

## Compatibility

At planning time Ollama documentation requires driver 570+ for CC 5.0–6.2;
BigBoy has driver 550.163.01. Test current official binaries first. If CUDA
execution is unsupported, record it and assess an explicitly labeled compatible
older release or current-source CUDA 12 build. Do not attribute CPU fallback to
Pascal GPU performance. Freeze resolved versions rather than using moving tags
in reproduction instructions.
