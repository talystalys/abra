# AGENTS.md — Research & Implementation Contract

This repository governs research into recovering modern quantized AI inference performance on legacy NVIDIA GPUs. All AI agents, contributors, and automated systems operating in this repository must strictly adhere to the following contract.

---

## 1. Experimental Integrity & Canon

- **Repository Evidence is Canonical**: Only recorded, reproducible evidence committed to the repository or produced by repository tooling constitutes valid experimental results.
- **Never Invent Measurements**: Synthetic, simulated, estimated, or fabricated benchmark numbers or hardware metrics are strictly prohibited.
- **Never Silently Alter Benchmark Conditions**: Any change to problem dimensions, warm-up iterations, clock states, data types, memory layouts, or compilation flags must be explicitly documented and justified.
- **Never Remove Correctness Validation to Improve Speed**: Correctness checks and CPU/oracle verification must never be stripped, relaxed, or bypassed to artificially inflate performance numbers.
- **Negative Results are Legitimate**: An experiment that shows no speedup, regression, or failure to meet a performance hypothesis is a valuable scientific result.

> **Do not optimize the benchmark itself, silently change comparison conditions, remove correctness checks, or claim performance improvements without raw reproducible measurements.**

> **If the hypothesis fails, record the negative result. Do not keep changing unrelated architecture until a speedup appears.**

---

## 2. Hardware Identity & Architecture Demarcation

### Current authorized port environment (2026-10-12)

The user authorized the GTX 1080 Ti port and execution of the milestones in
`docs/1080ti-port-plan.md`. All new qualification and optimization evidence
must explicitly identify these current BigBoy devices:

| Role | Device | CC | PCI | UUID |
| --- | --- | --- | --- | --- |
| Primary | GTX 1080 Ti, 11 GiB | 6.1 | `0000:03:00.0` | `GPU-3007f159-e0cf-9303-0a81-41ea8e945315` |
| Second Pascal device | GTX 1080 Ti, 11 GiB | 6.1 | `0000:81:00.0` | `GPU-00279cb0-a4dc-86ec-7aa2-223d35376e6c` |

Both current devices are Pascal. There is no installed Turing control.
UUID, PCI, name and CC assertions remain mandatory; CUDA ordinal is not an
identity. The GPU identities below apply only to historical M0–M4 evidence.
Do not overwrite historical logs or reinterpret historical qualification.

### Historical qualification environment

- **Physical Machine**: All hardware experiments and qualification must execute on host `BigBoy`.
- **Primary Qualification GPU (Pascal SM 6.1)**:
  - Device: NVIDIA GeForce GTX 1060 6GB
  - Compute Capability: 6.1 (`sm_61`)
  - PCI Bus ID: `0000:03:00.0`
  - UUID: `GPU-5078520c-bf92-4143-0d48-b02fd9d250dd`
  - Only physical execution on this specific GPU qualifies as Pascal execution evidence.
- **Secondary / Control GPU (Turing SM 7.5)**:
  - Device: NVIDIA GeForce GTX 1660 SUPER
  - Compute Capability: 7.5 (`sm_75`)
  - PCI Bus ID: `0000:81:00.0`
  - UUID: `GPU-8a043791-359d-1df4-9353-38c5bc98e5dd`
  - Used strictly for control, cross-architecture correctness, and detecting architecture-specific assumptions.
- **Absolute Rule**:
  - **GTX 1660 SUPER results must NEVER be called Pascal results.**
  - **GTX 1060 results must explicitly identify the physical device (PCI bus ID / UUID / CC 6.1).**
  - **Never assume CUDA enumeration order (e.g. GPU 0 vs GPU 1). Always identify devices explicitly by name, compute capability, PCI bus ID, and UUID.**

---

## 3. Milestone & Scope Discipline

The current user instruction authorizes execution of P0–P4 in
`docs/1080ti-port-plan.md`, including correctness tests, runtime integration,
and measured optimization. It supersedes the previous requirement to wait
for M5 implementation authorization within this scope. Historical milestone
PASS verdicts remain unchanged; report current evidence and acceptance
criteria rather than inventing technical-lead approval.
Workload scope: Qwen3-4B only, per the user's follow-up.

On 2026-10-12 the user delegated the choice between strict logit equivalence
and upstream Flash Attention with maintained quality. The selected policy is
to qualify upstream Flash Attention separately using fixed-corpus perplexity
and deterministic task checks before selecting it for inference. The recorded
strict logit failure remains a failure; custom kernel oracle and exact-output
checks retain their existing tolerances. See the quality criteria in the port
plan. This policy does not authorize lower precision or changed model weights.

- **Active Milestone**: Only the currently authorized milestone may be implemented.
- **Milestone Authorization**: The technical lead determines PASS/PARTIAL/FAIL for each milestone. Agents must never declare milestone PASS or proceed to subsequent milestones without explicit authorization.
- **No Speculative Infrastructure**: Do not implement unrequested runtime, compiler, autotuner, or model infrastructure ahead of the roadmap.
- **No Broad Unrelated Rewrites**: Keep changes minimal, modular, and strictly aligned with the task at hand.

---

## 4. Security & Repository Hygiene

- **No Secrets**: Never commit or expose passwords, API keys, tokens, or credentials in any repository file.
- **No SSH Key Modifications**: Do not copy, expose, print, replace, or modify SSH private keys or authentication credentials.
- **No Model Weights in Git**: LLM weight files, large binary models, and large checkpoint files must never be committed to Git.
- **No Binary Blobs**: Do not commit build directories, object files, or intermediate compiler artifacts.

---

## 5. Performance Claim Standards

- Any performance claim must include:
  1. Exact host and GPU identifiers (PCI Bus ID, UUID, CC).
  2. Driver version and CUDA Toolkit release.
  3. Clock and power configuration where applicable.
  4. Precise workload definition (dimensions, precision, layout).
  5. Warm-up count, measurement iterations, median, min, max, and spread.
  6. Raw benchmark logs and commands used for reproduction.
