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
