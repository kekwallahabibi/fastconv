---
name: rust-architecture
description: Use when designing new crates, domain types, CPU/GPU pipeline backends or larger refactorings in fastconv.
---

# Rust architecture (domain-first & laziness)

Run this checklist in order before writing implementation code.

## 1. Domain types first (`crates/domain`)
1. Model data, states and error enums completely in `crates/domain` — no I/O, FFmpeg, OS or GPU dependencies.
2. **Make illegal states unrepresentable:** replace combinable booleans and `Option` chains with narrow enums and validated constructors (e.g. `ExecutionPlan::GpuZeroCopy` never accepts CPU filters or audio).
3. Define explicit error enums for every forbidden state; no silent fallback as a substitute for a clean error type.

## 2. Get the type design approved
Before writing boundary or pipeline code, show the user: the proposed domain types and invariants; which illegal states become compile-time impossible; which crate owns which responsibility. Start implementation only after approval.

## 3. Isolate the system boundary (`crates/probe`)
1. Parse and validate external input (paths, `ffprobe` output, hardware capabilities, driver flags) exactly once in `crates/probe`.
2. Convert raw input straight into validated `fastconv-domain` types at the boundary.
3. No repeated defensive validation in `crates/pipeline`.

## 4. Concrete backends, no speculative traits (`crates/pipeline`)
1. Implement concrete modules first (`cpu`, `gpu::nvenc`, `gpu::vaapi`, `gpu::videotoolbox`) with an explicit `match` on `ExecutionPlan`.
2. **Laziness protocol:** no trait, generic, factory or wrapper layer for a single caller. Introduce an abstraction only when several concrete backends genuinely need it.
3. No silent CPU fallback: a failing GPU path fails with a clear, root-cause error.

## 5. Thin orchestration (`crates/cli`) & dependency check
1. `crates/cli` only parses the invocation, wires `probe` → `domain` → `pipeline` and maps errors to exit codes.
2. Check crate boundaries against `AGENTS.md`: `domain` has no workspace deps and zero I/O; `probe` and `pipeline` depend only on `fastconv-domain`, never on each other; `unsafe_code = "forbid"` stays the default — if FFI/GPU drivers later force `unsafe`, contain it minimally with a safety invariant in the module.
3. Minimize reader load: no comments restating code — document only the *why* of real driver/codec quirks.
