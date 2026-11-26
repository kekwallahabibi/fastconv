# fastconv — engineering rules

`fastconv` is a Rust CPU/GPU media converter. Keep the architecture flat, explicit, provably correct.

## Crate routing & allowed workspace deps

| Crate | Path | Role | Allowed workspace deps |
| :--- | :--- | :--- | :--- |
| `fastconv-domain` | `crates/domain` | Pure data models, invariants, error enums (no I/O, GPU or FFmpeg) | none |
| `fastconv-probe` | `crates/probe` | System boundary: parse and validate external input (files, media headers, HW detection) | `fastconv-domain` |
| `fastconv-pipeline` | `crates/pipeline` | Concrete CPU/GPU backends for validated `ExecutionPlan`s | `fastconv-domain` |
| `fastconv-cli` (`fastconv`) | `crates/cli` | Thin binary orchestrating `probe` → `domain` → `pipeline` | `fastconv-domain`, `fastconv-probe`, `fastconv-pipeline` |
| `xtask` | `xtask` | Local automation (`cargo xtask verify`, `gen-fixtures`, `bench`) | `fastconv-domain`, `fastconv-probe` |

## Hard invariants

1. **Not an FFmpeg wrapper.** Never spawn `ffmpeg`/`ffprobe` in `crates/`. `ffmpeg` is allowed only in `xtask` as an external benchmark opponent and for synthetic video fixtures — to measure the baseline our native Rust + CPU/GPU path must beat.
2. **Make illegal states unrepresentable.** Use narrow enums and validated constructors instead of stacked booleans/`Option`s (e.g. `ExecutionPlan::GpuZeroCopy` must reject CPU filters and audio).
3. **Boundary discipline.** Parse and validate external input (CLI args, filesystem, media headers, driver/system APIs) exactly once in `crates/probe`. From `crates/domain` onward only valid types exist.
4. **Laziness protocol.** No trait, wrapper or layer for a single caller; no speculative hooks. Comment only the *why* of real driver/codec/OS quirks.
5. **Prove it works.** A task is done only when `cargo xtask verify` is fully green. Conversions and optimizations also need real throughput (FPS / MB/s), target format and the actual hardware path demonstrated on `fixtures/`.
6. **Fix root causes.** No silent fallbacks (especially no silent CPU fallback on failed GPU setup), no `let _ = ...`, no workarounds masking driver/parser bugs.
7. **Lints.** Workspace lints (`unsafe_code = "forbid"`, `unwrap_used = "deny"`, `expect_used = "deny"`, `panic = "deny"`, `pedantic = "warn"`) apply everywhere. No local `#[allow(...)]` without a documented, compelling reason.
8. **License policy (default build).** `fastconv` is **AGPL-3.0-only** (`LICENSE`). The default dependency tree allows permissive licenses (MIT, Apache-2.0, BSD-2/3-Clause, Zlib, ISC, Unlicense, CC0, MPL-2.0) and AGPL-compatible copyleft (AGPL-3.0, GPL-3.0, LGPL-3.0, including dual licenses like `AGPL-3.0-only OR Commercial`). Forbidden: missing/unknown licenses, purely proprietary or non-commercial-without-AGPL licenses, and incompatible copyleft (e.g. `GPL-2.0-only`). `cargo xtask verify` checks `cargo metadata` and fails on violations.
9. **Fair benchmarks (no preset cheating).** Do not win by dropping compression or quality while `ffmpeg` runs its slow default. Compare the same compression class for lossless formats (both Deflate, or both uncompressed/PackBits). For lossy formats (AVIF, JXL, WebP, JPEG), keep file size and PSNR in a comparable range — no 5–10x larger output just to make the bar faster. Cap worker concurrency for memory-hungry 4K codecs (AVIF/JXL): no 3+ GB peak RAM for marginal speed gains.

## Git workflow

No direct changes or commits on `main`. Before and after every task:
1. **Check the branch:** `git branch --show-current` before any code edit.
2. **Create a branch:** if on `main`, first `git checkout -b feat/<short-name>` or `fix/<short-name>`.
3. **Verify & PR:** merge every feature and fix via PR only after green `cargo xtask verify`. Use the project skills in `.agents/skills/` (`rust-architecture`, `prove-it-works`) and global skills in `~/.agents/skills/` (`orchestrate`, `benchmark`, `babysit`, `cleanuptree`, `queue-features`, `rebenchmark-main`).
4. **Clean up after merge:** `git checkout main`, `git pull`, `git branch -d <branch>`.

Note: a local `.git/hooks/pre-commit` hook also blocks accidental commits on `main`.

## Model & delegation discipline

- **Decisions vs loops:** use expensive/limited models for decisions, architecture, deep performance audits and final diff reviews (1–2 turns). Cheap/unlimited models do the loops: coding, compiling, Clippy/pedantic sweeps, tests, fixtures, benchmarks, rebases, merge conflicts, PR babysitting.
- **Architect → worker → reviewer:** the expensive model designs the plan/APIs and delegates implementation to a cheap model in an isolated worktree (`isolation: worktree`) with exact paths, crate APIs and edge cases. The cheap model runs `cargo xtask verify` (never parallel benchmarks across worktrees). The expensive model reviews the resulting diff before merge.
- **Benchmark handoff → separate worker:** performance, codec or conversion changes also get a dedicated top-level `opencode/hy4-flash-free` thread running `$benchmark` (fair `cargo xtask bench` / `--audio` head-to-head → validated numbers in `STATE.md`) before the `$babysit` PR thread. Run the two threads serially, never concurrently.
- **PR hygiene (mandatory):** never merge while GitHub CI or CodeRabbit is still running; resolve every CodeRabbit thread before merge (`0 unresolved`); only after green CI and finished CodeRabbit, squash-merge, update `main`, delete the local branch/worktree.
- **Pre-PR hardening (especially cheap models):** cap allocations before `Vec::resize`; exclusive atomic staging (`create_new(true)` everywhere, `.mode(0o600)`/`OpenOptionsExt` behind `#[cfg(unix)]`); reject path aliases in `probe`; never swallow decode errors; buffer file I/O (`BufWriter`/reusable buffers).
- **Babysitting is never a subagent:** `$babysit` (and any PR watch) runs in a top-level thread of the executing model — delegated children cannot hold PR watches.
- **Read-only spelunking:** when the expensive model needs crate-API facts from `~/.cargo/registry/src` or a large refactor, delegate a read-only researcher that returns a compact report with line numbers — keeps the expensive model's context small.
