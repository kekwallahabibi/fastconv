---
name: prove-it-works
description: Use before completing any feature, bugfix or performance tuning in fastconv to objectively prove correctness, throughput and hardware execution.
---

# Prove it works

No task is complete until every relevant proof is backed by real terminal output.

## 1. Static & test verification (mandatory)
From the repo root: `cargo xtask verify`. All three stages must be green:
1. `cargo fmt --all -- --check`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo test --workspace`

## 2. Real conversion against `fixtures/`
When conversion, probe or pipeline logic changed:
1. Run the real `fastconv` binary on at least one real `fixtures/` input; write output to a temporary, ignored path.
2. Verify the output objectively: exists, non-empty, container, codec, resolution/sample rate, duration, streams (e.g. `ffprobe` or parser output).
3. Exercise error paths explicitly: missing source, identical source/target path, incompatible backend.

## 3. Throughput hillclimb vs baseline
For perf tuning or new backends:
1. Measure a reproducible baseline on the same `fixtures/` before the change (`cargo xtask bench` or a defined timing/throughput command).
2. Measure the exact same scenario after and compare runtime (ms/s), throughput (MB/s) and processing rate (FPS).
3. No measurable gain, or clarity lost without a real speedup → discard the optimization.

## 4. Prove no silent CPU fallback
For every GPU conversion:
1. Record the chosen `ExecutionPlan` (`GpuZeroCopy` with a concrete `GpuBackend`).
2. Check runtime evidence (driver/encoder telemetry, `nvidia-smi`/VAAPI/VideoToolbox metrics, or explicit encoder logs).
3. If the requested GPU is unavailable locally, the run must fail cleanly — never silently switch to CPU.

## 5. Final evidence
Document in the reply (and later the PR): exact commands run; `cargo xtask verify` result; measured before/after values (FPS, MB/s, runtime) and target-format check; explicitly note any hardware checks that could not run locally for lack of a GPU.
