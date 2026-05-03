# VectorFFT wisdom system

How VectorFFT decides which **factorization, codelet variants, and orientation**
to use for a given (N, K) — both at calibration time (when wisdom is built)
and at plan time (when wisdom is consumed).

The wisdom system is the second of VectorFFT's two novel pieces. The first
is the cost model (`docs/cost_model/`); this folder covers the wisdom that
sits *above* it as the high-quality calibrated path.

## The novel claim

> **Plan-level joint search over `(factorization × permutation × per-stage
> variants × orientation)` produces wisdom entries that record exactly
> what the calibrator measured to be fastest, with a noise-resistant
> top-K + deploy-rebench pipeline that gets within 3–12% of PATIENT-
> style joint Cartesian at ~7% of the wall time.**

Three components stack to make this work:

1. **Plan-level joint search.** At each (N, K) cell, search the joint
   product, not per-radix in isolation. Per-codelet isolation was
   tried and rejected — see ADR-001 in [09_decisions.md](09_decisions.md).
2. **Top-K + multi-pass refinement.** The search is layered to resist
   measurement noise: coarse pass with LOG3 priming, top-K refine
   cartesian, global pool of refine survivors, deploy-rebench with
   decorrelated noise.
3. **v5 wisdom file shape.** Explicit per-stage variant codes mean
   lookup builds the plan exactly as measured, no inference.

## Reading order

0. [00_thesis.md](00_thesis.md) — **start here** — what the system does and why it's structured this way
1. [01_architecture.md](01_architecture.md) — components and data flow
2. [02_codelet_taxonomy.md](02_codelet_taxonomy.md) — variant / dispatcher / protocol levels
3. [04_layer2_plan_level.md](04_layer2_plan_level.md) — `vfft_wisdom_tuned.txt` format + v3/v4/v5 evolution
4. [05_calibrator_pipeline.md](05_calibrator_pipeline.md) — coarse → refine → deploy with top-K cutoffs
5. [06_lookup_pipeline.md](06_lookup_pipeline.md) — `stride_wise_plan` flow + Bluestein/Rader recursion (honest comparison vs FFTW)
6. [07_dif_filter.md](07_dif_filter.md) — why DIF orientation is whole-plan-or-nothing
7. [08_blocked_executor.md](08_blocked_executor.md) — third orthogonal dimension; doesn't compose with variants in v1.1
8. [09_decisions.md](09_decisions.md) — ADR record

## Audiences

- **First-time reader / reviewer**: read 00 first.
- **Library maintainer**: 02 + 04 + 05 cover the data formats and pipeline.
- **Calibration runner**: 05 + 07 + 08 cover the calibrator's flow and the
  decisions about what gets benched.
- **User of the wisdom file**: 04 + 06 cover the file format and how the
  planner consumes it.
- **Reviewer auditing the methodology**: 00 + 04 + 05 + 09 — the technical
  claims and the decision rationale.

## Source of truth

Code, not docs:

- [src/core/planner.h](../../src/core/planner.h) — `stride_wisdom_t`, `stride_wise_plan`, `_stride_build_plan_explicit`
- [src/core/dp_planner.h](../../src/core/dp_planner.h) — `stride_dp_plan_measure` (top-K + variant cartesian)
- [src/core/registry.h](../../src/core/registry.h) — `vfft_variant_t`, `vfft_stage_variants`, `vfft_variant_iter_*`
- [build_tuned/calibrate_tuned.c](../../build_tuned/calibrate_tuned.c) — calibrator (top-level driver)
- [build_tuned/vfft_wisdom_tuned.txt](../../build_tuned/vfft_wisdom_tuned.txt) — shipped wisdom file

If a doc disagrees with the code, the code wins. Open an issue.

## Sibling docs

- [docs/cost_model/](../cost_model/) — the closed-form cost model used by
  `VFFT_ESTIMATE`. Wisdom is the high-quality alternative when measured
  numbers are available.
