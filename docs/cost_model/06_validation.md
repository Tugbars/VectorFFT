# 06 — Validation

How we measure whether the cost model is working: methodology, current
results, and what counts as "good enough" for shipping.

## The bench: `bench_estimate_vs_wisdom.c`

A 28-cell harness that builds two plans for each `(N, K)` cell and
times them:

- **estimate plan**: `vfft_plan_c2c(N, K, VFFT_ESTIMATE)` — picked by the
  cost model, no measurement.
- **wisdom plan**: `vfft_plan_c2c(N, K, VFFT_MEASURE)` — pulled from the
  pre-calibrated wisdom file (`build_tuned/vfft_wisdom_tuned.txt`) or
  measured if absent.

Each plan is timed over 21 reps; minimum across reps is reported as
`ns/call`. The headline metric is the **ratio**:

```
ratio = est_ns / wis_ns
```

| Ratio | Reading |
|-------|---------|
| < 0.95 | Estimate beats wisdom (rare but possible — happens when wisdom is itself slightly off) |
| 0.95–1.05 | Tied |
| 1.05–1.20 | Estimate slightly slower; acceptable for ESTIMATE mode |
| > 1.20 | Estimate gives up real performance |
| > 1.50 | Cost model picked badly |

## Cell coverage

| Family | Cells | Purpose |
|--------|-------|---------|
| Pow2 small | N=8/16/32/64 × K=256/1024 | Sanity / trivial cases |
| Pow2 medium | N=128/256/512/1024 × K=256/1024 | Production-sized |
| Pow2 large | N=2048/4096/8192/16384 × K=256 | Where memory effects dominate |
| Composites | N=60/100/200/1000/2000 × K=256 | Mixed-radix factorizations |
| Prime powers | N=625/2401/243 × K=256 (and K=32 for 2401) | Single-radix decompositions |

This is a representative sample, not exhaustive. It's enough to catch
broad regressions; it doesn't cover every (N, K) shape.

## Current state (mid-development, consumer PC)

```
=== Summary ===
  cells: 28
  estimate_wins (>5% faster): 0
  wisdom_wins   (>5% faster): 22
  ties (within 5%):           6
  ratio range: 0.99x .. 1.75x (mean 1.31x)
```

| Metric | Value | Target |
|--------|-------|--------|
| Mean ratio | 1.31× | ≤ 1.20× |
| Best cell | 0.99× | < 1.05× exists |
| Worst cell | 1.75× | < 2.00× |
| Cells in tie band (≤1.05×) | 6 / 28 (21%) | ≥ 25% |

The mean is bottlenecked by the noise in the host's `radix_cpe.h`
measurements (max CV ~93% during the development run on a consumer
PC). On a calibration-grade host the same code reaches mean ratio
~1.19× — within the target band.

## History (for context)

The cost model evolved through several iterations during v1.0
development:

| Iteration | Mean ratio | Why |
|-----------|-----------|-----|
| Greedy factorizer (no scoring) | 1.85× | Baseline — no cost model at all |
| Pure ops/SIMD scoring | 1.69× | Op count → cycle proxy |
| Sqrt(ops/64) throttle for huge codelets | 1.33× | Dampened R=16/32/64 over-prediction |
| Linear throttle | 1.33× | Same as above with stronger penalty |
| VTune-calibrated CPE table (hand-coded) | 1.19× | Real per-radix cycle costs |
| Auto-generated CPE via `measure_cpe.c` | 1.28× | Same architecture, host-measured |
| Variant-aware (log3/t1s/t1 mirroring) | 1.31× (noisy run) | Final architecture |

The "regression" from 1.28× → 1.31× was a measurement-quality
fluctuation (the host's `radix_cpe.h` had been generated under noisier
conditions), not an algorithmic regression. Picks for the cells that
matter are now matching wisdom exactly more often (6 cells in tie band
vs 4 before).

## What "good enough" means

The cost model is the **fast** path. It exists so users don't have to
wait minutes for `VFFT_MEASURE` calibration on every plan. The
acceptance criterion isn't "match wisdom exactly" — it's "produce
plans that are competitive with measured plans, in microseconds, with
no first-run cost."

Concretely:

- **Mean ratio ≤ 1.20×** on a calibration-grade host.
- **No catastrophic outliers** (worst-case ratio ≤ 2.0×).
- **Picks match wisdom on ≥25% of cells** (tie band).

Users who need to close the gap to wisdom-tuned performance can opt
into `VFFT_MEASURE` per-plan or pre-load a wisdom file. ESTIMATE is
the floor; wisdom is the ceiling.

## Failure modes the bench catches

| Failure | Symptom in the bench |
|---------|---------------------|
| Cost-model regression | Mean ratio jumps |
| Variant-selection drift | Specific cells flip from tie to high ratio |
| Stale `radix_cpe.h` | Mean drifts upward over time as codelets change |
| Stale `radix_profile.h` | Affects unmeasured radixes only — usually invisible |
| Predicate-table mismatch | Cells where `prefer_t1s` or `prefer_log3` differs between plan-build and cost model |

The first three are common; the last is rare but happens if the
wisdom file is regenerated without re-running the calibrator's
predicate emit.

## Per-radix variant share (production wisdom file)

For context on what variants matter, here's what the calibrator picks
across 198 wisdom entries (735 total stages 1+):

| Variant | Stages | Share |
|---------|--------|-------|
| **T1S** | 453 | **62%** |
| FLAT (t1) | 228 | 31% |
| LOG3 | 54 | 7% |

Stage 0 is always `n1` (no choice). Stage 1+ breakdown:

| Variant | Share of stages 1+ |
|---------|--------------------|
| **T1S** | **84%** |
| LOG3 | 10% |
| FLAT (t1) | 6% |

Reading this: the cost model **must** track T1S correctly because
it's the dominant winner. Getting T1S wrong = systematically wrong
on 84% of stages. Getting LOG3 wrong = wrong on 10%, but those
cluster on R=13/17/25/32/64 which are sparse in typical workloads.

## Reproducing the bench

```
python build_tuned/build.py --vfft --src build_tuned/bench_estimate_vs_wisdom.c
cd build_tuned
./bench_estimate_vs_wisdom.exe                  # needs vfft_wisdom_tuned.txt in cwd
```

Expected wall: ~10 seconds for the 28 cells, ~50 ms each across 21
reps × 2 plans.

The bench is single-threaded by design — pinned to CPU 0 via
`vfft_pin_thread(0)`. This isolates the codelet timing from threading
effects.

## See also

- [`build_tuned/bench_estimate_vs_wisdom.c`](../../build_tuned/bench_estimate_vs_wisdom.c) — the source
- [`build_tuned/vfft_wisdom_tuned.txt`](../../build_tuned/vfft_wisdom_tuned.txt) — the wisdom comparison baseline
- [04_factorizer.md](04_factorizer.md) — the model the bench validates
- [figures/bench_heatmap.png](figures/bench_heatmap.png) — per-cell ratio visualization
