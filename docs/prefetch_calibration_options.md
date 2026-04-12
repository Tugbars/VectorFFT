# Prefetch Calibration — Design Options

## Introduction

### What we're solving

Software `_mm_prefetch` instructions in spill-heavy codelets (R=16, R=32) can hide
the latency of stride-K twiddle loads that hardware prefetchers fail to predict.
The amount of prefetch that helps is **CPU-dependent**: aggressive prefetch on
compute-bound hardware adds front-end pressure and hurts performance, while
memory-bound hardware needs it to fill the idle memory slots.

We need a mechanism to select the right prefetch parameters **per codelet per
CPU**, without paying any overhead in the execution hot path.

### Why this matters

Evidence from VTune profiling on Intel Raptor Lake (i9-14900KF):

- **R=16 AVX2 t1_dit K=256**: prefetching next column's twiddles during current
  column's spill stores gave a **2.7x speedup** (4474 ns → 1631 ns). Retiring
  rose from 21.8% toward ~25%, L1 Bound dropped from 30.3% to 11.6%, DTLB
  Load dropped from 1.5% to 0.4%.
- **R=8 AVX2**: the same style of prefetch caused an **8–15% regression**
  because the codelet is already 72% retiring with no idle slots to absorb
  extra front-end work.

The same behavior is expected — with different optimal parameters — on Zen,
Apple Silicon, future Intel big/little configurations, and composite radixes
(R=20, R=25) that share the spill-heavy structure.

### Parameters that vary per CPU

- **depth** — how many columns ahead to prefetch (1 = next column, 2 = skip one)
- **count** — how many twiddle cache lines per insertion slot
- **hint** — cache level target: `_MM_HINT_T0` / `T1` / `NTA`
- **enabled** — some codelets (R=8) want zero prefetch

### Search space is bounded

Prefetch has diminishing returns. After enough lines are prefetched, more
prefetching adds front-end pressure without hiding additional latency. A
monotone walk (increment count until regression, try next depth) finds the
winner in ~5–10 measurements per codelet. No combinatorial explosion.

### Scope

- **Calibrate**: R=16 fwd+bwd, R=32 fwd+bwd (proven to benefit)
- **Skip**: R=2, R=4, R=8, small primes (no benefit or active harm)
- **Later**: R=20, R=25, R=27, other composites (same spill pattern, same
  mechanism applies when generators are touched)

### Hard constraint

**Zero overhead in execution.** Planning overhead is free — not on the FFT
latency path. The design must not add branches, function-pointer indirection,
or extra arguments to the codelet call in the execution path.

---

## Option 1: Compile-time variants only

Ship N precompiled variants per codelet, each baked with a fixed prefetch
config. Pick the winner via direct microbenchmark, use pruning step to ship
only the winner.

### Mechanics

Generator invoked N times at build time with different flags:
```
radix16_t1_dit_fwd_avx2_pf_d1c0t0   # baseline, no prefetch
radix16_t1_dit_fwd_avx2_pf_d1c2t0
radix16_t1_dit_fwd_avx2_pf_d1c3t0
...
```

Two-phase build:
1. **Calibration phase**: generate all variants, compile into a calibration
   bench, run bench, pick winner, write `winners.cmake`.
2. **Production phase**: generator invoked once per codelet with winner config,
   emits the single production codelet. Losers discarded.

### Costs

- Generator runs N times during calibration (fast, seconds)
- Calibration build has ~N×(radixes calibrated) extra symbols
- Two-phase cmake flow (`vfft_pf_calibrate` target → main build reads winners)
- Production binary: 1 symbol per codelet (same as today)

### Benefits

- Cleanest measurement (no branches in calibration codelets — just call each
  variant's function pointer directly)
- Minimal binary (losers pruned)
- Generator stays single-mode (parameterized, no calibration emit path)

### Drawbacks

- Search space bounded by precompiled grid (not a real issue — grid covers the
  monotone-walk range)
- Two-phase build is operational complexity for users building from source
- Requires a `default_winners.cmake` or equivalent fallback for users who skip
  calibration (adds a shipped file)

---

## Option 2: Runtime-param calibrator + compile-time production variant

Dual-mode generator: emits a parameterized "calibration" codelet that reads
prefetch config from a struct (branchy, used only by the planner), and a
separate "production" codelet with the winner baked in (zero branches, used by
the executor).

### Mechanics

Build phase 1 — calibration:
- Generator emits `radix16_t1_dit_fwd_avx2_calibrate(re, im, W_re, W_im, ios, me, const prefetch_cfg_t *pf)`.
- This ONE symbol replaces the N-variant grid of Option 1.
- Calibration bench calls it with varying `pf` values, picks winner.

Build phase 2 — production:
- Generator re-invoked with winner config, emits
  `radix16_t1_dit_fwd_avx2(re, im, W_re, W_im, ios, me)` — same signature as
  today.

### Execution

Unchanged from today. Executor calls the production symbol via function pointer.
Zero overhead. The calibration codelet exists in a separate build tree and is
never linked into `libvfft`.

### Costs

- Generator has two emit modes (calibration with config reads, production with
  unrolled baked prefetches). The non-prefetch portion of the codelet is
  byte-for-byte identical between modes — shared emit logic.
- Calibration measurement has branch noise (~0.5% of codelet time per slot —
  negligible but real).
- Still a two-phase build.

### Benefits

- One calibration symbol per codelet (vs N for Option 1) — simpler calibration
  build tree.
- Search space is not bounded by a precompiled grid — calibrator walks freely
  over whatever range we enumerate in the bench.
- Easier to extend: adding a new axis (e.g. `hint`) requires no new variants,
  just new config struct fields.

### Drawbacks

- Two code paths in the generator must stay semantically equivalent outside the
  prefetch slots. Drift risk, though low because the non-prefetch emit is
  factored and shared.
- Measurement bias from branches (small; mitigated by a head-to-head rebench
  of the top candidate against a baked variant if you want to be paranoid).
- Still two-phase build.

---

## Option 3: Runtime calibration at plan time, precompiled variant grid

All N variants precompiled at build time (single phase). Calibration happens
at **runtime** during the first `stride_measure_plan()` call on a given CPU.
Winner cached in wisdom. Execution dispatches directly to the winning
precompiled variant.

### Mechanics

Build phase (single):
- Generator emits all N variants (same as Option 1 calibration phase):
  ```
  radix16_..._pf_d1c0t0
  radix16_..._pf_d1c2t0
  radix16_..._pf_d1c3t0
  ...
  ```
- All N registered in the codelet registry with their (depth, count, hint)
  metadata visible.
- All N ship in `libvfft`.

Plan time:
- On first `stride_measure_plan()` call per CPU, planner walks the variant grid
  for each calibrated codelet (monotone, early stop).
- Winner recorded in wisdom as a codelet-level preference, orthogonal to the
  per-(N, K) factorization wisdom.
- Subsequent plan calls read from wisdom, skip re-calibration.

Execution:
- Plan stores the winner's function pointer. Executor dispatches via that
  pointer. Zero branches, zero overhead, identical dispatch to today.

### Costs

- Binary grows by N variants per calibrated codelet. Estimate for R=16 + R=32
  at N=8 each: ~40 KB total.
- First `measure_plan` call on a new CPU runs calibration (a few seconds,
  one-time).
- Variants can't be pruned — all must ship because different CPUs pick
  different winners.

### Benefits

- **Single-phase build.** No calibration cmake target, no two-phase flow, no
  `winners.cmake`, no generated-file cleanup step.
- **Per-CPU calibration is automatic.** User doesn't run a separate step —
  calibration happens the first time they measure-plan on that CPU.
- **Wisdom caches the result.** Fits the existing wisdom model exactly like
  blocked-executor calibration does today.
- Generator stays single-mode — no calibration emit path, no dual-mode
  complexity.
- Measurement is exact — each variant is called directly, no branches,
  no bias.

### Drawbacks

- ~40 KB binary bloat (all variants ship).
- First measure_plan call is slower (calibration runs). Subsequent calls use
  wisdom.
- `stride_auto_plan` (estimate mode) needs a sensible default — either "use
  the first variant" or "use the generator's documented Raptor Lake default."

---

## Comparison table

| | Option 1 | Option 2 | Option 3 |
|---|---|---|---|
| Execution overhead | 0 | 0 | 0 |
| Execution symbols per codelet | 1 (winner) | 1 (winner) | N (grid) |
| Calibration symbols per codelet | N | 1 | N (same as exec) |
| Generator modes | 1 | 2 | 1 |
| Build phases | 2 | 2 | 1 |
| Search space bound | precompiled grid | continuous | precompiled grid |
| Calibration trigger | build time | build time | first plan call |
| Binary size | smallest (1 variant) | smallest (1 variant) | ~40 KB larger |
| Per-CPU workflow | rerun `vfft_pf_calibrate` | rerun `vfft_pf_calibrate` | automatic |
| Measurement fidelity | exact | ~0.5% branch noise | exact |
| Wisdom integration | separate file | separate file | native |

---

## Decision axes

1. **Build complexity vs binary size**: Options 1 and 2 trade binary size for
   a two-phase build. Option 3 trades binary size for single-phase simplicity.
2. **User workflow**: Option 3 requires zero extra steps per CPU. Options 1
   and 2 require a calibration cmake target invocation.
3. **Wisdom model fit**: Option 3 integrates natively with the existing wisdom
   system used for blocked-executor calibration. Options 1 and 2 live outside
   wisdom.
4. **Measurement fidelity**: Options 1 and 3 measure exactly what executes.
   Option 2 has small branch noise.
5. **Search space flexibility**: Option 2 is the only one with a continuous
   search space. The bounded-walk argument makes this mostly irrelevant.

---

## Recommendation (for reference only — decision is yours)

Option 3 fits the existing wisdom architecture cleanly and removes an entire
build phase. The 40 KB binary cost is small next to the operational
simplification. Option 2's runtime calibrator is elegant but the continuous
search space isn't needed given the monotone-walk bound.

If binary size is a hard constraint — Option 1 or Option 2.
If operational simplicity is the priority — Option 3.
