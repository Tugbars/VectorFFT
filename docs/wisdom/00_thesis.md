# 00 — Thesis

What problem the wisdom system solves and what makes the design novel.

## The claim

> **For each `(N, K)` cell, the calibrator searches the joint product of
> `(factorization × permutation × per-stage variants × orientation)` with
> a noise-resistant top-K + deploy-rebench pipeline, producing wisdom
> entries that ship with explicit per-stage variant codes. Per-codelet
> isolation wisdom was tried and rejected because OoO interactions
> between stages don't compose from per-codelet timings; the plan-level
> bench is the only verdict that holds up.**

The system has many codelet variants — `flat`, `t1s`, `log3`, `buf`,
each with DIT and DIF orientations, and within `flat` multiple buffered
sub-variants. That's the "lots of codelet types" the wisdom system
has to choose between, per stage, per cell. What makes it work in
practice is that the search isn't done per-codelet; it's done at
plan level, with a search structure designed to resist the noise
that would otherwise make any single bench's verdict unstable.

## What the wisdom system does

For each `(N, K)`:

1. Enumerate candidate factorizations + permutations
2. Coarse-bench with default variants, with LOG3-priming so
   LOG3-friendly multisets aren't pruned at coarse
3. Refine: top-K coarse survivors → variant Cartesian × {DIT, DIF}
4. Pool refine winners across multisets, threshold-filter to within
   10% of best
5. Deploy-rebench survivors with an independent harness; deploy-fastest
   wins
6. Verify roundtrip error, write a v5 wisdom entry: **factorization +
   per-stage variant codes + orientation + blocked-executor metadata**

The result is shipped as `vfft_wisdom_tuned.txt`. At plan creation
time, `stride_wise_plan` looks up `(N, K)`; on a hit, the executor
gets the exact plan the calibrator measured to be fastest, with the
exact per-stage codelet variants.

## Why this is novel

Three components stack:

1. **Plan-level joint search.** Per-codelet isolation wisdom was the
   original idea: bench each codelet alone, pick winners by `(R, me,
   ios)`, hand the planner predicates. Validation pilot at N=4096
   showed this was wrong by **+40% (K=4) and +16% (K=256)** vs whole-
   plan joint search. OoO pipeline state from previous stages —
   cache, DTLB, port pressure — biases what wins next, and isolated
   benches can't see that. Plan-level joint search measures the
   composition directly. *That experiment, and its rejection, is the
   only reason the per-codelet idea is mentioned at all in this doc
   set; it has no other role here.*

2. **Top-K + multi-pass refinement.** Variant choice can flip the
   factorization ranking — a multiset that loses with default
   variants can win with LOG3 — so the calibrator can't pick a single
   coarse winner and refine only that. Top-K-at-every-level DP
   (Upgrade D) keeps sub-problem runners-up alive across recursion
   frames; LOG3-aware coarse probe (Upgrade F) ensures LOG3-friendly
   multisets survive coarse; deploy rebench with an independent
   harness (Upgrade H) breaks variant-axis ties via decorrelated
   noise. Each layer addresses a specific way single-pass search
   would fail under noise.

3. **Wisdom file shape.** The v5 file format records explicit per-
   stage variant codes, not just the factorization. Lookup builds the
   plan exactly as measured, no inference. v3/v4 (factorization-only)
   were stages on the way; v5 is what makes plan-level joint search
   reproducible at deploy time.

## What's not novel (acknowledged honestly)

- **Wisdom files in general** — FFTW invented this pattern. We use a
  fixed-cell table; FFTW uses an accumulated planner-state hashtable.
  Different shapes, same concept.
- **Bluestein/Rader recursion with wisdom** — FFTW's `mkplan_f_d`
  passes the planner through to inner FFTs (`dft/bluestein.c:206`,
  `dft/rader.c:241` in FFTW 3.3.10). Same design pattern as our
  `stride_auto_plan_wis`. The novelty is in *what* gets carried (a
  fixed-cell table with explicit variant codes vs FFTW's hashtable),
  not in *that* something is.
- **Top-K candidate lists in DP planning** — standard combinatorial
  search technique. We apply it at every recursion level which is
  unusual but not unprecedented.

What IS specifically novel is the **finding** (per-codelet wisdom
empirically doesn't compose) plus the **architectural response**
(plan-level joint search with a search pipeline structured to resist
the specific noise patterns that would otherwise make wisdom
unreproducible).

## Cost-quality knob

`MEASURE_TOPK_DEFAULT` controls the search budget:

| K_top | Wall (N=4096 K=256) | Quality |
|-------|---------------------|---------|
| 1 (legacy) | low | misses variant-axis-flips-multiset cases |
| **5 (default)** | **~250 s/cell** | **within 3-12% of EXTREME** |
| ∞ (= EXTREME / PATIENT) | ~3000 s/cell | optimal joint cartesian |

Shipped wisdom uses `K_top = 5`. EXTREME is opt-in.

## See also

- [01_architecture.md](01_architecture.md) — components and data flow
- [04_layer2_plan_level.md](04_layer2_plan_level.md) — wisdom file format
- [05_calibrator_pipeline.md](05_calibrator_pipeline.md) — the search pipeline
- [06_lookup_pipeline.md](06_lookup_pipeline.md) — how plans are built from wisdom
- [09_decisions.md](09_decisions.md) — ADR-001 captures the per-codelet rejection in formal context
