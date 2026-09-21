# In-place engine forms below 2048 — roadmap item

**Date:** 2026-09-21 · **Status:** OPEN (owner: "we can pursue this later") ·
**Evidence:** `build_tuned/results/gauntlet_2026-09-20/gauntlet_ip.csv` (511 cells, 2..512,
both flips, real in-place races) beside `gauntlet.csv` (the same cells out of place).

## 1. The finding

The in-place cell is its own contract since 2026-09-21: the K=1 planner races every
arm executed z -> z and banks the winner on the cell's `place=ip` row
(`docs/design/planning_model.md`, `measurement_arms.md`). Measured at every N from 2
to 512 against MKL with DFTI_INPLACE:

```
                         in place      out of place      (same cells, worse of two flips)
 ratio vs MKL, median      1.33            1.33
 cells below 1.0x            76              74
 our time, ip / oop        0.998   (p10 0.96, p90 1.03)
 MKL's time, ip / oop      0.996
```

The in-place race re-picks among near-ties (104 of 242 non-prime cells took a
different plan: 34 route changes, 62 re-factorizations, 8 forms) and those cells run
at 0.995x of their out-of-place time, the same as the plan-identical cells (0.998x).
In place is exactly as fast as out of place, for us and for MKL.

## 2. Why

Below 2048 every K=1 interleaved engine consumes its input through a staging plane
(the pair's `mid`, chain3's `mid1`/`mid2`, the flat DIT's `stg`, ZTURN-T's plane
drivers) or an alias-tolerant solo kernel, and writes the output afterwards. In
place therefore saves the CALLER a buffer; it saves the LIBRARY no pass, no load and
no store. Nothing in the pool has an in-place form that does less work than its
out-of-place form, so the race cannot find one.

Where an in-place-native engine exists it does win: the in-place Sande-Tukey ZTURN-T
at 2048..262144 is 3-30% faster than its out-of-place forward
(`docs/design/ztt_scrambled_design.md`; memory: scrambled ZTURN-T shipped 09-14).

## 3. What "in place faster" would take

An engine form whose in-place execution does strictly less than its out-of-place
one, raced as an arm in the in-place cell:

- **Pair / chain3 without the plane.** The first stage writes its output over the
  input in place (the column pass is a same-slot pass already: the `t2c` kind), and
  only the last stage needs an out-of-place store. Saves one full plane write+read
  per plan (one of three passes for chain3).
- **Flat DIT in place.** The stages already run in `stg`; an in-place form runs them
  in the caller's buffer and drops `stg` (the natural last stage is the obstacle:
  its group redirection needs a destination that is not the source).
- **The in-place ZTURN-T form extended below 2048** (the natural class): today its
  plane drivers cost MORE in place at 16 (the pair wins there).

The measurement bar is the one every arm meets: forward vs the long-double reference
in place (`benches/k1_fwd_ref_probe.exe --ip`), then the cell's own in-place race
against the existing arms; a form that does not win a cell is not kept.

## 4. Not to be confused with

- The in-place natural-order *design* of 2026-07-04 (`natural_order_inplace_design.md`):
  the split library's shadow-plane machinery, superseded for interleaved callers.
- The reference-row mechanism the in-place door used until 2026-09-21 (served the
  out-of-place verdict in place): deleted; the shipped store's legacy
  `place=ip lay=il | eng=k1 mode=ilp` rows are dead and are dropped at the next merge.
