# r2c inner-cell wisdom hygiene — platform runbook

*Action #1 from the 2026-07-15 r2c attribution bench
(`docs/roadmap/r2c_c2r_il_design.md`, geometry contracts §6a20). This is the
data-only task: no code changes required — the plumbing is proven (the bench's
run-2 improvement arrived through the wisdom bundle, so r2c create does
consult the c2c table for its inner plan).*

## Why (evidence + honest effect size)

At (512,256), the same r2c cell ran 354 µs with a container-fresh MEASURE
inner plan and 244 µs with a proper (256,256) wisdom entry. MKL moved 18%
between those runs (container phase), so the net-of-drift signal is the
ratio: **1.396× → 1.177× vs MKL ≈ 16% attributable to the inner plan
alone.** Cheapest 16% available; cells whose halves already carry good
entries see nothing (they were never broken).

## What r2c consults, per dispatch path

| r2c cell class | path | wisdom that matters |
|---|---|---|
| even N, high K | decoupled: c2c(N/2, K) inner + recombine | **the (N/2, K) row in the c2c table** — this runbook |
| even/odd N, low K | rfft-native | `rfft_calibrate` artifacts (separate mechanism; only audit if low-K r2c matters to you — note we already beat MKL 1.22–1.62× there in-container) |
| odd N | `_r2c_plan_odd` | different inner structure — out of scope here |

The K threshold between paths is the dispatcher's race decision (see the
persistence check below).

## Procedure

**1. Enumerate your r2c production cells** into a file, one `N K` per line
(only even-N, high-K cells matter):

```
# r2c_cells.txt
512 256
1024 64
2048 256
...
```

**2. Audit for missing half-cells:**

```bash
WIS=path/to/spike_wisdom.txt
while read N K; do
  [ $((N % 2)) -eq 0 ] || continue
  H=$((N / 2))
  grep -q "^$H $K " "$WIS" || echo "MISSING: $H $K   (inner of r2c $N,$K)"
done < r2c_cells.txt
```

The N grid in the current file is dense (16…823543; 256/500/512/1024 all
present) — expect the gaps to be **K columns** at existing halves (e.g.
N=256 present at K=4 from c2c sweeps but absent at K=256), not missing N.

**3. Fill the gaps** with your canonical calibration sweep at your canonical
rigor — exactly as you'd calibrate any c2c cell. These are ordinary c2c
entries; nothing r2c-specific about the rows themselves.

**4. Validate** by re-running the r2c bench (`benches/bench_r2c_tax.c`, the
`-DVFFT_R2C_PROFILE` binary is enough) with a bundle containing the updated
file, before/after per cell. Judge by the **vs-MKL ratio**, not absolute µs.
To confirm the plan actually changed, compare the `(N/2, K)` wisdom line's
factor list against what the pre-fill create was choosing (or just trust the
ratio move).

## Two checks while you're in there

**A. Dispatch-race persistence.** The rfft-vs-decoupled decision at r2c
create is measured (a race). No banking site for the result was found during
the bench recon — if it isn't persisted on your platform either, every
create re-pays the race. Confirm by timing two identical creates back-to-back
with a warm bundle; if the second isn't near-instant, that's a persistence
gap worth its own (small) fix. Don't fix blind — locate where the dispatcher
would naturally bank it first.

**B. Rigor consistency.** Any half-cells that got banked *implicitly* by past
r2c creates (calibrate-on-miss at whatever rigor that create used) may be
low-rigor rows shadowing your canonical quality. If your table has rows for
half-cells you never deliberately swept, re-run those at canonical rigor and
replace.

## Expectations & scope

- Effect concentrates on cells whose inner previously fell to fresh MEASURE /
  auto-plan; well-covered cells are unchanged by design.
- ~16% net-of-drift at the measured cell class is the anchor; your box's
  number will differ (container caveats apply to the anchor itself).
- This runbook deliberately excludes: odd-N cells, low-K/rfft cells, and the
  optional code follow-up (auto-bank-on-miss at r2c create, the 6a19 fft3d
  composition pattern) — that last one only matters if maintaining the grid
  by sweep becomes annoying.
