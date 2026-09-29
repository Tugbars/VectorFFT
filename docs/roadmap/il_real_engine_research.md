# The IL real engine: the research record

The standing record of the interleaved real-transform work: what the engine is
to be, which methods are on the table, and each method's current measured
verdict. A verdict is overwritten when re-measured; the runs it comes from are
in `gauntlet/results/`. The design input is `fftw_real_study.md` beside this
file. This is a declaration of the current state, not a journal.

## 1. The primitive

One batched real-rows-to-CCE-rows transform and its c2r mirror:

- input: K contiguous real rows of N reals at pitch P (out of place P = N; in
  place the padded 2·(N/2+1)); output: K CCE rows of N/2+1 pairs at pitch Q;
  one call for the whole batch; natural order (a real spectrum has no other);
- 1D K>1 is rows = K (transform-contiguous, the default geometry for
  interleaved real once the primitive lands; lane-major stays with split);
  2D is rows = N1 followed by the il2d complex column pass over the N2/2+1
  columns; 3D is rows = N1·N2 followed by the il3d column passes;
- a c2r plan may destroy its input where that buys speed; the plan declares it.

The complex column machinery exists; this record is about the rows and about
how the fold is placed.

## 2. The baseline

Today's library through the front door: even N = zr2c (a c2c(N/2) child and a
separate fold pass), odd N = the promote-to-complex bridge, K>1 = the split
real engines behind the z-doors, 2D = per-row zr2c or ROWSPLIT rows then the
il2d columns. Measured by the gauntlet's real contract (`--real r2c|c2r`,
`--k K`, `--cmp fftw|mkl`; the bench cell `--realfwd/--realbwd/--2drealnat`)
against FFTW 3.3.10 (MEASURE, out of place) and MKL on the same banked plans.
Cell sets: 1D K=1 (196 N: pow2, 2^a·odd, smooth even, primes, odd composites;
2..4096 and 8192..65536, the larger N waiting on their in-place child
verdicts), 1D K=8 and K=32 rows (12 N), 2D (67 shapes). Runs:
`real1d_base_2026-09-29`, `realk_base_2026-09-29`, `real2d_base_2026-09-29`.

Verdict (2026-09-29): at parity or behind FFTW wherever FFTW's SIMD real path
runs, ahead only where FFTW is scalar; ahead of MKL nearly everywhere.
Speedups are ours over the comparator, r2c / c2r:

| Cells | vs FFTW | vs MKL |
|---|---|---|
| 1D K=1, 196 N | median 1.04 (85 below parity) / 0.99 (100 below) | 1.19 (43 below) / 1.13 (47 below) |
| 1D K=1 by class | 2^a·odd 0.92 / 0.87 (34 and 41 of 42 below); even composite 1.00 / 0.94; pow2 1.00 / 0.96; odd composite 1.26 / 1.22; prime 2.13 / 2.07; N ≤ 15 at 0.31–0.58 | odd composite 0.93 / 0.88 is the one class behind |
| rows K=8, 12 N | 0.97 (10 below) / 0.90 (11 below); N=16 rows 0.38 / 0.36 | 1.08 / 0.99 |
| rows K=32 | 0.93 (8 below) / 0.86 (10 below) | 1.10 / 0.91 |
| 2D, 67 shapes | 0.96 (38 below) / 0.89 (48 below); tall pow2 0.85 / 0.82, wide pow2 0.95 / 0.82, the 16-column shapes 0.45–0.60, wide mixed 2.34 / 2.24 | 1.05 (29 below) / 1.19 (13 below) |

The losses are the primitive's cells: the even N whose fold is a separate
pass (2^a·odd and pow2 at parity or below), the tiny N and the 16-column
shapes where the child-plus-fold overhead and the promote bridge dominate,
and the batched rows served by the split engines. The FFTW arm of these runs
planned per process (its 1D c2r control spread 0.75–0.98 is that variance);
later runs replay one wisdom file per run.

## 3. The methods

Each method is a create-time switch, then a raced arm, so the races see
combinations: a method that loses alone may win inside one. The verdict line
names the run that decided it.

| # | Method | Mechanism | Verdict |
|---|---|---|---|
| 1 | Fold in the last stage, 1D (form A / form B) | the Hermitian untangle rides in the row transform's last pass: a mirror-store mid (`t2h`) or a real leaf (`r2z`) plus the mirror-store mid; no fold pass | not built |
| 2 | ZTT-r | ZTT with the untangle in its terminator (`tlfh`); ingest and mids unchanged | not built |
| 3 | Fold radix as a raced axis | the fused stage's radix chosen per cell by the chain race, from a radix-2 fold pass to a fused radix-16 stage | not built |
| 4 | Deferred cross-row fold, 2D/3D | rows as plain c2c(N2/2), the fold as the last column stage's store edge over the 2D mirror pair; the c2c banded walk applies whole; column count N2/2 | not built |
| 5 | Rows as lanes | short rows vectorized across rows (complex mids through the column-stride kinds; the real leaf by gather), raced against along-row | not built |
| 6 | Real odd leaf | the odd blocked emission with real inputs: real s/r pairs, four columns per ymm, half the FMAs; replaces the promote bridge in every rank | not built |
| 7 | One row verdict per cell class | the primitive's wisdom keyed on (N2, row-count class, pitch class), shared by 1D K, 2D and 3D | not built |
| 8 | Rows threaded | the batch split across workers by row range, the TC batch's mechanism | not built |
| 9 | Real DC/Nyquist columns | the two real-valued columns of the 2D/3D half-spectrum transformed as real columns | not built |
| 10 | c2r destroy-input arm | the column pass in place on the caller's spectrum, no scratch plane | not built |

## 4. The protocol

Every measurement is the gauntlet's: core 2 pinned with its SMT sibling held,
one process per cell, both engine orders, best-of-5 in two windows, the
control cell every 100 cells, the same banked plans for both comparators, the
FFTW arm replaying one wisdom file per run (`fftw.wis` beside the store:
MEASURE once per problem, so the comparator holds still across cells,
directions and method races). A method is judged by the class medians and
the count of cells below parity, never by one cell.
