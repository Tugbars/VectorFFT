# The 3D interleaved c2c at T > 1: where it wins, where it loses — roadmap item

**Date:** 2026-09-25 · **Status:** OPEN (the form for the large volumes is a design
decision) · **Tier:** `src/core/transforms/fftnd/fftnd_il.h`, design in
[`../design/3D_mt_il_strategy.md`](../design/3D_mt_il_strategy.md) · **Evidence:**
`gauntlet/results/3d-pow2_mt8_2026-09-24/` (the eight-thread grid, 951 of 1,288 cells
before the run was stopped) with its 271 losers re-raced in
`gauntlet/results/3d_mt8_losers2_2026-09-25/` after the wisdom verdicts were made to
replay (commit 65ccf182). The comparator is the engine the gauntlet measures against, at
the same thread count as ours, both out of place, natural order.

## 1. The problem

Threading costs the 3D tier a third of its lead. On the same 951 volumes the median
speedup is 1.56x at one thread and 1.27x at eight, and the drop is uniform across sizes:
the comparator scales better than we do everywhere, not in one class.

| volume | cells | T=1 | T=8 | T=8 below parity | our scaling | comparator's |
|---|---|---|---|---|---|---|
| under 256 KB | 275 | 3.80x | 2.86x | 5 | 1.01x (serial plans) | 1.36x |
| 256 KB to 2 MB | 217 | 1.73x | 1.27x | 25 | 2.67x | 3.32x |
| 2 to 8 MB | 163 | 1.40x | 1.22x | 8 | 5.42x | 6.12x |
| 8 to 32 MB | 159 | 1.28x | 1.15x | 23 | 6.20x | 6.74x |
| 32 to 64 MB | 137 | 1.22x | 0.92x | 89 | 3.37x | 4.74x |
| **all 951** | | **1.56x** | **1.27x**, 1.33x after the re-race of 2026-09-25 | **150**, then 121 | | |

| N1 | cells | T=1 | T=8 | T=8 below parity | our scaling | comparator's |
|---|---|---|---|---|---|---|
| 2 to 8 | 443 | 1.73x | 1.34x | 70 | 1.93x | 2.53x |
| 16 | 133 | 1.37x | 1.15x | 30 | 3.41x | 5.03x |
| 32 to 128 | 341 | 1.45x | 1.24x | 50 | 4.62x | 5.47x |
| 256 and up | 34 of 371 | 2.12x | 1.97x | 0 | 3.15x | 3.61x |

Scaling is each engine's one-thread time over its eight-thread time at the same cell. The
337 unmeasured cells are all at N1 ≥ 256, the class that loses nowhere.

## 2. Where we win

- **Volumes under 256 KB** (275 cells, median 2.86x): the plan stays serial and the
  comparator's threading buys it 1.36x at best. Nothing to do here.
- **Tall volumes, N1 ≥ 256** (median 1.97x, no cell below parity): the long axis-0 chain
  gives the plane arm eight wide column strips of real work.
- **Long-N3 volumes with a two-plane middle** (Nx2x2048, Nx2x8192: 2.75x to 3.54x):
  256x2x2048 runs 399 us against 1,388, faster relative to the comparator at eight threads
  than at one.
- **The middle of the grid** (2 to 32 MB, N1 ≥ 32): 1.15x to 1.33x, eight cells below
  parity, scaling within 10% of the comparator's.

## 3. Where we lose

**The large volumes with a short first axis** (N1 ≤ 16 at 32 to 64 MB, 64 cells): medians
0.53x (N1=8), 0.61x (N1=4), 0.69x (N1=16), 0.81x (N1=2). Every one of these cells stood at
or above parity at one thread. The worst of the grid:

```
 cell          volume   ours (us)   comparator (us)   T=8    T=1
 8x128x2048    32 MB      3349          1154          0.35   1.03
 4x64x8192     32 MB      4302          1877          0.43   1.09
 8x32x8192     32 MB      2631          1159          0.45   1.03
 8x256x1024    32 MB      2720          1194          0.45   1.13
 8x8192x32     32 MB      3929          1775          0.45   1.02
 8x4096x64     32 MB      3518          1587          0.45   1.06
```

**The small volumes with a short plane count** (N1 16 to 128 under 1 MB, about 20 cells):
0.75x to 0.91x, and 128x16x16 at 0.54x on the band arm. Our scaling there is 2.5x to 2.8x
against the comparator's 3.3x to 4.4x.

## 4. The cause, measured twice

Both engines' page accesses, forks and phases were traced on the loser cells, a thread ladder
ran at 1, 2, 4, 6 and 8 threads, and a second round tested each mechanism in isolation, all on
2026-09-25.

**Two defects in our own verdict path come first.** They are not forms; they are wrong rows.

- **The strip width was not banked beside the threaded strips form.** 181 cells of the
  eight-thread grid carry a threaded strips verdict with no width, so the replay served the
  slower cycle form. Forced to the strips form, 26 of 27 sampled cells run faster, up to 3x,
  and the width hardly matters from 32 columns up. That is 28 to 32% of the grid's lost lead,
  most of the 2 to 8 MB class's loss and about half of the small threaded cells'. Two earlier
  findings were this defect: the uneven cycle loads at 32x32x256 and the axis-0 remainder at
  256x8x128. The store code has banked the width since the morning of 2026-09-25; the rows
  need re-racing.
- **The threading race could not see the child structure, and raced the wrong contract.**
  Under recalibrate the flat structure's row plan re-raced the length-N3 cell after the 2D
  child was built, the child's clones then read the new chain and the clone check refused
  them, so every calibrated eight-thread race ran without the child arm and banked flat. The
  race also ran in place on a hot, unaligned buffer whatever the plan's placement. Both fixed
  2026-09-25: the flat structure is built first, and both rank-3 races run the plan's own
  placement on aligned buffers. On the three worst cells the child arm now engages and wins
  two of them in the race's own short protocol; in the fifteen-round measurement the child
  beats the served flat plan on every loser cell, by 13 to 20% on reused input and up to 36%
  on fresh input, two thirds of the fresh-input loss. A 120-cell re-race under the fixed races
  serves the child on 54 of the 60 worst losers, 12% faster at the median, and lifts the 60
  worst width-defect cells onto the strips form; the 120 cells go from a median of 0.79x to
  0.95x.

**The count to beat.** A walk that, for each plane, runs the rows from the input into the
output and then that plane's columns in place in the output while the plane is hot, and after
the last plane makes one pass along axis 0 over the whole output, reads the input once and
touches the output five times: three read-and-write passes in all, with no transpose, no copy
pass and no scratch plane. That count is what the comparator's times at these cells correspond
to.

**Large volumes with a short first axis: bytes per plane, not the threads.** Both engines keep
92 to 97% of their eight threads busy and the clock holds 5.7 GHz at every thread count. Our
DRAM demand per call is 2.1 to 2.6 times the comparator's on these cells and equal on the
control. The in-flight set is not the lever: staging four planes at a time on disjoint workers
gains nothing. What costs is the plane-sized scratch each plane worker sweeps three times for
the natural axis-1 pass: re-using a warm scratch recovers 15 to 30% of the lost lead, and the
per-plane cost at eight workers is about seven plane volumes moved against three. The rest is
structural: our axis-0 pass is a separate full sweep of the volume, followed by a plane phase
that fetches the output again. Our eight-thread walk run on one thread is as fast as our
one-thread plan, so the lost lead is that walk's scaling, not a switch of plan.

**The measurement is part of the number.** The gauntlet feeds the same input on every call.
On fresh input the lost lead falls 30 to 54% at the three large ladder cells and rises at the
three 4 MB cells; 16 to 28% over the seven.

**Small volumes: mostly the width defect, then threading overhead.** Re-raced under the fixed
races, 128x16x16 goes from 0.54x to 1.04x (26 to 14 us), 32x32x16 from 0.54x to 1.27x and
64x64x8 from 0.74x to 1.23x, all on the strips form. What is left of the class is the fan-out:
three thread launches per call where one region would do, a stage split unevenly across threads
and a phase on four of eight threads. Of the 57 serial verdicts at 32 to 128 KB, 42 survive the
fixed probe and 15 flip to threaded.

## 5. What is settled

- Two forms that fuse axis 0 into the plane walk are measured and refuted at one thread and stay
  refuted (the scratch-cube fused natural form, 2026-09-15; the slab form of fused N3-lane strips
  at 4 to 8 lanes, 2026-09-24: half-empty vectors lose 12 to 50%, 16 lanes break even). A gathered
  panel narrower than 16 columns is that refuted form.
- Streaming stores are not the lever, here or on the 2D tall planes: the pass order alone
  reaches the count to beat.
- "Eight full planes exceed L3" was wrong twice over: the loss is neither the plane count nor the
  in-flight set, but the scratch each plane sweeps and the separate axis-0 sweep.
- The clock is not a factor, and neither is fork cost above 256 KB.
- The verdict machinery replays (the strip width beside the threaded form, the rank-3 row's
  forms and structure, the thread count in the row's key since wisdom2 v1.3) and the rank-3
  races run the plan's contract with every arm present. The shipped eight-thread rows predate
  both fixes.

## 6. Roadmap

1. **Re-race the eight-thread verdicts: done 2026-09-25.** 534 cells (the 271 losers and
   every grid row with the width defect) re-raced under the fixed races and the fixed probe:
   the 452 with an earlier record go from a median of 1.10x to 1.23x, below parity 150 to 121,
   below 0.8x 63 to 40; the 82 tall cells measured for the first time stand at 1.53x with three
   below parity. The composite over the 951-cell grid moves from 1.27x to 1.33x. The
   large short-N1 class moves from 0.67x to 0.78x and stays the loser, 59 of 61 below parity:
   the structural remainder. Twenty-three cells lost more than 10% to the eight-thread race's
   short protocol (two repetitions, three rounds at the large cells); a longer race budget
   there is a small follow-up.
2. **The plane phase's scratch.** First re-use: a warm scratch per worker across its planes, 15
   to 30% at the large losers. Then the axis-1 pass as gathered panels of 16 or more columns,
   our dense strip form applied to axis 1, one read and write of the plane for all of axis 1,
   raced in situ at T against the flat pass and the 2D child. Open: the panel at N2 = 8192,
   where 16 columns are a 2 MB scratch, the whole L2.
3. **Planes first, axis 0 last, for a single-stage first axis (N1 ≤ 16).** The structural
   remainder: a plane phase that runs first streams the input once and leaves the output planes
   hot for the axis-0 pass, while our order sweeps the volume for axis 0 and then fetches the
   output again for the planes. Raced after item 2.
4. **The gauntlet's fresh-input arm.** A contract decision: rotate the input and output over at
   least 80 MB of buffer pairs at the large cells, or report hot and fresh side by side. It
   moves the large cells one way and the 4 MB cells the other.
5. **Small volumes: one region.** Axis 0 as dense strips, a barrier, the planes, with a balanced
   plane partition and no idle workers dispatched.
6. **Counted bytes and the full grid.** DRAM bytes per execute from the uncore counters, which
   need an elevated process on this host, for both engines at one and eight threads on the
   loser cells; then the full eight-thread grid including the 337 tall cells the stopped run
   never measured. The tall class is not a demonstrated win: at N2 = 8 the comparator scales
   4.9x to our 3.2x.

Hygiene: the clones' scratch buffers are plain `malloc` where the rule is `VFFT_ZS_ALLOC`.
Dropped: streaming stores on the axis-0 pass, and "256x8x128's axis-0 pass scales only 3.1x",
which was the width defect.
