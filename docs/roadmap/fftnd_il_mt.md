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
| **all 951** | | **1.56x** | **1.27x** | **150** | | |

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

## 4. The cause, measured

Both engines' page accesses, forks and phases were traced on the loser cells, and a thread
ladder ran at 1, 2, 4, 6 and 8 threads, on 2026-09-25.

**The count to beat.** A walk that, for each plane, runs the rows from the input into the
output and then that plane's columns in place in the output while the plane is hot, and after
the last plane makes one pass along axis 0 over the whole output, reads the input once and
touches the output five times: three read-and-write passes in all, with no transpose, no copy
pass and no scratch plane. That count is what the comparator's times at these cells
correspond to.

**Large volumes with a short first axis: the extra passes, not the threads.** Both engines keep
92 to 97% of their eight threads busy and the clock holds 5.7 GHz at every thread count. What
differs is how much each unit of work slows under concurrency: our busy thread time grows
2 to 4x from one thread to eight, the comparator's 1 to 2x. The reason is traffic. We make
five read-and-write passes over the volume where three suffice. The two extra ones are the flat
structure's natural axis-1 pass through a plane-sized scratch plane: plane to scratch, scratch
in place, scratch back to the plane, three sweeps of a plane-sized buffer per plane. At one
thread that is one plane plus its scratch, 8 MB, inside the 36 MB L3, so the passes are nearly
free, which is a large part of the one-thread lead. At eight threads it is 64 MB in flight and
the passes go to DRAM: the modelled traffic is 2.6 to 3.6 times the comparator's. The ladder shows it
directly: our plane phase stops improving at four threads, where four times 8 MB fills L3, and
on both 32 MB cells eight threads are slower than four while the comparator keeps scaling. The control is
16x64x2048, which runs our strip form with no plane-sized scratch and 2 MB per worker: its
plane phase scales 8.1x and the cell keeps its lead.

**About half of the gap on those cells is the measurement.** The gauntlet feeds the same input
on every call. Our 64 MB in-flight set evicts that input between calls; an engine with a
smaller in-flight set keeps it in L3. On fresh input the comparator runs 1.7 to 2.0x slower
and we 1.24 to 1.30x slower, and the lost lead roughly halves.

**Small volumes: threading overhead.** At 128x16x16 we launch threads three times per call
where one region would do; one stage is split unevenly across threads (1.41x) and another runs on
four of eight threads for 30% of the call, so the call is fastest at four threads. At
32x32x256 the cycle loads are uneven by 1.25x.

## 5. What is settled

- Two forms that fuse axis 0 into the plane walk are measured and refuted at one thread and stay
  refuted (the scratch-cube fused natural form, 2026-09-15; the slab form of fused N3-lane strips
  at 4 to 8 lanes, 2026-09-24: half-empty vectors lose 12 to 50%, 16 lanes break even). A gathered
  panel narrower than 16 columns is that refuted form.
- Streaming stores are not the lever, here or on the 2D tall planes: the pass order alone
  reaches the count to beat.
- "Eight full planes exceed L3" as this item first stated it was wrong: the loss is not the
  plane count but the plane-sized scratch that doubles our per-plane footprint and the passes
  through it.
- The clock is not a factor, and neither is fork cost above 256 KB.
- The verdict machinery is complete and replays (the strip width beside the threaded form, the
  rank-3 row's forms and structure, the thread count in the row's key since wisdom2 v1.3). The
  losses above are measured forms, not measurement, except for the hot-input bias.

## 6. Roadmap

1. **The plane phase without the plane-sized scratch.** The axis-1 pass as gathered panels of
   16 or more columns: the worker gathers the columns into a dense N2 x w scratch, runs the whole
   axis-1 chain there with the leaf writing natural order, and scatters back, one read and write
   of the plane for all of axis 1 and a per-worker footprint of plane plus N2 x w x 16 B. This is
   our own dense per-worker strip form applied to axis 1; the codelets exist. Raced in situ at T
   with the clones running, against the flat natural pass and the 2D child. The count to beat at
   8x128x2048: three pass pairs and about 32 MB in flight, which is what the comparator's 1.2 to
   1.6 ms correspond to. Open: the panel
   at N2 = 8192, where 16 columns are a 2 MB scratch, the whole L2.
2. **The gauntlet's fresh-input arm.** A contract decision: rotate the input and output over at
   least 80 MB of buffer pairs at the large cells, or report hot and fresh side by side. No
   eight-thread grid is re-run before it is decided, since the reused input biases every large
   cell against the engine with the larger in-flight set.
3. **Small volumes.** First the verdict fix: the 57 serial verdicts at 32 to 128 KB were raced by
   the probe that pinned its caller onto worker 1's core; re-race them under the fixed threaded
   protocol (worth the class median 2.9x to 3.7x). Then one region: axis 0 as dense strips, a
   barrier, the planes, with a balanced plane partition and no idle workers dispatched.
4. **Planes first, axis 0 last, for a single-stage first axis (N1 ≤ 16).** The same pass count
   as our order, argued on residency alone: a plane phase that runs first streams the input once
   and leaves the output planes hot for the axis-0 pass, while our order re-reads the output
   after the axis-0 pass has streamed the whole volume through L3. Raced after item 1, only if the plane
   phase still spills.
5. **Counted bytes and the open cells.** DRAM bytes per execute from the uncore counters (VTune
   reaches them from an elevated process on this host) for both engines at one and eight threads
   on the loser cells, replacing the modelled 2.6 to 3.6x; the eighteen-cell timing and bandwidth
   ceilings; and 256x8x128, where our axis-0 pass scales only 3.1x and is 64% of the call.
6. **Hygiene.** The clones' scratch buffers are plain `malloc` where the rule is `VFFT_ZS_ALLOC`;
   the 3D race runs in place on a `malloc`'d buffer while the product runs out of place.
7. **The full grid at eight threads** once items 1 to 3 have forms, including the 337 tall cells
   the stopped run never measured. The tall class is not a demonstrated win: at N2 = 8 the
   comparator scales 4.9x to our 3.2x.

Dropped: streaming stores on the axis-0 pass. The pass order reaches the count without them,
the axis-0 phase is a fifth of the call, and a streamed output would leave the volume cold for
the plane phase that reads it next.
