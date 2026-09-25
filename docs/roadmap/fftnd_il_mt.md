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

## 4. The cause

**Large volumes: the plane phase's footprint, not the sweep count.** The phase log of
8x128x2048 at eight threads (`VFFT_ILND_PROF=1`) reads 0.8 ms for the axis-0 pass, which
moves 64 MB at 80 GB/s and is fine, then 2.8 ms for the plane phase: eight 4 MB planes, one
per worker, each a 2D child that takes 1.3 ms when it runs alone. Eight children at once
take 2.15 times longer than one, because eight planes with their staging scratch exceed
the shared L3 and every child's column gather then comes from memory. The comparator's
whole transform takes 1.17 ms, under one and a half memory sweeps of the volume: it does
not run the short axis-0 pass and the planes as two full-volume phases.

**Small volumes: the fan-out.** At 128x16x16 the threaded axis-0 pass takes 30 us where the
serial pass takes 12, and the plane phase 14.5 us, the comparator's whole transform. Two
fork-joins plus one child execute per plane cost more than the work they distribute, and
the band arm's axis-0 pass runs column ranges at full pitch, the shape the 2D tier
replaced with dense per-worker strips.

## 5. What is settled

Two forms that fuse axis 0 into the plane walk are measured and refuted at one thread and
stay refuted; the problem above is a footprint at eight threads, not the sweep count they
were built against, but any new form must be argued on its own count:

- the scratch-cube fused natural form (2026-09-15: a cube write plus cold destination
  writes; lost 3% to 27% at every one-thread cell and 13 of 14 threaded cells);
- the slab form of fused N3-lane strips (2026-09-24: half-empty vectors at 4 to 8 lanes lose
  12% to 50%; 16 lanes break even).

The verdict machinery is complete: the plane and band arms race at T with the serial form,
the strips form banks its width beside the threaded verdict, the rank-3 row replays every
verdict (commit 65ccf182), and the calibrate probe measures threaded arms under the pinned
protocol. The losses above are measured forms, not measurement.

## 6. Roadmap

1. **Measure the plane phase's footprint first.** At 8x128x2048 and 8x8192x32: each clone's
   staging scratch and the per-child L3 traffic (the phase log per worker), so that the form
   decision below rests on a count, not on the 2.15x alone.
2. **The plane child's route raced in situ.** The child is the 2D plan's own one-thread
   verdict, raced alone in a quiet cache; at eight threads it shares L3 with seven neighbours.
   Race the child's route and strip width inside the 3D race with the clones running, and
   bank the verdict on the rank-3 row beside `cmts=`. Cheap: the clone sets exist.
3. **The axis-0 pass with streaming stores at 32 MB and up.** The one-thread lever of
   2026-09-24 (the NT-store column kind), now worth up to the 0.8 ms of the axis-0 phase.
4. **The form for short N1 over large planes** — the owner's design decision, after item 1:
   a walk that never has eight full planes hot at once, argued on its memory count against
   the comparator's one and a half sweeps, raced like every arm, refused if it does not win.
5. **Small volumes: one parallel region.** The axis-0 pass and the plane phase under one
   fork-join with a barrier, and the dense strips as the band arm's axis-0 pass. Gate: bitwise
   the serial plan, then the losing cells through the gauntlet's threaded cell protocol.
6. **The full grid at eight threads** once every losing class has a form (the owner's rule),
   including the 337 cells the stopped run never measured.
