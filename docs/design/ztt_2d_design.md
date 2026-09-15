# ZTURN-T in the 2D and 3D interleaved tiers — the row axis, through the door

Design, 2026-09-15. Basis: the survey of `transforms/fft2d/fft2d_create.h`,
`il2d_tier.h`, `il2d_cols.h`, `fftnd/fftnd_il.h` (2026-09-15), the owner's
09-13 ruling ("2D/3D pow2 is going to be handled by zturn-t and zturn-t
scrambled too"), `ztt_odd_design.md`, `ztt_mt_design.md`, and the standing
2D/3D numbers in `docs/performance/v1_0_results.md` (the native interleaved
tier beats MKL CCE 10 of 10 in 2D and 11 of 11 in 3D, single-thread).

## What the tiers are, and where a 1-D engine can enter

The 2D interleaved c2c plane is row-major with N2 contiguous. The tier owns
**one kernel family: the column pass** (`vfft_ilcol_t`: `t2c` twiddled
stages and an `n1c` leaf, vectorized ACROSS columns — a lane is a column,
the count axis is j). It owns **no row kernels**: the row pass is a child
1-D plan created through the public front door (`fft2d_create.h:292/305/
369/399`, `il2d_tier.h:2098/2253`: `dims=1, n=N2, howmany=1, layout=IL,
order=NATURAL, nthreads=1`, in place, or out of place under the `ro=`
verdict) and executed one row at a time. Columns run first, rows last, in
both directions; the passes commute. The 3D tier is the same shape one
level up: axis 0 and axis 1 are column passes of the same family, axis 2
is a K=1 row plan at N3 through the door (`fftnd_il.h:959`), or a 2-D child
per plane (`:931`) whose rows are again the door's.

So a ZTURN-T plan enters a plane exactly where a K=1 natural interleaved
cell banks it: **the row axis, by inheritance**. Since 2026-09-09 that is
every pow2 row length from 2048 to 262144 (and 1024 and 16 by verdict), and
since 2026-09-15 every 2^a·odd row length in the band. Nothing in the 2D or
3D tiers names an engine; nothing there needs a change for the rows.

The column axis is a different geometry. ZTURN-T's kinds hold four adjacent
complexes of ONE transform per register; the column kernels hold four
COLUMNS at one element index. A ZTURN-T column pass would need a transpose
in and out (the four-step form), which is what the native column pass was
built to avoid — and that pass is the reason the tier beats MKL CCE 10/10.
The column axis stays the tier's own family. This is the one ruling in
this doc: **ZTURN-T serves the rows of a plane; the columns are the 2D
tier's kernels.**

## What that means for the two order classes

2D natural = natural N1 (the `nat` column leaf) and natural N2 (the row
child is always created `VFFT_ORDER_NATURAL`). The row child's ZTURN-T is
the natural class with the `tlf`/`tlfi` terminator. 2D scrambled = N1 in
the column chain's digit order, N2 natural. The scrambled ZTURN-T class
(the plain schedule) does not enter here: the tier's contract keeps N2
natural in both classes, and a scrambled row would change the tier's
permutation with no measured gain to buy. If a 2D-scrambled-rows class is
ever wanted it is its own contract and its own doc; nothing here builds it.

## Threading

The plane's threads are the tier's: column strips, bands, and row SLABS
with one serial row clone per worker (`_il2d_c2c_mt` modes; the clones are
created at `nthreads=1`, `wisdom_write=0`, and must be pool-free). A
ZTURN-T row plan created at `nthreads=1` never binds its own threaded arm,
so the clone law holds by construction and the two threading schemes never
nest. ZTURN-T's own MT arm is for the 1-D contract; inside a plane the
rows are parallel across rows, which is the better cut.

## What must be measured (the design's only open question)

Whether the rows being ZTURN-T moves the 2D and 3D cells. The 08-25 2D
table (rows served by the pairs, chain3 or the cascade at the time) is the
datum; the same cells through the canonical bench today, with the row
child's engine logged, is the measurement. The long-row cells are where it
can show: 16×4096, 32×1024, 512×512, 1024×1024, and the 3D cells whose N3
is in the band. Short-row cells (N2 = 64) are the pairs' and unchanged.

The 2D row references its child's verdict through the child's own 1-D cell
(the 2D row banks `chain= wl= tf= ro= cmt=`, never the child's route), so a
K=1 calibration run that re-banks ZTURN-T rows re-serves every plane that
uses that length with no 2D re-race — and no 2D restamp is needed either
way, except where a `ro=` (row out of place) verdict was raced against a
row engine that no longer serves.

## Measured (2026-09-15, `bench_1d_vs_mkl --2dil`, single thread, scratch store)

The row child's engine, from the door's log: ZTURN-T at N2 = 1024 and 4096
(the store's own replay), the pairs at 64, 256 and 512. Ours (the native
interleaved tier, in place) / MKL CCE in place, ns; the 08-25 column is the
tier before ZTURN-T served any row:

| cell | rows | ours | MKL CCE | ratio | 08-25 ours / ratio |
| --- | --- | --- | --- | --- | --- |
| 512×512 | pairs | 447,463 | 927,763 | 2.07x | 528,700 / 1.91x |
| 1024×1024 | ZTURN-T | 2,206,587 | 4,809,862 | 2.18x | 2,747,300 / 2.08x |
| 16×4096 | ZTURN-T | 90,763 | 135,127 | 1.49x | 102,410 / 1.50x |
| 32×1024 | ZTURN-T | 40,002 | 69,593 | 1.74x | (not in the table) |
| 64×256 | pairs | 17,955 | 32,263 | 1.80x | 18,457 / 1.91x |
| 4096×64 | pairs | 805-820 k | 864-897 k | 1.08-1.11x | 646,288 / 1.40x |

The ZTURN-T-row cells gained 11-20% in our time against the 08-25 record
with the same column verdicts (the 2D rows replayed unchanged); 512×512
gained 15% from the pair-pool work of 09-09/11. **4096×64 reads 25% slower
than 08-25 on a noisy run** (the control arm's spread 45%; the 2D row is
byte-identical to the shipped one, `chain=32.4.4.8 wl=32`); its rows are
64-point pairs, not ZTURN-T, so this is either the day's noise or the N=64
row cell after the pair-pool sunset — an open item for a quiet re-measure,
outside this design.

## Measured in 3D (2026-09-15, `bench_1d_vs_mkl --3dil`, natural order, single thread, scratch copy of the shipped store)

The bench's cell list gained three long-N3 cells for this: 16×16×4096 and
32×32×4096 (N3 a pow2 band cell) and 8×16×12288 (N3 a 2^a·odd band cell).
The door's log at every create names the axis-2 row plan's engine:
`[k1ztt] N=4096: replay ZTURN-T chain 8.8.8.8 tile=1024 src=wisdom` and
`[k1ztt] N=12288: replay ZTURN-T chain 8.4.4.3.8.4 tile=3072 src=wisdom` —
ZTURN-T by inheritance, no 3D change, in both order classes (the rows are
natural in both). Cool machine, 9 rounds, one-thread samples paced 300 ms,
T=8 unpaced with the engagement counter (162/162, 124/124, 81/81 executes
threaded); roundtrips 6e-16..1e-15. Ours (the native 3D tier, out of
place) / MKL CCE out of place, ns; `~` = inside the control spread:

| cell | order | T | ours | MKL CCE | ratio |
| --- | --- | --- | --- | --- | --- |
| 16×16×4096 | scrambled | 1 | 2,757,100 | 3,511,725 | 1.27 |
| 8×16×12288 | scrambled | 1 | 4,291,150 | 6,990,275 | 1.63 |
| 32×32×4096 | scrambled | 1 | 13,752,100 | 18,261,437 | 1.33 |
| 16×16×4096 | natural | 1 | 3,517,687 | 3,465,250 | 0.99~ |
| 8×16×12288 | natural | 1 | 4,875,012 | 6,787,563 | 1.39 |
| 32×32×4096 | natural | 1 | 18,463,900 | 17,418,512 | 0.94 |
| 16×16×4096 | natural | 8 | 443,837 | 457,287 | 1.03~ |
| 8×16×12288 | natural | 8 | 726,375 | 884,075 | 1.22~ |
| 32×32×4096 | natural | 8 | 5,904,212 | 4,195,463 | 0.71 |

Read: the scrambled class wins every long-N3 cell (1.27-1.63x). The
natural class pays the 3D tier's own natural cost (the band fusion it gives
up: 1.28x over scrambled at 16×16×4096, 1.34x at 32×32×4096, the standing
17-66% of `v1_0_results.md`), which puts the two pow2 cells at parity or
behind MKL's natural at one thread and 32×32×4096 at 0.71x at T=8 (ours
scales 3.1x there, MKL 4.1x). That cost sits in the 3D tier's natural
mechanism, not in the row engine; its fused alternative was built, raced
and refuted the same day (`ilnd_natural_fused_design.md`): the cost is
structural to the permuting pass.

## Ruling (owner, 2026-09-15)

Accepted as designed: ZTURN-T serves the rows of a plane by inheritance,
the columns are the 2D tier's kernels, no four-step arm, no scrambled rows.
The 4096×64 reading is taken as thermal (the control arm's spread), not a
logic change: "the logic should be same".

## Gates

The existing ones hold this design: `wisdom2_2d_gate` (create-twice
bitwise coherence, naive-DFT correctness), `ilnd_probe` (3D), `nat2d_probe`,
`il2d_band_race` (banded arms bitwise the unbanded), `il2d_c2c_mt_probe`
(MT bitwise ST), plus the K=1 gates that hold the row engine itself
(`ztt_gate`, `ztt_odd_gate`, `k1_pow2_gate`). One assertion is added to
`ztt_odd_gate`'s front-door pass if the measurement shows a ZTURN-T row: a
2D plane whose N2 is a band cell serves rows bitwise the 1-D ZTURN-T plan.

## Build order

1. **Measure.** `bench_1d_vs_mkl --2dil` at the long-row cells on a scratch
   store with the row child's engine logged; `--3dil` at the cells whose N3
   is in a band. Against the 08-25 / 09-06 tables.
2. **Rule from the numbers.** If a cell improved, the record notes it and
   nothing is built. If a long-row cell did NOT get a ZTURN-T row, find the
   door reason (a banked pair row that pre-dates ZTURN-T at that length is
   the likely one) and let the calibration run re-bank it.
3. **Records.** `v1_0_results.md` 2D/3D sections gain the re-measured rows;
   `include/vfft.h`'s 2D paragraph names the row engine.
