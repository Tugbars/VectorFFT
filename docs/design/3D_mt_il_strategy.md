# Multithreading the 3D interleaved FFT
### The partition arms, the joint race at T, and the measurement that keeps the verdicts honest

**Scope.** This paper declares how the native interleaved rank-3 c2c tier
(`src/core/transforms/fftnd/fftnd_il.h`) is parallelized on one
shared-memory socket, and — the part that generalizes to every threaded
tier in the library — how a threaded verdict is selected and measured so
that it stays consistent with the single-thread standings. Companions:
[`il2d_real_mt.md`](il2d_real_mt.md) (the 2D real tier, whose partition
this paper ports), [`measurement_arms.md`](measurement_arms.md) (F1.5, the
axis record), [`../roadmap/fftnd_il_design.md`](../roadmap/fftnd_il_design.md)
(the tier), [`../performance/v1_0_results.md`](../performance/v1_0_results.md)
(the numbers). Nothing here applies to the split rank-N tier (`fftnd.h`):
a different layout with a lane axis of its own and its own threading;
the two are never compared.

---

## 1. The problem

A row-major interleaved cube `N1×N2×N3` (N3 contiguous, `z[2i]` real,
`z[2i+1]` imaginary) is transformed by three commuting Kronecker factors:

| axis | pass | data shape |
|---|---|---|
| 0 | a column pass over the virtual plane of N1 rows × (N2·N3) complex | one leg stride of N2·N3 complex, contiguous counts of N2·N3 |
| 1 | a column pass per plane, N2 rows × N3 | leg stride N3, count N3 |
| 2 | a row pass over N1·N2 rows of N3 | contiguous rows |

Axes 1 and 2 together are "the per-plane structure": either a plain 2D IL
child plan per plane or the tier's own axis-1 pass plus a K=1 row plan
(the raced structure, `s=`). Axis 0 may run banded: a wide prefix of
stages, then bands of `wl` planes whose stage suffix is followed at once by
the structure on those planes while they are L2-hot (`wl=`, `cut=`).

The target is one socket: 8 performance cores with private 2 MB L2 each,
a shared L3, one memory system. There is no lane axis to slice, so every
partition is a restriction of the loop nests above over disjoint,
contiguous runs of complex values.

## 2. The partition arms

Both arms are pure loop restrictions of the serving walk: the same kernels
with the same tables run over the same elements in the same per-element
order, only the loop bounds and base pointers change. Threaded output is
therefore bitwise the serial output, and the probe gates it at every cell.

**The BAND arm** (needs a banded axis 0 with at least two bands). Each
wide prefix stage is split on its digit axis: a stage's kernel walks
digits itself, so running digits `[d0, d0+nd)` is three pointer edits
(base, table, group count) and whole planes stay with one worker; one
dispatch per stage, stages ordered. Then workers take disjoint band
ranges: a band is its suffix stages plus the fused structure on its own
planes, exchange-free, because the planes a worker finishes are the
planes it just produced. Backward mirrors the Hermitian chain: bands
first (reversed suffix, then the planes), then the reversed prefix.

**The PLANE arm** (the only arm of an unbanded or Bluestein axis 0).
Workers take disjoint column strips of the virtual plane and run the
whole axis-0 chain on them — barrier-free, because a column pass never
mixes columns; Bluestein windows share their scratch disjointly by column
range. Then workers take disjoint plane ranges for the structure.

Neither arm is a default. An axis 0 that cannot split a prefix stage (a
one-digit stage) runs that stage serial inside the band arm and lets the
race decide whether the arm still pays.

## 3. State: clones per worker, one pool owner

The per-plane structure mutates plan state — a 2D child's scratch, the
row plan's scratch, an axis-1 Bluestein buffer — so worker t > 0 runs a
clone of it. A 2D child clone is accepted only when it is route-equivalent
to the primary (chain, kernel pointers, banded walk, N-arm, natural class,
row route, row plan); a row-plan clone only when `_tc_clone_equiv` says
so, plus its own axis-1 scratch. Clones read warm wisdom and never bank.
Any clone failure tears that structure's set down; a structure without
clones cannot thread, and with no clones at all the verdict is serial —
banked exactly like a yes. Tables are read-only and shared.

The pool has one owner (`support/threads.h`): the plan's thread count is
the snapshot, `stride_pool_workers_for` the one clamp,
`stride_pool_run` the one fork-join (measured: 68 ns at 2 workers, 272 ns
at 8, empty body). A fork-join is not what makes a small cell slow.

## 4. The verdict is raced at T, structure included

The single-thread create races the structure and the band width. That
verdict is the one a one-thread plan replays, and it is not the threaded
winner: at 64³ the two structures tie at one thread, while threaded the
child structure with the band arm runs 70 µs and the flat structure with
the plane arm 103 µs. So a plan created for T > 1 keeps both structures
alive, with their clones, until the threaded verdict, and races

    serial(s0)  +  {band, plane} × {child, flat}

as arms of one race. The winner banks three fields on the cell's rank-3
row: `cmt=` (0 serial, 1 band, 2 plane), `cmtt=` (the T raced at) and
`cmts=` (the structure the threaded verdict runs with). `s=` stays the
one-thread verdict. A verdict serves only at its own T; another T races
again. The losing structure and its clones are freed after the verdict.

## 5. Measuring a threaded arm

Three things were measured on the way that decide whether a threaded
number means anything. Each one produced numbers that were wrong by a
factor of 1.5 to 100 with nothing visibly broken.

**The transient.** After a pool rebuild, or after another arm ran, the
first two to three milliseconds of a threaded plan's executes run 1.5–5×
slower than its steady state (81×27×27, plane arm: round-0 mean 57–223 µs
against a steady 40 µs; 64³, band arm: 234–297 µs means for two rounds
against a steady 128). Each worker's cache partition of the cube has to
settle, and the caller core wakes from whatever the harness did between
arms. A sample that is one execute, or a handful, measures the transient.

Rule: **a threaded sample is REPS executes after warm passes**, REPS sized
from one serial timing to roughly 20 ms of serial-equivalent work, and the
race keeps its alternated min-of-3. A bench sample runs at least 5 ms of
untimed warm executes on every side before its timed reps.

**Two thread teams in one process.** Timing a threaded plan against a
threaded MKL in one process needs all of:

| trap | what happens | the rule |
|---|---|---|
| (a) our spinners | the pool's workers spin forever; seven of them steal seven cores from any MKL arm | tear the library pool down before every MKL sample, through `vfft_set_num_threads(1)` |
| (b) MKL's spinners | its OpenMP team spins `KMP_BLOCKTIME` (200 ms) after a compute | 300 ms of cool before our arm after an MKL arm |
| (c) inherited affinity | the pool pins the caller to core 0 and Windows threads inherit the creator's mask, so an OpenMP team created after our first threaded create lands on one core (3D CCE 16³ at T=8: 34 ms instead of 3 µs) | create MKL's team before any vfft plan pins anything |
| (d) a second pool | a bench TU that includes `threads.h` owns its own copy of the pool state; its startup pool is seven idle spinners on the very cores the library's workers use | never touch the pool through `stride_*` from a TU that is not `vfft.c` |

**Engagement.** A threaded plan that declines runs the serial walk and
reports a serial time under a threaded label. `vfft_ilnd_mt_passes()`
moves once per threaded execute; a result whose counter did not move for
every execute is not a threaded result. The bench prints the count per
cell; the probe asserts it.

## 6. What the standings look like once this holds

Same-run, the create race itself at T=8 (steady-state samples, min of 3):
speedup over the same tier at one thread runs from 1.5× at 27×9×15 to
8.4× at 128³, with the odd cells at 3.6–3.9× and the 512 KB cubes at
2.0–2.4×. Against MKL's CCE arm at the same T=8, the tier wins or ties
at 8 of 11 cells (128³ 2.8×, 64³ 1.5×, 64×128×32 1.5×), and loses at the
two 512 KB cubes and at 256×64×16. The figures live in
`v1_0_results.md`; this paper declares only what decides them.

The remaining losses are named levers, not defaults: the one-thread width
verdict shapes the threaded arms without being raced at T (256×64×16 flips
between `wl=32` and `wl=64` from run to run, and its threaded arms differ
by 60% between the two); and at 512 KB the two fork-joins plus a per-plane
child execute per plane cost more than MKL's single fan-out. Both are
measured items for the tier's next pass.

## 7. Verification

- `build_tuned/benches/ilnd_probe.c`, cold store, twelve cells: each
  structure env-pinned, the banded flat arm memcmp-equal to the unbanded,
  the one-thread verdict, then the same cell at T=8 — output memcmp-equal
  to the one-thread verdict's, engagement 7 of 7 executes. A warm rerun
  replays every verdict with zero races.
- `api_matrix_gate` (3D IL rows), `mt_c2c_gate` (the pool contract),
  `il2d_real_gate` (the 2D tier the clones come from).
- The plan fingerprint carries `ilnd=[arm ax0 ax1 mt=verdict/T/clones]`
  and recurses into the child and row plans.

## 8. File map

| file | role |
|---|---|
| `transforms/fftnd/fftnd_il.h` | the arms (`_ilnd_mt_tramp`, `_ilnd_mt_phase`, `_ilnd_execute_mt`), the clones, the joint race (`_ilnd_mt_race`), the create |
| `transforms/fft2d/il2d_tier.h` | `_il2d_stage_digits_mt` (the digit split), `_il2d_col_pass_range`, `_il2d_blu_cols_range` (lent, no rank-3 code) |
| `wisdom2/wisdom2_2d_reader.h` | `cmt= cmtt=` through the axis bank; `cmts=` through `vw2_ilnd_mts_lookup/bank` |
| `support/threads.h` | the pool owner |
| `vfft.c`, `include/vfft_diagnostics.h` | `vfft_ilnd_mt_passes()` |
| `build_tuned/benches/bench_1d_vs_mkl.c` | `--3dil --mt`, the two-team protocol |
