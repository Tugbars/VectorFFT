# ZTURN-T threaded — the staged walk, sectioned

Design, 2026-09-15. Basis: `cascade_mt_method.md` (the method the deleted
cascade proved: sections are loop-range restrictions, MT is bitwise ST, one
join per stage, a per-T verdict), `oop/il_flatdit_mt.h` (the flat DIT's arm
of 2026-09-07: the unit view, the cut, the bind, the race, the tokens),
`measurement_arms.md` (the "cores sharing one transform" banking class),
`3D_mt_il_strategy.md` (the transient rule and the two-team traps),
`ztt_odd_design.md` (the staged executor this arm sections). It fills the
one hole the cascade's deletion left: T > 1 at N >= 2048 in the K=1
interleaved tier has no engine.

## Terms

- **T** — the plan's thread count (`h->nthreads`), the caller plus T-1 pool
  workers; the pool is the process's one (`support/threads.h`), workers
  spin, never park.
- **unit** — the indivisible piece of one stage: a column (ingest,
  terminator, plain stage 0), a group (a mid), a tile (the tiled prefix) or
  a block (the plain suffix). A thread's range holds whole units.
- **arm** — one way of cutting the walk: `mt=1` BLOCKS (every stage cut by
  units, one fork-join per stage), `mt=2` TILES (the tiled prefix cut by
  tiles, each tile's whole prefix on one thread, depth-first; the sweeps by
  units). The flat DIT's two arms, on ZTURN-T's loops.

## The principle

The staged executor (`_ztt_staged_run`) already contains every parallel
axis: each stage is a loop over columns, groups, tiles or blocks whose
iterations touch disjoint spans and share nothing but read-only tables
(`rb[]`, the group-invariant twiddle streams). Threading it restricts each
loop's range per thread. Nothing is recomputed and no order inside a unit
changes, so **the MT output is bitwise the ST output**, and the gate that
holds it is a `memcmp`. Stages stay ordered with one fork-join each
(`stride_pool_run`, ~100 ns on the spinning pool).

This holds at every cell, pow2 included: at T > 1 a ZTURN-T plan runs the
sectioned staged walk, not its fused codelet. The fusion is worth 0-3%
above 2048 (`ztt_scrambled_design.md`); the threads are worth multiples.
The fused codelets stay the pow2 solution's serial form and nothing else.

## Units and cuts, per stage

Every kernel takes `(zin, zin2, zout, zout2, tw, tw_im, Ls, Gs, OLs, OGs,
count)` with the group loop inside, group pitch `2·R·Ls`, and a column
quad's twiddle record at `tw + (k/4)(R-1)·8`. A thread's range `[lo, hi)`
of units becomes one call with shifted pointers and a shorter count:

| stage (natural) | unit | how the call shifts for `[lo, hi)` | contract |
| --- | --- | --- | --- |
| ingest `t0tp` | column | `zin + 2·lo`, `rb + lo`, `count = hi-lo`; the plane base unchanged (stores go to `rb[c]·R0`, absolute) | `lo, hi` multiples of 4 |
| tiled prefix | tile | tiles `[lo, hi)`: each tile's mids run on one thread as today (`B = W + t·tile·2`) | none |
| sweeping mid `tmg` | group | `B = W + lo·2·R·L`, `Gs = hi-lo`; the stream is group-invariant | none |
| terminator `tlf`/`tlfi` | column | `W + 2·lo`, `zout + 2·lo`, `tw + (lo/4)(R-1)·8`, `count = hi-lo` | multiples of 4 |

| stage (plain) | unit | shift | contract |
| --- | --- | --- | --- |
| stage 0 `t0d` | column | `zin + 2·lo`, `zout + 2·lo`, `tw + (lo/4)(R-1)·8`, `count = hi-lo` (legs at stride `Len_1` from the shifted base; block-split stores at `p·Ls + k`, disjoint per column) | multiples of 4 |
| sweeping mid `tmgd`/`tmgb` | group | as natural | none |
| per-block suffix (mids + `tld`) | block | blocks `[lo, hi)`, each block's suffix on one thread | none |
| untiled `tld`/`tldb` | group of R | `zout + 2·R·lo` (both pointers), `count = hi-lo` | multiples of 4 |
| backward `tlfb` | column | as the natural terminator, in place | multiples of 4 |

Column cuts round to multiples of 4 (the kinds' `count % 4 == 0`); a
thread whose range is empty does nothing. The plane offset
(`_ztt_plane_for`) is computed once by the caller and passed to every
thread: the plane is one buffer per plan and the threads write disjoint
spans of it. No per-thread scratch exists or is needed. Nothing here is
per-cell: the cut is derived from the plan's stage table at bind time, so
it serves the pow2 registry cells and the odd band alike.

## What declines

An arm declines, and the serial walk serves, when a stage would hand a
thread nothing: fewer tiles than threads for the tile arm, fewer groups or
column quads than threads for a sweep, a live pool clamped below the bound
T (`stride_pool_workers_for`), or T < 2. Below 2048 the transform is a few
microseconds and the joins are not free; the race is expected to keep
those cells serial, and the arm is not built there by rule but by verdict.
Nesting is forbidden: the threaded walk runs only on the caller thread of a
plan whose own `nthreads > 1`; a transform-contiguous worker clone is
created at T = 1 and never threads inside.

## The verdict: raced per T, banked on the row

Threading is a raced plan parameter, never a rule. At create, when
`nthreads > 1` and the plan is ZTURN-T, `_ztt_mt_replay_or_race` (the flat
DIT's law, `k1_commit.h`): env pin (`VFFT_ZTT_MT=0|1|2`, never banked) >
the banked `il_mt` at THIS T > the race. The race times `serial`,
`blocks` and `tiles` on the plan's own buffers with the flat DIT's
protocol (warm passes ≥ 5 ms, reps to ~20 ms of serial-equivalent work,
min-of-3, unpaced inside the race: the threaded arm is measured hot, the
parking law), excludes arms that did not engage, and leaves the plan at the
winner. The verdict banks on the cell's `il_route=ztt` row of the plan's
own order class as `il_mt_t=<T> il_mt=<0|1|2>` — the tokens the flat DIT
already banks, so one reader serves both engines — and, for a plan bound
in place, the aliased pair `il_mt_ip_t=<T> il_mt_ip=` (an aliased z→z
walk through the plane is a different measurement from the out-of-place
one; the cascade banked them apart for that reason). A T mismatch
re-races and re-banks; the row's chain and tile are untouched.

Engagement is counted (`vfft_ztt_mt_passes`, the `_vfft_ztt_mt_count`
tentative definition in `vfft.c` beside the flat DIT's): an MT result
without an engagement proof is vacuous.

## Expected numbers (the deleted cascade's, the only prior)

The cascade's sectioned walk on odd chains at T=8: 24576 3.0x, 49152 3.3x,
98304 4.2x; 3072..12288 banked serial. ZTURN-T's stages are the same
shapes with fewer sweeps, so the spike expects the same band: gains from
~16384 up, growing with N; parity or decline below. Against MKL at T=8
(`bench_1d_vs_mkl --k1noop --mt`, one process per cell, P-cores pinned,
MKL's team created before our pin, 300 ms cool) the target is to keep the
1.5-2x lead the serial cells hold.

## Gates

1. **MT == ST bitwise.** At every T in {2, 4, 8}, both arms, both order
   classes, both placements, untiled and every ladder width, at pow2 cells
   (2048..262144, a chain per size) and odd cells (the odd gate's list):
   `memcmp` against the serial walk. The serial walk is already bitwise the
   fused codelet at pow2, so MT is bitwise the fused form too.
2. **Engagement.** Every threaded execute increments the counter; a cell
   whose verdict is threaded and whose counter does not move is a FAIL.
3. **Roundtrip** through the threaded backward, both placements.
4. **Replay.** A second create at the same T replays the verdict without a
   race; a create at another T re-races.
5. **Speed.** The spike cells (12288, 98304, 245760, both classes, in
   place and out) on the house protocol for MT: the arm must beat serial
   where the cascade did, and never be banked where it loses.

## Build order (1-5 DONE 2026-09-15)

`oop/ztt_mt.h` (included by `vfft.c` only): the cuts, the three dispatch
shapes, both arms, the race; plan fields `mt`/`mt_t`; the stage table
resolved for every plan; `vfft_ztt_mt_passes`; `_ztt_mt_replay_or_race` in
`k1_commit.h` with the tokens above, called from both doors' exits;
`_ztt_serve` behind every ZTURN-T dispatch. Gates: `benches/ztt_mt_gate.c`
(1224 threaded executions bitwise the serial walk at T = 2/4/8, both arms,
both classes, both placements, both directions, every width, pow2 and odd;
engagement; roundtrip), the odd gate's threaded front-door pass (T = 8
bitwise the T = 1 plan, engaged, a second T = 8 create replays, T = 4
re-races), the pow2 gate's T = 8 arm reading both counters. Measured:
`probes/ZT/zt_mt_spike_results.md` — T = 8 speedups 1.4-2x at 12288..16384,
3.3-5.4x at 65536, 4.3-6.1x at 98304, 6.2-7.9x at 245760..262144; against
MKL at T = 8 through the canonical bench 1.51-1.87x at 12288..262144.

1. **The sectioned walk.** `vfft_ztt_mt_bind(p, T)` derives the per-stage
   cuts from the stage table (both arms); `vfft_ztt_execute_mt(p, zin,
   zout, bwd)` runs them through `stride_pool_run`, one dispatch per stage
   (blocks) or per prefix (tiles), and returns 0 when it declines. Every
   plan resolves its stage table at create (the fused cells too); `staged`
   keeps selecting the serial form.
2. **Gate 1-3** in `benches/ztt_mt_gate.c`, before any door change.
3. **The race and the bank** (`_ztt_mt_race`, `_ztt_mt_replay_or_race`) and
   the counter; the OOP and in-place doors call it where they call the
   flat DIT's; `_k1x_ztt` probes the arm before the serial call, arming the
   pool; `_tc_inner_mt_safe` unchanged (clones are serial by creation).
4. **Gate 4** and the `k1_pow2_gate` / `ztt_odd_gate` T=8 arms read both
   counters.
5. **The spike** (gate 5) and the MKL run at T=8; the results doc and
   `v1_0_results.md`.
6. **Records.** `include/vfft.h`'s threading paragraph names the arm;
   `measurement_arms.md` B4a gains the ZTURN-T row.

## Assumptions stated for the owner's ruling

- The tokens are the flat DIT's (`il_mt_t`, `il_mt`, plus the `_ip` pair);
  no new token family.
- Both arms of the flat DIT are built and raced; a third shape (a
  2D-style column split inside a tile) is not, until measured wanting.
- Below 2048 the arm exists and races like anywhere else; the race, not a
  floor, keeps small cells serial.
