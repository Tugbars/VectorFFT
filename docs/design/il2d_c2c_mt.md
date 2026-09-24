# The threaded 2D complex transform on interleaved data
### The walks, the two races at T, the verdict rows, and the protocol that measures them

**Scope.** How the native interleaved 2D c2c tier (natural order, either placement, K=1)
runs on T > 1 threads, and how every threaded decision is selected by measurement at the
thread count it serves. Implementation: `src/core/transforms/fft2d/{il2d_tier.h,
il2d_cols.h, fft2d_create.h, il2d_col.h}`, `src/core/vfft_execute.h`; the verdict rows in
`src/wisdom/wisdom2_2d.txt` (`src/core/wisdom2/wisdom2_2d_reader.h`). Companions: the routes
and the one-thread races in [`il2d_c2c_strategy.md`](il2d_c2c_strategy.md); the strip width
ladder in [`il2d_large_plane_design.md`](il2d_large_plane_design.md); the leaf's staged and
strided forms in [`il2d_natural_leaf_design.md`](il2d_natural_leaf_design.md); the verdict
classes in [`planning_model.md`](planning_model.md); the real tier's threading and the
selection methodology in [`il2d_real_mt.md`](il2d_real_mt.md); the open tall-plane form in
[`../roadmap/il2d_tall_planes_rows_first.md`](../roadmap/il2d_tall_planes_rows_first.md).

---

## 1. The problem at T threads

A plane of N1 x N2 complex points (16 B each, row-major) is transformed by a column pass
(an N1-point transform down each of the N2 columns) and a row pass (an N2-point transform
along each row); the passes commute. The one-thread tier owns three routes for the plane
(`il2d_c2c_strategy.md`): the **chain** (the column chain of radix stages, then the rows),
the **skewed column pass** (`csk`: a single column stage into a scratch at pitch N2 + 8,
the rows moving it to the destination) and the **turn** (the rows turned into an N2 x N1
scratch, its rows through the 1D plan at N1, one back-turn).

At T > 1 the same routes serve, each through a threaded walk that is a loop restriction of
its serial walk over disjoint ranges, so a threaded result is bitwise the serial one. The
machine is one socket of eight performance cores with a private 2 MB L2 each and a 36 MB
L3; the thread pool pins worker t to logical core 2t and reserves logical 0 for the caller,
who is worker 0. `T` is the plan's thread count (`cfg.nthreads`, clamped by the live pool
that `vfft_set_num_threads` sized).

## 2. The threaded walks

| route | threaded walk | phases (a barrier after each) | per-worker state |
|---|---|---|---|
| chain, **block** (`mtarm=0`) | the prefix stages digit-split across the whole plane, then the leaf by block ranges with the row transform fused in the leaf (staged: each finished row leaves as one streaming store) | one per prefix stage + the leaf | the row-plan clones, the leaf staging |
| chain, **strips** (`mtarm=1`, `msw`) | each worker a range of columns, walked in sub-strips of `msw` (0 = the whole range), every sub-strip through the worker's **dense strip scratch** (one contiguous N1 x w block at pitch w, the 3D tier's strip form); then the rows as row slabs | 2 | the dense strip scratch (T x N1 x swcap), the row-plan clones |
| chain, **tile** (`mtarm=2`) | stage 0 digit-split across the plane (its digit is the row within a first-stage sub-problem), then every first-stage sub-problem (N1/R0 rows) a tile one worker owns: the middle stages in place on the tile, its leaf blocks with the rows fused | 2 | the row-plan clones, the leaf staging |
| chain, scrambled class | bands of `wl` columns over workers (suffix stages + fused rows per band), or the strips above | 1 or 2 | the row-plan clones |
| csk | the column stage over 16-lane column blocks into the skewed scratch; the rows over row slabs (route 0 through a clone of the out-of-place row plan; the batched routes through their stateless kernels on per-worker slots) | 2 | the csk row-plan clones (route 0) |
| turn | the turned rows over 4-row slabs into the N2 x P scratch; the scratch rows (the plane's columns) through clones of the N1 plan; the back-turn over destination row slabs | 3 | the N1-plan clones |
| Bluestein column axis (prime N1) | the window pipeline over column ranges, then row slabs | 2 | the row-plan clones |
| turned prime pass (`tpc`) | serial | — | — |

The dense strip scratch is sized at create for the widest strip a worker can run: the
ladder's widest admissible width (8, 16, 32, 64, 128, 256 columns; N1 x w x 16 B within L2,
at least T strips across N2) or the unsized whole range when it fits L2. A sub-strip wider
than the scratch falls back to the shared full-pitch scratch. Every walk bumps the tier's
engagement counter (`vfft_il2d_col_mt_passes()`), which a measurement reads before and
after its timed executes: a plan whose counter did not move ran serial.

**Clone sets.** A worker other than the caller runs its own copy of whichever 1D plan the
route's rows need (the serving row path mutates shared plan state): the in-place row child
at N2 for the chain, the out-of-place row plan at N2 for csk's route 0, the in-place plan
at N1 for the turn. `_il2d_clone_set` builds T - 1 clones from warm wisdom (never banking),
checks each route-equivalent to the primary, and tears the set down on any failure, after
which the walk declines and the counter shows it. At T > 1 the sets of all three routes are
built before the axis race and the unneeded ones dropped after it.

## 3. The two races at T

The plan at T is decided by two measurements at create, on aligned scratch planes in the
cell's own placement, each arm through the very code execute serves with.

**The axis race at T** (`_il2d_axis_race`, the same function as at one thread). Every
route arm runs twice: in its threaded form (the chain under the dense strips, the walk that
dominated the block arm at every plane measured) and in its serial form (`+s`), so a plane
the threading race will keep serial still picks its route by the serial walks. A natural
cell prunes the band widths and sub-strip tiles the threaded walk ignores. Two rounds of
`1e6 / (N1 N2)` executes per arm, alternated, two untimed passes first. The winner is banked
**beside** the one-thread verdict as the T verdict: `axt=` (the T raced at), `rot= wlt=
swt= rbkt= turnt= cskt=` (route, band width, strip width, row tile, turn, skewed pass) and
`axns=`; the one-thread tokens `ro wl sw rbk turn csk` stay the one-thread race's. A create
at that T serves it; any other T re-races; the row is created chain-only first when the
cell's chain row does not exist yet (a recalibrating create).

**The threading race** (`_il2d_c2c_mt_race`) for the route the axis race picked: the serial
walk against the route's threaded forms: for the chain the block, the unsized strips, one
strips arm per ladder width, and the tile, each with the staged and the strided leaf
(`nls`); for the turn and csk their one walk. Min of three, two untimed passes first, never
paused (a threaded arm parks its workers when paused). The verdict lands on the existing
row through the field-update path, so the axis tokens survive: `cmt=` (threaded or not),
`cmtt=` (the T raced at), `mtarm=` (0 block, 1 strips, 2 tile), `msw=`, `nls=`, `mtns=`
(the winning time). A create serves `cmt` only at `cmtt`; an axis race at T > 1 invalidates
the banked `cmt`, which was raced for the old route.

**Serving order at create.** The chain and its forms (the one-thread races or their
replay) -> the T verdict's route when `axt` matches the plan's T, else the one-thread
route and, at T > 1, the axis race -> the clone sets and the dense scratch -> `cmt` at
`cmtt`, else the threading race. Pins for probes, read at create only: `VFFT_IL2D_NO_COLMT`
(0 forces threaded, else serial, no race), `VFFT_IL2D_MTARM` (the arm), `VFFT_IL2D_DENSE=0`
(the shared scratch), `VFFT_IL2D_PHASES` (per-phase times on stderr), `VFFT_IL2D_LOG`
(every arm of every race).

A row after both races at T = 8 (512x128, natural, out of place):

```
chain=8.4.4.4 wl=0 tf=0 ro=0 sw=0 rbk=8 turn=0 csk=0
axt=8 rot=0 wlt=0 swt=0 rbkt=8 turnt=0 cskt=0 axns=76293
cmt=1 cmtt=8 mtarm=1 msw=16 nls=1 mtns=58000
```

The first line is the one-thread verdict, the second the route at eight threads, the third
its threaded form: the strips at 16 columns with the staged leaf.

## 4. The laws the design rests on

- **Every threaded decision is measured at its T and served only there.** Column threading
  is the cores-share-one-transform class: how the work is cut depends on T, so a verdict
  carries the T it was raced at (`axt`, `cmtt`) and another T re-races.
- **A threaded walk is a loop restriction of the serial one.** The same kernels on disjoint
  ranges, the same values in the same order: MT == ST bitwise, which is the gate.
- **No environment read on an execute path.** A pin is bound at create into a plan field
  or once into a static; eight workers calling `getenv` at a phase start serialise on the
  C runtime's environment lock (512x128 at T = 8: 54-72 us with two such reads, 32 without).
- **The strip's scratch is dense and private.** A strip walked through the shared
  full-pitch scratch puts its pieces in the input rows' L2 sets and streams every stage
  through L3 (the column phase of 512x128 at T = 8 took 73 us for eight 16-column strips
  against a 40-us serial column pass; dense, 27).
- **Verdicts bank without erasing.** The threading race writes through the field-update
  path; a measured bank rebuilds the row and drops the axis race's tokens.
- **A measurement is valid only under the threaded protocol** (section 5).

## 5. The protocol that measures

A process that races or times threaded arms confines itself to the eight performance cores
(`bench_pin_pcores` in `gauntlet/sibling_guard.h`: mask 0x5555, `VFFT_PCORE_MASK`
overriding, 0 = the unmasked control), pins the caller to logical 0 (the core the pool
reserves for it; pinned to core 2 the caller shared a physical core with worker 1 and every
barrier waited for the pair), raises its priority, and sizes the pool before the create. It
warms every timed arm untimed first (the races: two passes; the bench: at least 5 ms), never
paces inside a sample, and takes the engagement counter over the timed executes as the
proof that the plan threaded. The one-thread protocol (core 2 with the SMT-sibling guard)
applies at T = 1 only. The calibrate probe (`gauntlet/recal_1d_probe.c`) and the bench's
threaded cell modes (`--2dilnat --mt`, `--3dilnat --mt`, `gauntlet/bench_1d_vs_mkl.c`)
both run this way; the comparator runs at the same T in the same process.

## 6. The measured state (2026-09-25, T = 8, the pow2 grid's 77 losing cells re-raced)

| class | on the grid | with this design |
|---|---|---|
| serial routes (csk 54 cells, turn 27) | 0.17-0.96 | threaded: 16x8192 1.40, 8192x16 3.08, 64x1024 1.50 |
| mid planes (N1 128-512, N2 64-512) | 0.42-0.97 | 512x128 1.11, 128x128 1.29, 256x256 1.21, 256x64 1.35 |
| tiny planes (up to 32x32) | 0.91-0.99 | serial by verdict, 0.89-0.97 (sub-microsecond) |
| tall short-N2 planes (8192x128, 4096x256, 8192x256) | 0.52-0.85 | 0.65-0.96: a traffic count, the roadmap's rows-first form |

Of the 77, 7 remain below 1.0 and 1 below 0.8. The comparator threads every cell at T = 8,
including 64x64.

## 7. Where things are

| what | where |
|---|---|
| the threaded walks and their dispatch | `_il2d_c2c_mt`, `_il2d_c2c_mt_tramp` (modes 0-12), `_il2d_csk_exec_mt`, `_il2d_turn_exec_mt` in `il2d_tier.h`; the dispatch in `vfft_execute.h` ahead of the serial branches |
| the dense strip pass and scratch | `_il2d_col_pass_nat_strip` (`il2d_cols.h`), `_il2d_nat_sscr_build` / `natsscr natswcap natdense` (`il2d_tier.h`, `il2d_col.h`) |
| the clone sets | `_il2d_clone_set`, `_il2d_c2c_build_clones`, `_il2d_c2c_build_clone_sets_all`, `_il2d_c2c_drop_unneeded_clones` (`il2d_tier.h`); the fields `il2d_roww il2d_cskw il2d_turnw` (`vfft_internal.h`) |
| the axis race at T and its bank | `_il2d_axis_race` (`il2d_tier.h`); the T verdict read `il2d_axmt` (`fft2d_create.h`) |
| the threading race and its bank | `_il2d_c2c_mt_race` (`il2d_tier.h`); `vw2_2d_il_chain_bank`, `vw2_2d_il_tok_seti/geti` (`wisdom2_2d_reader.h`) |
| the engagement counter | `vfft_il2d_col_mt_passes()` (`vfft.c`) |
| the protocol | `gauntlet/sibling_guard.h`, `gauntlet/recal_1d_probe.c`, `gauntlet/bench_1d_vs_mkl.c`; the gates `il2d_*_gate`, `mt_c2c_gate` (`build_tuned/benches`) |
| the records | `gauntlet/results/2d-pow2_mt8_2026-09-24` (the grid), `mt8_losers_2026-09-24`, `mt8_still*_2026-09-2[45]` (the losers re-raced) |
