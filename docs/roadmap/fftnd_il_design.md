# fftnd IL c2c — the rank-N INTERLEAVED tier (design of record, living)

*Declaration of `src/core/transforms/fftnd/fftnd_il.h` (2026-09-06). The split
rank-N tier (`fftnd.h`, K-lane batched split axes) is a different layout with a
different axis model and stays untouched; the two share a directory and nothing
else. Rank 3 is native today; rank 4 composes the same way.*

## 1. Thesis

An interleaved row-major cube `N1×N2×N3` (N3 contiguous) is three Kronecker
factors, and the 2D IL tier already implements the two shapes they take:

| axis | what it is | machinery |
|---|---|---|
| 0 | a COLUMN pass over the virtual plane of N1 rows × (N2·N3) complex | the 2D column-axis pass (`il2d_col.h` descriptor, `_il2d_col_build` / `_il2d_col_exec`) with pitch N2·N3 — unchanged kernels, chains, tables; wide over the cube |
| 1 | a column pass over each of the N1 planes of N2 rows × N3 | the same pass, pitch N3, per plane |
| 2 | the ROW pass over N1·N2 rows of N3 | the K=1 IL row plan, in place, natural |

Every pass commutes with every other, so forward and backward run the same
order: axis 0 `src → dst` (the out-of-place move is stage 0's; the stage kinds
are alias-tolerant), then per plane in place on `dst`. No transposes, no
layout conversion, no split machinery anywhere on the path.

## 2. The structure and the band width are ONE raced set

How a plane is finished is not an architectural default (owner, 2026-09-06:
"only racing both to each other can tell"). Two structure arms, both built
at create, times every legal axis-0 band width (§2a), every configuration
an arm of one alternated race of the whole forward on a scratch cube (min
of 3), the losing structure freed:

| arm | `s=` | per plane |
|---|---|---|
| child | 1 | a plain 2D IL c2c plan on (N2, N3), in place, order as requested — axis 1 and the rows with every 2D verdict (chain, forms, band width, row route) raced as a standalone 2D transform on its own rank-2 cell |
| flat | 2 | this tier's own axis-1 column pass (its chain raced in the 3D context by the same build function) followed by the K=1 row plan over the plane's rows |

### 2a. The axis-0 banded walk

A "row" of the virtual N1 × (N2·N3) plane is a plane of the cube, so the 2D
tier's banded column walk (E1.2) applies unchanged: the wide prefix stages
`0..cut-1` over the cube, then per band of `wl` planes the stage suffix
depth-first followed at once by the per-plane structure on those planes
while they are L2-hot (the 2D `tfuse`). Backward mirrors the Hermitian
chain (per band the reversed suffix then the planes, then the reversed wide
prefix). Same kernels, tables and count as unbanded: the output is BITWISE
identical (checked by the probe). Width pool = `{8,16,32,64,128,256}` plus
the chain's own stage spans gated by live L2 residency (`w·plane·16 ≤ L2`),
each filtered by `wl | N1` and a suffix stage with `L_s | wl`; the cut is
derived from the width (the tcut law). Odd axes usually offer only their
spans (9 at 27, 45, 81). Bluestein and natural axes stay unbanded.

`VFFT_ILND_ARM=1|2` and `VFFT_ILND_WL=w` pin for a probe (never bank).
`VFFT_IL2D_LOG` prints the `[ilnd]` create lines (every arm's ns, the
structure and width sources).

### 2b. Multithreading (the 2D tier's INC-C at rank 3)

The strategy paper is `docs/design/3D_mt_il_strategy.md` (the arms, the joint race at T,
the measurement rules); this section is the tier's summary of it.

Two partition arms, both pure loop restrictions of the serial walk (no
arithmetic changes, so MT output is BITWISE the serial output; the probe
gates it), RACED against serial at create per (cell, T) TOGETHER WITH THE
STRUCTURE — the structure that wins at one thread is not the one that
wins threaded (64³: child + band 70 µs, flat + plane 103 µs, a tie at one
thread), so at T > 1 both structures stay alive with their clones until
the threaded verdict: arms = serial (one-thread structure) + {band,
plane} × {child, flat}. Banked:

| arm | `cmt=` | partition |
|---|---|---|
| serial | 0 | — |
| band | 1 | the wide prefix stages digit-split over the workers (whole planes per digit, one dispatch per stage), then workers take disjoint BANDS of `wl` planes: suffix stages + the fused per-plane structure, exchange-free. Needs a banded axis 0 with at least two bands. |
| plane | 2 | workers take disjoint COLUMN STRIPS of the virtual plane and run the whole axis-0 chain (barrier-free: a column pass never mixes columns; Bluestein windows share their scratch disjointly), then disjoint PLANE ranges for the structure. The only arm of an unbanded axis 0. |

`cmts=` names the structure the threaded verdict runs with (it may differ
from `s=`, the one-thread verdict, which stays what a T=1 plan replays).

Every race sample runs REPS executes after two warm passes, REPS sized
from one serial timing to ~20 ms of serial-equivalent work: a worker's
cache partition settles over the first milliseconds of executes (round-0
means 1.5–5× the steady state at every cell measured), and single-execute
samples alternating between arms time that transient, never the steady
state (measured 45³: single-execute race band 49 / plane 51 µs, steady
state band 45 / plane 52; 81×27×27 single-execute plane 131 µs, steady 40).

The per-plane structure mutates plan state, so worker t > 0 runs a CLONE:
a 2D child clone that must be route-equivalent to the primary (chain,
kernel pointers, band, N-arm, natural class, row route, row plan), or a
row-plan clone (`_tc_clone_equiv`) plus its own axis-1 Bluestein scratch.
Clones read warm wisdom and never bank; any clone failure tears the set
down and MT declines, loudly — never a half-cloned dispatch. The pool is
the one owner (`support/threads.h`): the plan's T is the snapshot,
`stride_pool_workers_for` the one clamp, `stride_pool_run` the one
fork-join. `cmtt=` is the T the verdict was raced at; a banked verdict
serves only at its own T. `VFFT_ILND_MT=0|1|2` pins (never banks);
`vfft_ilnd_mt_passes()` is the engagement counter, and a threaded number
without it is vacuous; `VFFT_ILND_PROF=1` prints per-phase ns.

## 3. Wisdom

One row per cell in `wisdom2_3d.txt`: `t=c2c n=N1xN2xN3 q=1 ord=scr place=oop lay=il`.

| tokens | owner | meaning |
|---|---|---|
| `chain= blu= forms=` | axis 0 | the column pass's chain, N-arm and forms verdicts, spelled exactly as the 2D row spells them |
| `wl= tf=` | axis 0, the joint race | the banded walk's width (0 = unbanded) and its fusion flag |
| `chain1= blu1= forms1=` | axis 1 (flat arm) | the same verdicts with the axis as suffix |
| `s=` | the joint race | 1 = child, 2 = flat |
| `cmt= cmtt= cmts=` | the MT race | 0 serial, 1 band, 2 plane; the T raced at; the structure the threaded verdict runs with |

Axis 0's chain bank creates the row; every later verdict is a field update
on it. The child's verdicts live on the child's own rank-2 cell, never
copied. DEFAULT and SCRAMBLED are one serving and one cell (`ord=scr`);
NATURAL will be its own cell (`ord=nat`), never compared with it.

The N-arm (column-axis Bluestein) verdict banks as a field update when the
row exists and the bank carries no measurement — before 2026-09-06 that bank
built a fresh record, the measured row was kept, and the verdict re-raced on
every create wherever no axis race re-banked it (in 2D the axis race masked
this; at this tier's axis 0 nothing did).

## 4. Contracts and phases

| phase | contract | status |
|---|---|---|
| 2 | C2C, rank 3, howmany 1, OUT OF PLACE, order DEFAULT/SCRAMBLED, one thread | SHIPPED 2026-09-06 |
| 4 | MT: band arm vs plane arm vs serial, raced per (cell, T), clones per worker | SHIPPED 2026-09-07 |
| 3a | IN PLACE: the same plan and row serve both placements (every pass alias-tolerant; output bitwise the out-of-place output) | SHIPPED 2026-09-07 |
| 3b | NATURAL order (its own `ord=nat` cell): axis 0 stays scrambled, the per-plane pass writes each finished plane to its natural position along the digit-reversal cycles with one plane of buffer — `docs/design/3D_natural_il_design.md` | SHIPPED 2026-09-07 |
| 5 | real 3D (r2c/c2r) | after 3 |
| 6 | rank 4 (axis 0 wide, then per plane the rank-3 tier or the flat form, raced) | after 5 |

Anything outside the shipped contract is refused loudly by `_vfft_create_fftnd_il`
(`fftnd_create.h` dispatches rank-3 INTERLEAVED C2C there; real and rank 4
INTERLEAVED keep the old loud refusal). No bridge, no split fallback.

## 5. Ordering contract

DEFAULT/SCRAMBLED output: each column axis digit-reversed by its own chain
(axis 0 by `chain=`, axis 1 by the child's chain or `chain1=`), the rows
natural — the 2D contract applied per axis. A consumer that needs the bin
address finds it exactly as the 2D consumer does, per axis.

## 6. Verification

- `build_tuned/benches/ilnd_probe.c` (cold scratch store): per cell the DC
  identity, the roundtrip `bwd(fwd(x)) = T·x`, and a naive-DFT spot bin
  searched across the two digit-reversed column axes at its natural row
  column — each structure arm env-pinned unbanded, the flat arm pinned at a
  legal width (its output memcmp-equal to the unbanded one), then the raced
  verdict; a second run on the warm store must show `src=wisdom` and zero
  races. Cells: 16³, 32×16×64, 27×9×15, 36×20×28, 64³, 128×64×32.
- `api_matrix_gate`: 3D c2c OOP IL 16³ DEFAULT and SCRAMBLED, 9×15×27, and
  IN PLACE 16³ and 9×15×27 SCRAMBLED are served; NATURAL and howmany 2 are
  refused.
- The probe's sixth and seventh passes create the cell IN PLACE at T=1 and
  T=8: their outputs must be memcmp-equal to the out-of-place verdict's.
- The probe's fifth pass creates the cell at T=8: its output must be
  memcmp-equal to the T=1 verdict's, and the engagement counter must move
  once per execute (`engaged=7/7`).
- The plan fingerprint carries `ilnd=[arm ax0=nst/blu/wl ax1=nst/blu
  mt=verdict/T/clones]` and recurses into the child and row plans.

## 7. Measurement

`bench_1d_vs_mkl --3dil` (env `VFFT_3DIL_CELLS`, `VFFT_3DIL_ROUNDS`): arms
O-NATIVE (this tier), M-inter (DFTI 3D CCE NOT_INPLACE, the yardstick),
M-split (DFTI REAL_REAL NOT_INPLACE, shows CCE is MKL's best), ctl memcpy —
all out of place, median + spread, a delta below the ctl spread is not a
result. The split rank-N tier is not an arm and not a comparison (owner,
2026-09-06: "split is not our concern. IL is what matters"). `--3dil --mt`
runs both sides at T (`VFFT_MT`, default 8) under the two-team protocol
(the `--ilmt` traps plus a third found here: MKL's OpenMP team must be
created BEFORE our pool pins the caller to core 0, or its workers inherit
the one-core mask); the engagement count is printed per cell. Numbers
live in `docs/performance/v1_0_results.md` (the 3D section), never here.

## 8. File map

| file | role |
|---|---|
| `transforms/fftnd/fftnd_il.h` | `vfft_ilnd_t`, execute, destroy, the arm builders, the structure race, `_vfft_create_fftnd_il` |
| `transforms/fftnd/fftnd_create.h` | rank-3/4 create dispatch: INTERLEAVED C2C rank 3 → this tier |
| `transforms/fft2d/il2d_col.h`, `il2d_tier.h` | the column-axis descriptor, `_il2d_col_build`, `_il2d_col_exec`, `_il2d_col_free` (lent to this tier; no rank-3 code lives there) |
| `wisdom2/wisdom2_2d_reader.h` | `vw2_ilcol_key_t` (rank, dims, ord, axis), the axis-suffixed chain/forms bank and lookup, `vw2_ilnd_arm_lookup/bank` |
| `vfft_execute.h` | dispatch (`h->ilnd`, inside the rank≥2 INTERLEAVED branch) and destroy |
| `vfft_internal.h` | `struct vfft_ilnd_s *ilnd` on the plan |
