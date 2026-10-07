# fft3d real IL — the rank-3 INTERLEAVED real tier (design of record)

*Declaration of `src/core/il/rank3/fftnd_real_il.h` (DESIGN 2026-10-07; nothing
built). The rank-3 c2c tier (`fftnd_il.h`, [`fftnd_il_design.md`](fftnd_il_design.md))
and the rank-2 real tier (`il/rank2/`, [`fft2d_real_il_design.md`](fft2d_real_il_design.md))
are the two pieces this tier is made of. The split rank-N real tier
(`split/rank3/fftnd_r2c.h`) is a different layout and stays untouched: the two
share nothing. Rank 4 composes the same way, later.*

## 1. Thesis

A row-major interleaved real cube `N1×N2×N3` (N3 contiguous) transforms to the
CCE volume `N1×N2×hp3`, `hp3 = N3/2+1` complex bins per row, in interleaved
layout. That is three Kronecker factors, and two tiers already implement every
shape they take:

| axis | what it is | machinery |
|---|---|---|
| 2 | the REAL row pass over N1·N2 rows of N3 reals → hp3 complex | the 2D real tier's row plan (`rx=`: the rows kernel, a real engine per row, the odd-N2 route, the door batch) |
| 1 | a c2c column pass over each of the N1 planes of N2 rows × hp3 | the 2D real tier's column plan (`cx=`: the one-kernel leaves, the chain's natural forms) with pitch hp3, per plane |
| 0 | a c2c column pass over the virtual plane of N1 rows × (N2·hp3) complex | the same column plan with pitch N2·hp3 — unchanged kernels, chains, tables; wide over the cube |

Unlike c2c, the passes do not all commute with the fold: the real pass must be
FIRST in r2c and LAST in c2r. So the walk is fixed, and the structure question
is only how the two c2c axes are finished.

## 2. The walk

**r2c:** PLANES FIRST, then axis 0. Each plane of the real cube (N2 × N3 reals)
transforms out of place into its CCE plane in the caller's output (N2 × hp3):
the real rows and axis 1, by the structure of §3. Then axis 0 runs IN PLACE on
the output, wide over the cube. No scratch beyond the structure's own.

**c2r:** AXIS 0 FIRST, then planes. The caller's CCE volume is preserved (the
2D real tier's contract), so the backward axis-0 pass runs OUT OF PLACE from
the caller's volume into the tier's CCE scratch volume (N1·N2·hp3 complex, the
input's size); then each scratch plane transforms into its real plane of the
caller's output by the structure's backward (the 2D real c2r: the plane's
axis 1 into the child's own column-inverse plane, the backward rows into the
output). Under the `destroy_input` permission the axis-0 backward runs in place
in the caller's volume and the scratch volume is not allocated — the 3D form of
the 2D destroying c2r (one kernel law: the per-plane structure is unchanged).

Memory: r2c = the plan's children only; c2r = one CCE volume (2× the input's
size in flight) unless `destroy_input`. DECIDED 2026-10-07 (owner): the
private volume is the default c2r form, `destroy_input` the caller's opt-in
to the in-place axis 0. (FFTW's default is the reverse: its c2r destroys the
input, and its multi-dimensional c2r cannot preserve it; ours is stricter.)

## 3. The structure is a raced arm (owner 2026-09-26: "race both at every occasion")

| arm | `s=` | per plane |
|---|---|---|
| child | 1 | the 2D REAL door's plan on (N2, N3), out of place, order as requested, one thread — the real rows and axis 1 with every rank-2 real verdict (rx, cx, the fused walk, the real axis on N1 at odd N3, the pitch forms, the c2r twins) raced as a standalone 2D real transform on its own rank-2 real cell, under the `plane_` child store |
| flat | 2 | the CHILD'S VERDICTS RECOMPOSED CUBE-WIDE (owner 2026-10-07): the child's row engine run over all N1·N2 rows of the cube at once (the rows kernel at the cube's count, the engine per row over the cube, the door batch over N1·N2 rows seeded from the child's `rp_` recipe), then the child's axis-1 chain and forms run plane by plane, then axis 0. No race of its own: it differs from the child only in COMPOSITION (cube-wide passes against per-plane passes), which is what the structure race measures |

Both arms are built at create and raced on the whole transform on scratch (the
2D law since 2026-10-06: at T>1 every arm runs every pass in serving order);
the loser is freed. `VFFT_ILND_ARM=1|2` pins for a probe, never banks.
DECIDED 2026-10-07 (owner, the c2c rule restated): both arms, raced at every
cell, in phase 1.

What the flat arm can win that the child cannot: the row pass batched over
the whole cube (N1·N2 rows at once: the door's slabs, the rows kernel's
count). What the child wins: the 2D real tier's whole-plan forms (the fused
walk, the pitch forms, the real axis on N1) which exist only on a 2D real
plan. The c2c tier gave its flat arm an axis-1 chain raced next to axis 0;
this tier forgoes that so nothing at N3 or at (N2, hp3) is raced twice and
the 2D row-plan race is never generalized to N1·N2 rows; if the structure
race keeps picking flat at the big cubes, an own axis-1 race is a later
piece. The create then races only what is new at rank 3: axis 0, the
structure, the MT arms.

## 4. Axis 0: the 2D real column plan over the virtual plane

Axis 0 is `_il2d_col_build` over (N1 rows, N2·hp3 complex per row) with the 2D
real tier's column-plan race in role (`cx=`): the one-kernel leaves at
N1 ≤ 128 (n1c / b816 / b448, their backward twins for c2r), else the chain
in its forms — natural strided (the leaf scatters WHOLE PLANES: the virtual
row is a plane, so the natural scatter is a contiguous plane write), natural
staged, and the scrambled chain for SCRAMBLED. The banded walk (`wl=`) is
raced as the 2D real tier races it: bands of planes, the per-plane structure
OUTSIDE the walk (in r2c it ran first; in c2r it runs after). Bluestein axes
(prime N1) stay unbanded as in the c2c tier.

Not in phase 1, each its own later form: the SCRAMBLED CHILD (§5), the FUSED WALK at rank 3 (the planes
of stage 0's digit-d set {d + j·N1/R0} produced straight before the digit's
butterfly — the 2D real fused walk with planes for rows; L3-only there), the
c2r band fusion (the plane structure inside the band suffix, legal in c2r
only), the real axis on N1 or N2 for a prime N3.

## 5. Order

NATURAL ONLY (DECIDED 2026-10-07): DEFAULT and NATURAL are the one cell
(`ord=nat`; policy L4 as in the c2c tier); an explicit SCRAMBLED request is
refused loudly until the scrambled output contract is integrated into the 2D
real door (owner: later in the development). Scrambled output does not exist
for ANY real transform today: the front door refuses an explicit SCRAMBLED
order for every r2c/c2r request, 1D and 2D (vfft.c, the order gate: "r2c/c2r
are inherently natural-order"); the 1D IL real engines are natural-only with
their wisdom key pinned to ord=nat; the 2D real branch's scrambled chain path
is unreachable from the door. Natural order on axis 0 comes
from the column plan's natural forms (§4), on axes 1 and 2 from the
structure's own 2D verdicts. There is no cycle walk: the c2c tier needed one
because its axis 0 ran first and scrambled the planes; here the planes are
finished before axis 0 touches them, so axis 0 itself decides the plane order.

THE SCRAMBLED CHILD (a later form, owner 2026-10-07): a child whose axis 1
runs scrambled keeps the plain in-place column chain and leaves the same
row permutation inside every plane; a plane row is a contiguous block of hp3
complex and axis 0 never mixes columns, so axis 0 absorbs the permutation by
addressing its output per row block (the natural leaf called per block with
the block's natural row as its output base): same bytes, other addresses, no
extra sweep. Free in c2r (the axis-0 backward pass writes the private volume
out of place anyway, in the child's row order); in r2c axis 0 runs in place,
so it needs an out-of-place axis-0 pass or a block cycle walk with one
N1 x hp3 buffer. Raced as a second child kind inside the structure race,
c2r first, after the scrambled 2D real contract exists and is gated.

## 6. Contracts and phases

| phase | contract | status |
|---|---|---|
| 1 | R2C, rank 3, howmany 1, OUT OF PLACE, interleaved, DEFAULT/NATURAL (SCRAMBLED refused), any N1,N2 ≥ 2 and N3 ≥ 2 (odd N3 through the row plan's odd door), one thread | DESIGN |
| 2 | C2R, the same cell (direction-shared row, `_c2r` tokens), input preserved; `destroy_input` = axis 0 in place | after 1 |
| 3 | MT (§7): the plane arm transposed, raced per (cell, T) with the structure | after 2 |
| 4 | the later forms of §4; rank 4 | owner's call |

In place is refused (the 2D real tier's law: the in-place real door needs the
padded-pitch caller contract). Everything outside the contract is refused
loudly by the tier; never bridged, never the split engine behind a repack.

## 7. Multithreading (the c2c tier's plane arm, transposed) — designed now, built after phase 2

Two phases per direction, each a pure loop restriction of the serial walk, so
MT == ST BITWISE (the probe gates it):

| direction | phase A | phase B |
|---|---|---|
| r2c | workers take disjoint PLANE RANGES and run the per-plane structure (worker t > 0 on its CLONE) | workers take disjoint COLUMN STRIPS of the virtual plane and run the whole axis-0 chain in place (barrier-free: a column pass never mixes columns) |
| c2r | the strips: axis 0 backward from the caller's volume into the scratch volume (or in place under `destroy_input`) | the plane ranges: the structure's backward from each scratch plane into the output |

Arms raced at the plan's T, whole transform, on scratch, TOGETHER WITH THE
STRUCTURE (the structure that wins at one thread is not the one that wins
threaded): plane × {child, flat} × {full team, half team} plus serial on a
small cube only (`vfft_policy_ilnd_mt_serial_arm`, the c2c law). The half
team (`cmtp=`) exists for the same reason as in c2c: a worker's plane-sized
scratch (the child's column-inverse plane and stagings, the flat arm's axis-1
scratch) is warm from its second plane on. The band arm is not raced (the
c2c tier's verdict: 528 of 534).

Clones: a 2D REAL child clone is route-equivalent to the primary iff every
verdict that decides output bits matches — the row plan (`rx=`: engine
token, the rows kernel or the engine's recipe, `_tc_clone_equiv` on the
door batch), the column plan (`cx=`: chain, kernel pointers, natural form,
stack states are not bits), the fused walk, the real axis on N1, the
pitch forms, the destroying form; the flat arm's clones are row-plan clones
(the 2D real tier's `il2d_rxw` machinery) plus their own axis-1 descriptors
sharing the tables. Clones read warm wisdom and never bank; any clone
failure tears that structure's set down and MT declines loudly. The pool is
the one owner; the plan's T is the snapshot. The child is created at
`nthreads = 1`: the 3D tier owns the threads (the 2D real tier's own T>1
forms never run inside a 3D plan).

Banked on the rank-3 real row keyed at the plan's thread count (`nthreads=`):
`cmt= cmts= cmtp=` as the c2c tier spells them, both directions on the
direction-shared row. `VFFT_ILND_MT=0|2` and `VFFT_ILND_PT=w` pin; the
engagement counter is the c2c tier's (`vfft_ilnd_mt_passes()`); a threaded
number without it is vacuous. Measurement: every race sample runs REPS
executes after warm passes (the c2c tier's protocol: the cache partition
settles over the first milliseconds); the longer race budget at ≥ 4 MB.

## 8. Wisdom

One row per cell in `wisdom2_3d.txt`: `t=r2c n=N1xN2xN3 q=1 ord=nat|scr
place=oop lay=il [nthreads=T]`, DIRECTION-SHARED as the 2D real row is (the
c2r plan is the r2c plan's twin over the backward passes; each direction's
own token set on the one row):

| tokens | owner | meaning |
|---|---|---|
| `chain= blu= wl= cx= cxs=` / `cx_c2r= cxs_c2r=` | axis 0 | the column plan's chain, N-arm, band width, form and stack state, spelled as the 2D real row spells them |
| (none) | the flat arm | it carries no verdict of its own: the child's `plane_*` recipe is its recipe; `s=2` says it runs cube-wide |
| `s=` | the structure race | 1 child, 2 flat |
| `cmt= cmts= cmtp=` (on the `nthreads=T` row) | the MT race | the c2c tier's spelling |
| `plane_*` | the child | the 2D real child's recipe in role (the child store; `il/wisdom/wisdom2_child.h`) |

Axis 0's chain bank creates the row; every later verdict is a field update.
THE WISDOM LAW (owner 2026-10-07): the 3D create MAY READ the 2D shard's
banked rows — the child's rank-2 real cell (N2, N3), the flat arm's row cell
at N3 and its axis-1 column cell — to seed a child store whose parent row
carries no recipe yet, so a cell the 2D tier has already calibrated is not
re-raced inside the 3D create; it NEVER WRITES the 2D shard (the children's
stores persist nothing; `wisdom_write = 0`), and the WHOLE PLAN — axis 0, the
structure, the children's recipes under their prefixes, the MT verdict — lands
on the rank-3 row in `wisdom2_3d.txt`. The structure race (child vs flat) is
the in-role race and always runs on a miss; a borrowed 2D verdict is the
child's plan inside it. Token spelling = the owner's (no new token without
asking).

## 9. Verification

`fftnd_real_probe.c` (cold scratch store, then warm): per cell and direction
the output against FFTW's `r2c_3d` / `c2r_3d` (max relative error ≤ 1e-14;
the roundtrip c2r(r2c(x)) = N1·N2·N3·x), each structure arm env-pinned and
checked against FFTW on its own (the two arms are not bitwise each other:
different kernels), the served verdict, then the warm replay bitwise with
zero races; at T>1 the same T plan with the pool at T vs shrunk to one =
bitwise, the engagement counter moving once per execute. Cells: 16³, 32×16×64,
27×9×15 (odd N3), 36×20×28, 64³, 128×64×32, 8×128×2048 (the c2c loser class).
`api_matrix_gate`: 3D r2c/c2r OOP IL served; in place, howmany 2 and SPLIT
layout refused.

## 10. Measurement

`bench_1d_vs_mkl --3dreal` (DFTI REAL 3D CCE NOT_INPLACE, both directions,
out of place, natural, the two-team protocol at T) and `bench_1d_vs_fftw`'s
twin (`r2c_3d` / `c2r_3d`; FFTW's c2r destroys its input: timed like the 2D
cell, input restored outside the timed call). The gauntlet group `3d-real`:
the pow2 grid plus odd-N3 cells. Numbers live in the results folders, never
here.

## 11. File map

| file | role |
|---|---|
| `il/rank3/fftnd_real_il.h` (NEW) | `vfft_ilndr_t` (axis 0 descriptor, the child, the flat arm's row plan and axis-1 descriptor, the c2r scratch volume, the clones), the walks, the structure race, the MT arms, `_vfft_create_fftnd_real_il` |
| `il/rank3/fftnd_il.h` | `_vfft_create_rank34_il` dispatches rank-3 interleaved R2C/C2R here; its clone and race helpers lent where they fit (the strips, the plane team, the race protocol) |
| `il/rank2/il2d_col.h`, `il2d_tier.h`, `il2d_real_plan.h` | the column build/execute and the real row-plan race, lent (no rank-3 code lives there) |
| `il/planning/policy_il.h` | `vfft_policy_ilndr_ok` (the contract), the MT serial-arm law reused |
| `wisdom2/wisdom2_2d_reader.h` | the rank-3 real key (`real=1` on `vw2_ilcol_key_t`), the axis-suffixed banks |
| `vfft_internal.h`, `vfft_execute.h` | `struct vfft_ilndr_s *ilndr` on the plan; dispatch and destroy |
| `docs/design/measurement_arms.md` | E3.x entries for the structure race, axis 0, the MT race |
