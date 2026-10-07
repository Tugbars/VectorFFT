# fft3d real IL — the rank-3 INTERLEAVED real tier (design of record)

*Declaration of `src/core/il/rank3/fftnd_real_il.h` (DESIGN 2026-10-07, the five
decisions closed with the owner the same day; nothing built). The rank-3 c2c tier (`fftnd_il.h`, [`fftnd_il_design.md`](fftnd_il_design.md))
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
| flat | 2 | THE PAY-ONCE FORM (proposed 2026-10-07 after the owner's objection to paying the order tax per axis): the child's row engine over all N1·N2 rows of the cube into a PRIVATE CCE volume, then the PLAIN in-place column chain on axis 1 per plane (the child's chain tokens, the natural request off: no natural leaf), then axis 0's stages in place on the private volume and its LEAF out of place into the caller's output with the addresses chosen once for both permutations (the plane to its natural plane, each row block of hp3 complex to its natural row). No race of its own beyond the structure race |

Both arms are built at create and raced on the whole transform on scratch (the
2D law since 2026-10-06: at T>1 every arm runs every pass in serving order);
the loser is freed. `VFFT_ILND_ARM=1|2` pins for a probe, never banks.
DECIDED 2026-10-07 (owner, the c2c rule restated): both arms, raced at every
cell, in phase 1.

The two arms are two ways to pay the ORDER TAX. The child pays it per axis as
leaf addressing (axis 1's natural leaf writes rows at natural positions, a
few percent of a column pass; axis 0's natural leaf moves whole planes, near
free) and needs no private volume in r2c; it carries the 2D real tier's
whole-plan forms (the fused walk, the pitch forms, the real axis on N1). The
flat arm pays it ONCE, at the last pass's own write: no natural leaf
anywhere, one private CCE volume in r2c (c2r has it by §2). Neither pays an
extra sweep; neither needs a scrambled real engine at any rank (a scrambled
1D real engine would remove a tax the 1D c2c engines do not pay — natural
order is free there by the shipped verdicts — and turn the fold into a
gather; a "scrambled 2D real engine" is this tier's own column call with the
natural request off). The block-addressed leaf is the existing natural leaf
called per row block of the virtual row with the leg stride N2·hp3 and the
output base at the block's natural row. c2r mirrors: axis 0's leaf gathers
from the caller's volume with the same block addressing into the private
volume in scrambled order, the stages run in place there, then the plain
backward chain and the backward rows per plane into the output. THE
BACKWARD ROWS ARE WRITTEN PER LEAF GROUP, not per row (probe finding
2026-10-07): the rows kernel takes two rows at least, and the scrambled
axis-1 chain's leaf makes the row permutation block-affine — positions
g·Rl + r hold bins b0 + r·N2/Rl — so each leaf group's natural rows are an
arithmetic progression the kernel writes in one call with a descending
output stride; the group that wraps at row 0 takes two calls. An engine per
row or the door route goes row by row as before.

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

THE AXIS-0 FORMS MUST INCLUDE THE STRIP FORM (probe 2026-10-07): the virtual
row's pitch N2·hp3 is a multiple of 4 KB at many cells (16x256x256: 129
pages), where an in-place leaf over the cube thrashes the cache sets — the
in-place leaf served 0.99 of FFTW there, dense column strips (N1 x W complex
gathered into a strip scratch, the chain there, rows scattered to natural
planes; the c2c tier's nf=2 form) 1.26, and the pay-once leaf out of place
1.27; at 8x128x2048 strips 1.06 vs in place 0.95. At the small cubes the
in-place leaf wins (32³ 1.09 vs strips 0.69, 64³ 1.11 vs 0.84). Raced per
cell like every other form; the strips align to row blocks (§7).

Not in phase 1, each its own later form: the FUSED WALK at rank 3 (the planes
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

A scrambled-axis-1 plane with the permutation absorbed by axis 0's
block-addressed write is exactly the flat arm of §3 (the pay-once form); it
needs no scrambled 2D real contract, so no separate "scrambled child" form.

## 6. Contracts and phases

| phase | contract | status |
|---|---|---|
| 1 | R2C, rank 3, howmany 1, OUT OF PLACE, interleaved, DEFAULT/NATURAL (SCRAMBLED refused), any N1,N2 ≥ 2 and N3 ≥ 2 (odd N3 through the row plan's odd door), one thread; a threaded request is served by the serial walk until phase 3 | BUILT 2026-10-07 (`il/rank3/fftnd_real_il.h`; gate `r3d_gate.c`: vs FFTW ≤ 6e-16, replays bitwise with zero races, pins never bank, the 2D borrow fires) |
| 2 | C2R, the same cell (direction-shared row: `s_c2r= nf_c2r= nsw_c2r= wl_c2r=`, the chain tokens shared), input preserved through the private volume; `destroy_input` = axis 0 in place; structures child, pay-once, band (every legal cut an arm) | BUILT 2026-10-07 (gate `r3d_gate_c2r.c`: vs FFTW ≤ 1.2e-15, input preserved, replays bitwise, pins never bank, the destroying twin correct) |
| 3 | MT (§7): the plane arm transposed, strips aligned to row blocks, raced per (cell, T) with the structure, the form and the team; the c2r band arm per worker-owned bands | BUILT 2026-10-07 (gate `r3d_gate_mt.c` at T=8: engaged, MT == ST bitwise, replays bitwise with zero races, both directions) |
| 4 | the later forms of §4; rank 4 | owner's call |

In place is refused (the 2D real tier's law: the in-place real door needs the
padded-pitch caller contract). Everything outside the contract is refused
loudly by the tier; never bridged, never the split engine behind a repack.

## 7. Multithreading (the c2c tier's plane arm, transposed) — DECIDED 2026-10-07, built as phase 3

Two phases per direction, each a pure loop restriction of the serial walk, so
MT == ST BITWISE (the probe gates it). THE STRIPS ALIGN TO ROW BLOCKS: a strip
of the virtual plane is a whole number of row blocks of hp3 columns, so under
the pay-once form each worker's permuted writes land in rows nobody else
writes (and under the child arm a strip never splits a plane row).

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
scratch) is warm from its second plane on. THE BAND ARM, c2r ONLY: in r2c it
cannot be formed (the planes are finished before axis 0 runs); in c2r it is
the forward chain's wide prefix out of place into the private volume, then
per band of L[cut] planes the suffix in place and the band's planes' c2r
while hot. The owner excluded it from the race by the c2c verdict (528 of
534 against) unless the scratch probe's number admitted it — and it did
(2026-10-07, one thread, serving/FFTW): the band form is the best c2r arm at
32³ 1.31, 64³ 1.33 and 128³ 1.15 (the child 1.28 / 1.28 / 1.12), the
pay-once form at 16x256x256 1.54, 8x128x2048 1.10 and 36x20x28 1.18. So the
c2r race carries three structures: child, pay-once, band. The natural
class's cycle and strip forms have no counterpart as ORDER forms (no pass
here permutes the planes); the STRIP FORM does exist as an axis-0 EXECUTION
form (§4).

Clones: a 2D REAL child clone is route-equivalent to the primary iff every
verdict that decides output bits matches (`_ilndr_child_equiv`): the row
engine by its name (the rows kernel or the engine's recipe) and the door
batch's row plan (`_tc_clone_equiv`), the column plan (chain, kernel
pointers, natural form, leaf, staged leaf), the whole-plan forms (fused walk,
real axis, skewed plane, destroying c2r), the column-inverse plane's pitch,
the odd door; stack states are not bits. The pay-once arm's workers run the
clone's row engine and share the axis-1 tables read-only (the plain chain
runs in place: no scratch); each worker owns a strip scratch. The in-place
natural pass of a multi-stage axis 0 (the cube-sized pre-leaf scratch) does
not thread; its arm is excluded from the threaded race. Clones read warm wisdom and never bank; any clone
failure tears that structure's set down and MT declines loudly. The pool is
the one owner; the plan's T is the snapshot. The child is created at
`nthreads = 1`: the 3D tier owns the threads (the 2D real tier's own T>1
forms never run inside a 3D plan).

Banked on the rank-3 real row keyed at the plan's thread count (`nthreads=`):
`cmt= cmtt= cmts= cmtf= cmtp=` (r2c) and their `_c2r` twins plus `cmtw_c2r=`
(the threaded band's width) on the direction-shared row — each direction
carries its own marker (`cmtt`), because the row is shared and the r2c
verdict's presence must not read as a c2r one (a bug caught by the gate
2026-10-07). The threaded verdict's structure and form may differ from the
serial `s=`/`nf=` on the same row; the serial pieces are kept alive through
the threaded race and freed by its verdict. `VFFT_ILND_MT=0|2` and `VFFT_ILND_PT=w` pin; the
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
| `chain= blu= forms=` | axis 0 | the column builder's spelling (the chain, the N-arm, the kernel forms), as the c2c rank-3 row spells them |
| `nf= nsw=` / `nf_c2r= nsw_c2r=` | axis 0's execution form, per direction | 1 in place (the natural pass or the one-stage kernel; the pay-once twin: stages in place, the leaf out of place), 2 strips, with the strip width — the c2c rank-3 row's own names, the same meaning |
| `chain1=` | the pay-once arm's axis 1 | the plain per-plane chain, raced at the plane's size (cheap; the one race the flat arm owns) |
| `s=` / `s_c2r=` | the structure race, per direction | 1 child, 2 pay-once, 3 band (c2r only; `wl_c2r=` its width in planes); the flat arm carries no other verdict: the child's `plane_*` recipe is its row engine's |
| `cmt= cmts= cmtp=` (on the `nthreads=T` row) | the MT race | the c2c tier's spelling |
| `plane_*` | the child | the 2D real child's recipe in role (the child store; `il/wisdom/wisdom2_child.h`) |

TOKEN SPELLING DECIDED 2026-10-07 (the table above): the 2D real row's
spelling for axis 0 and the children, the c2c rank-3 spelling for `s=` and
the MT verdicts; no token the flat arm owns. Axis 0's chain bank creates the
row; every later verdict is a field update.
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

`bench_1d_vs_mkl --3dreal --realfwd|--realbwd` (BUILT 2026-10-07): the real
cell at rank 3, the shape N1xN2xN3 in the N slot, against MKL (DFTI REAL 3D,
CCE strides {0, N2·hp3, hp3, 1}, NOT_INPLACE) or, under `--cmp fftw`, the
runtime-bound FFTW arm (`r2c_3d` / `c2r_3d`; its c2r destroys its input and is
timed on a restored copy as the 2D cell is); `--mt` = the threaded cell at
`$VFFT_MT` under the two-team protocol (the P-cores for both engines, MKL's
OpenMP team created before the caller is pinned, our pool down while MKL runs),
MKL at the same T, the `engaged` column = the rank-3 tier's threaded executes
in the timed arm (0 = a serial verdict). The gauntlet: `--real r2c|c2r --group
3d-real [--threads T]` (the pow2 cube grid 4..256 per axis
up to 2^20 points plus odd-N3 cubes; 3D shapes in `--cells` too), the row
reader on `wisdom2_3d.txt`'s `t=r2c` row, `recal_1d_probe --r2c|--c2r --3d`.
`VFFT_ILNDR_PROF=1` (bound at create) prints the planes' and axis 0's ns per
execute. Numbers live in the results folders, never here.

## 11. File map

| file | role |
|---|---|
| `il/rank3/fftnd_real_il.h` (BUILT, phase 1) | `vfft_ilndr_t` (axis 0's plain and natural descriptors, the child and its store, the pay-once axis-1 descriptor and private volume, the strip scratch), the walks, the 2D borrow, the structure x form race, `_vfft_create_fftnd_real_il`, execute, destroy |
| `il/rank3/fftnd_il.h` | `_vfft_create_rank34_il` dispatches rank-3 interleaved R2C/C2R here; its clone and race helpers lent where they fit (the strips, the plane team, the race protocol) |
| `il/rank2/il2d_col.h`, `il2d_tier.h`, `il2d_real_plan.h` | the column build/execute and the real row-plan race, lent (no rank-3 code lives there) |
| `il/planning/policy_il.h` | `vfft_policy_ilndr_ok` (the contract), the MT serial-arm law reused |
| `wisdom2/wisdom2_2d_reader.h` | the rank-3 real key (`real=1` on `vw2_ilcol_key_t`), the axis-suffixed banks |
| `vfft_internal.h`, `vfft_execute.h` | `struct vfft_ilndr_s *ilndr` on the plan; dispatch and destroy |
| `docs/design/measurement_arms.md` | E3.x entries for the structure race, axis 0, the MT race |
