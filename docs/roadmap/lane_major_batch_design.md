# Lane-major 1D batches on the column pass (c2c, r2c, c2r)

**Status (2026-10-14): step 1 BUILT (the door and c2c); steps 2 to 5 not started.**
Owner: "we will build lane major everywhere; the library is almost done, so it may
start now. 1D C2C / R2C / C2R all will have lane major with one bold stroke." This
document is the design to build against; it supersedes slice 2 of
`il_real_lane_major_batch.md` (2026-10-03) and leaves its slice 1 (the real leaf over
lanes at N <= 64) as a possible later arm. The owner's ruling of the same day: the two
geometries are a contract in an ordinary create and race only in the max-performance
planning mode, later; no stride feature.

Step 1 as built: `il/rank1/lm_batch_il.h` (the door, the bound executor, the cell's
race of the serial lane tile and the threaded form), the create branch in
`il/il_create.h`, the lean-door binding, the plan flag `lm`, the destroy and the route
name (`lm`, `lm+blu`, `lm+tpc`), the law `vfft_policy_batch_lane_major` with its
policy-gate mirror over the request table, the API matrix rows, and the row: the 1D
cell at `q = K` on the layout axis value `lay=lm` (placeholder name, the owner's to
confirm), carrying the column tokens; the chain bank's new-record path keys it the same
way. The regression `src/tools/gates/lmbatch_gate.c` (115 checks, cold): eight cells
from 16x8 to 1031x2 against K separate K = 1 plans and the naive DFT, the scrambled
roundtrip, in place, the replay bitwise; T = 8 at 1048x256, 131x1024 and 256x512
engaged (the strips with their tile at the wide planes, the blocks at 1048x256) and
replayed bitwise; the refusals named. At a prime N with two lanes the turned pass wins
the race and the row carries its child recipe (`tpc_*`), as on a 2D row.

## 1. The request and the data contract

A 1D interleaved batch of K >= 2 transforms in the LANE-MAJOR geometry
(`vfft_config_t.batch_geom`, include/vfft.h): element e of transform t at
`z[2 (e K + t)]` for complex data, `x[e K + t]` for real samples, bin f of transform t
at `z[2 (f K + t)]`, f < N/2 + 1. The plane is N rows of K complex (or K reals) at
pitch K: exactly the 2D tier's column plane with N1 = N and N2 = K, and no row pass.

| request | geometry that reaches this door |
|---|---|
| c2c, INTERLEAVED, dims 1, K >= 2 | `VFFT_BATCH_LANE_MAJOR` (explicit; DEFAULT stays transform-contiguous) |
| r2c / c2r, INTERLEAVED, dims 1, K >= 2 | `VFFT_BATCH_DEFAULT` and `VFFT_BATCH_LANE_MAJOR` (the public contract names lane-major as the real batch's DEFAULT once the native engine exists) |

Placement: c2c both (the column pass runs in place on the plane); r2c and c2r out of
place first, in place as the second piece (the padded plane of (N + 2) K doubles, the
CCE convention along the transform axis). Order: c2c NATURAL (the default) and
SCRAMBLED, per lane, the column class's contract (the chain's digit-reversed comb, any
self-consistent permutation); real transforms natural. Threads: any T the pool admits.
The transform-contiguous geometry keeps its own door (the K = 1 plan run K times, its
slabs at T > 1). The two geometries never race each other inside a create: the caller's
buffers fix the layout, and comparing layouts is the planning model's max-performance
mode (`docs/design/design_contracts.md`), which tells the caller which layout to bring.

## 2. The engine is the column pass

The plan owns one column-axis descriptor (`vfft_ilcol_t`, il2d_col.h), built by
`_il2d_col_build` at (N, rn = K) in this plan's role on this plan's own row, executed
by `_il2d_col_exec` (serial) and `_il2d_cols_mt_desc` (threaded: strips over lanes,
blocks over digits). Everything the 2D and 3D column axes have, the batch inherits
with no code of its own: the chain race on the serving walk, the Rader stage at a
prime above the kernel pool with its inner and width race (`rd=` / `rdw=`), the forms
race, the lane tile (`sw=`) that keeps a tile of lanes L1-resident through every stage,
the natural leaf redirection, the banded walk, the column Bluestein and the turned
prime pass at a prime N, and the column MT race (`cmt=`, `mtarm=`, `msw=`, `nls=`).

What is new is the plan kind and its door: a 1D batch handle whose forward and backward
are the column pass, both directions through the descriptor's own tables (the walkers
serve both), with the plan's natural permutation for NATURAL and the comb for
SCRAMBLED.

Why the column pass and not a new kernel set: every lane runs its own transform
through the same butterfly, two complex lanes per AVX2 register with no shuffle between
transforms and one broadcast twiddle per stage and index; every length vectorizes,
primes included (the Rader column cells measured 3x to 7x against MKL on 2026-10-14).
Where a smooth length's in-row engine is already full-width the gain is the removed
shuffles only, and the race per (N, K) says what it is worth; the geometry is the
caller's, so the verdict is banked, never traded for the other geometry.

## 3. The real transforms

**r2c at an even N**, three passes, the real-axis form's own (il2d_real_axis.h, steps
1 to 3, without its rows):

1. PACK: `W[m][k] = x[2 m][k] + i x[2 m + 1][k]`, m < M = N / 2: a complex M x K plane
   (a copy; the two reals of a packed sample sit K apart in lane-major data, so the
   zr2c reinterpretation of a row does not exist here).
2. COLUMNS: the c2c column chain at M down every lane of W, in place, the SCRAMBLED
   class (the chain the column builder serves at (M, K), raced in this plan's role on
   the plan's own row; a prime M takes the column Bluestein or the turned pass).
3. FOLD, per lane, reading the column spectrum through the chain's permutation (which
   un-scrambles it for free): `A[0] = Re Z[0] + Im Z[0]`, `A[M] = Re Z[0] - Im Z[0]`,
   and for k = 1..M/2 `E = (Z[k] + conj Z[M-k]) / 2`, `O = -i (Z[k] - conj Z[M-k]) / 2`,
   `A[k] = E + w^k O`, `A[M-k] = conj(E - w^k O)`, w = e^{-2 pi i / N}: the (M + 1) x K
   bins, natural, straight into the caller's lane-major spectrum.

**c2r at an even N** is the twin, built as its own piece right after r2c (the owner's
rule: r2c and c2r are separate pieces): UNFOLD per lane (the fold's inverse:
`Z[k] = E + i w^{-k} O` with E and O from the bins k and M - k, `Z[0]` from bins 0 and
M), the backward column chain (the descriptor's own tables), UNPACK
(`x[2 m][k] = Re W[m][k]`, `x[2 m + 1][k] = Im W[m][k]`), unnormalized like every
backward in the tree.

**An odd N** has no real column engine (the pack trick needs an even length; the odd
real engines are row engines): the request is refused at create with the reason
printed, a named gap, until the real odd column engine is built with piece C of the
Rader spread (`rader_leaf_design.md` section 13).

## 4. Threads

The chain threads through the column MT forms at the plan's T (its own T row, raced as
the column cells race theirs). The elementwise passes (pack, fold, unfold, unpack) split
by lane ranges across the workers, the strips partition; they thread whenever the chain's
threaded form serves and run serially otherwise. The engagement counters prove a
threaded pass ran; a T > 1 result without that proof is vacuous (the project's rule).

## 5. Wisdom

The cell's own row, keyed by the batch itself: the 1D cell of the transform at `q = K`
(`t=c2c|r2c n=N q=K ord place lay=il [nthreads=T]`) with the geometry on the key, since
the `q = K` interleaved c2c row carries the transform-contiguous wrapper's verdict today
(`eng=tcb tcmt=`) and the two geometries are two cells. The row carries the column
tokens as a column row does (`chain= wl= tf= ro= forms= rd= rdw= blu= tpc= sw= cmt=
mtarm= msw= nls=`); the real row carries the M-chain's tokens (the chain at M, the
direction-shared real row, `rx`-style twins for the backward where a verdict differs).
The geometry axis's name and the row spelling are the owner's (no new token without
asking); the proposal is a key axis beside `lay=`, e.g. `geom=lm`, and no marker on the
transform-contiguous rows (their geometry is the layout's DEFAULT).

## 6. Policy and the gates

`vfft_policy_batch_lane_major(cfg)` in planning/policy_il.h names the admitted
requests (section 1), mirrored by the policy gate over the request space. The door
refuses everything else with the reason printed (an odd N real batch; a rank-2 lane-major
request, whose batches are contiguous planes).

The workstream's own regression, `src/tools/gates/lmbatch_gate.c` (the owner's rule: a
focused workstream gets its own regression, not the old gates): c2c at N in {16, 45,
131, 256, 524, 1031} and K in {2, 3, 8, 64}: natural against the naive DFT at small
cells and against K separate K = 1 plans elsewhere, the scrambled roundtrip, in place
bitwise against out of place, T = 8 engaged and within 1e-12 of T = 1, the replay
bitwise; r2c and c2r at even N against K separate real plans and the naive DFT, the
pair roundtrip, T = 8; the refusals named. The Rader regression's column sections cover
the stage itself. The canonical bench gets a lane-major batch mode against MKL's
strided batch descriptor (NUMBER_OF_TRANSFORMS = K, distance 1, stride K), one thread
and T = 8, the cells above; its numbers go to `docs/performance/v1_0_results.md`.

## 7. Order of work

1. The door, the plan kind, the c2c forward and backward, both placements, both orders,
   the T row, the policy, the gate.
2. r2c at an even N (pack, the M-chain, the fold).
3. The c2r twin.
4. The bench mode, the race against MKL, the results page.
5. The in-place real batch (the padded plane) and, with piece C, the real odd column
   engine.

## 8. What this retires

The two "lane-major lost to transform-contiguous at every measured cell" refusals
(c2c_oop_create_il.h, c2c_ip_create_il.h) are a verdict on the split-convert path of
2026-09, not on a native engine; they go with the door. The V02 refusal of the real
batch's DEFAULT geometry (vfft.c, 2026-10-14) goes when the real door serves. The
split bridge's lane-major crossing was cut with the layout separation; nothing of it
returns.
