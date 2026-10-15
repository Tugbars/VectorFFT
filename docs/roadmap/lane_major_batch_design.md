# Lane-major 1D batches on the column pass (c2c, r2c, c2r)

**Status (2026-10-15): steps 1 to 4 BUILT (the door, c2c, r2c and c2r at an even N, the
bench mode and the race against MKL), the in-place real batch of step 5, and the FUSED
STRIPS for the real batches (section 3; the eight-thread power-of-two cells that lost:
512x512 r2c from 0.56x to 1.42x of MKL's lane-major batch, c2r from 0.40x to 0.80x, the
composite cells 1.4x to 1.7x faster than before; the results page), and the c2c batch's
SERIAL DENSE TILES under the serial-form law (a Rader-stage chain walks the plane, a
smooth chain races the tile widths alone: the tile won at the widest planes, 16x4096
from 1.06x to 1.74x, tied in the middle and lost at the Rader cells), and step 6's rank 2
(section 8: the 2D c2c batch and the 2D real batch through the 2D doors, raced against
MKL's lane-major 2D batches on the results page; rank 3 open). The real odd column engine
waits for piece C of the Rader spread.**
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

Step 2 as built (the same day): `_vfft_create_lmr_il` in `lm_batch_il.h`, the plan's
`lm_M`, `lm_W`, `lm_perm`, `lm_tw`; the column chain at M = N / 2 built by the column
builder at (M, K) on the real batch's own row (`t=r2c n=N q=K ... lay=lm`, the chain
tokens of the M-chain), its own race of the lane tile and the threaded form; the pack
and the fold as lane-range passes (`_lmr_pack`, `_lmr_fold`: the real-axis form's
kernels with a lane range) dispatched across the workers whenever the chain's threaded
form serves; the executor bound forward-only. The policy admits r2c at an even N >= 4,
out of place, on the DEFAULT or the explicit lane-major geometry; the front door's
refusal names the three cases left (c2r, an odd N, the in-place plane). The gate's real
section: seven cells from 16x8 to 4096x2 against K separate K = 1 real plans and the
naive DFT (1e-16 to 4e-16), the replay bitwise; T = 8 at 2096x256, 262x1024 and 512x512
engaged and bitwise on replay; the refusals named. 173 checks.

Step 3 as built (the same day): the c2r twin in the same create (`_vfft_create_lmr_il`
serves the request's direction on the shared real row), its own executor bound
backward-only: UNFOLD per lane with the fold's halves dropped (that factor two and the
unnormalized M-point backward make the N of the library's backward convention), `Z[0] =
(A[0] + A[M]) + i (A[0] - A[M])`, `E = A[k] + conj A[M-k]`, `O = conj(w^k) (A[k] - conj
A[M-k])`, `Z[k] = E + i O`, `Z[M-k] = conj(E - i O)`, written at the pass row the backward
chain reads row k from; the backward column chain down the lanes in place; UNPACK, the
pack's inverse. The gate's c2r section: the seven cells against K separate K = 1 c2r
plans (1e-16 to 2.5e-15) and the pair roundtrip to N x; T = 8 at the three wide planes
engaged, bitwise on replay. 246 checks. The odd N and the in-place plane remain the
named refusals.

Step 4 as built (2026-10-15): the canonical bench's modes `--lmc2c` (the old `--kzb`),
`--lmr2c` and `--lmc2r` in `gauntlet/bench_1d_vs_mkl.c`, each with `--mt`. A cell times
four arms: ours lane-major; the loop over K one-thread K = 1 plans on contiguous data;
MKL on our lane-major layout (`NUMBER_OF_TRANSFORMS = K`, distance 1, strides K; the real
descriptors CCE, the backward taking the bins' distance as its input); MKL on its
contiguous layout. The row carries the route and, threaded, the engagement count. The
race over 24 one-thread and 12 eight-thread cells is the results page's section
"Lane-major batches" (`docs/performance/v1_0_results.md`): one thread ahead of MKL's
lane-major batch at every cell, 2.9x to 7.3x where the length carries 131 or 191;
eight threads 2.3x to 5.9x there, behind MKL at the power-of-two cells (0.92x c2c
256x512, 0.56x r2c and 0.40x c2r at 512x512).

Step 5 as built (2026-10-15), the in-place real batch: one buffer of (N + 2) K doubles,
the N K reals first and the (N/2 + 1) K bins over them. The passes do not change: the
pack copies the whole input into W before the fold writes a bin, and the c2r unfold
consumes every bin before the unpack writes a real (each phase a whole dispatch). The
policy admits both placements; the gate checks in place bitwise against out of place in
both directions at one and eight threads (286 checks). The odd N stays the named refusal.

First measurement (2026-10-14, `gapfill/work/lm_vs_mkl_probe.c`: one thread, core 2 at
HIGH priority, 15 rounds alternated, 200 ms pace, the min over rounds; MKL's lane-major
batch is its descriptor with NUMBER_OF_TRANSFORMS = K, distance 1, strides K; the two
contiguous arms run the same transforms in the other layout, as a reference):

```
   N      K   ours-lm ns   mkl-lm ns  ratio   mkl-tc ns  ours-tc ns   err vs mkl
   131   1024     305,200   1,082,400   3.55     979,500     406,100   8.3e-16
   524    256     399,900   2,632,800   6.58   2,557,600     494,000   5.8e-16
  1048     64     181,800   1,300,500   7.15   1,270,800     189,800   7.7e-16
  2101     64     500,800   1,580,600   3.16   1,423,400     523,600   7.7e-16
    53   1024      87,300     111,400   1.28     439,375     139,075   1.3e-15
   256    256      84,000      99,700   1.19      68,100      81,300   2.4e-16
    64   1024      72,050      90,950   1.26      38,200      44,650   2.2e-16
    16   4096      60,125      63,025   1.05      28,300      36,200   2.4e-16
```

The real batch the same way (`gapfill/work/lmr_vs_mkl_probe.c`: MKL's DFTI REAL with
CCE storage, NUMBER_OF_TRANSFORMS = K, distance 1, strides K in and out):

```
   N      K   ours-lm ns   mkl-lm ns  ratio   mkl-tc ns  ours-tc ns   err vs mkl
   262   1024     468,100   1,217,500   2.60   1,029,900     430,100   8.5e-16
  1048    256     508,900   2,728,100   5.36   2,613,000     629,500   7.4e-16
  2096     64     225,000   1,338,800   5.95   1,287,900     212,500   8.1e-16
  4202     64     561,300   1,818,500   3.24   1,492,800     605,300   7.4e-16
   106   1024     134,900     513,700   3.81     456,100     163,900   6.0e-16
   512    512     261,500     303,400   1.16     198,700     200,900   2.8e-16
   128   1024      86,500      94,233   1.09      68,800      63,267   4.3e-16
    32   4096      73,160      75,080   1.03      70,480      73,440   1.9e-16
```

And the c2r twin (`gapfill/work/lmc_vs_mkl_probe.c`, the same descriptors backward,
the bins made by MKL's own forward):

```
   N      K   ours-lm ns   mkl-lm ns  ratio   mkl-tc ns  ours-tc ns   err vs mkl
   262   1024     498,500   1,808,200   3.63   1,136,100     481,200   2.3e-15
  1048    256     581,800   3,000,100   5.16   2,711,600     547,000   2.0e-15
  2096     64     241,700   1,394,000   5.77   1,327,400     223,400   2.0e-15
  4202     64     584,300   2,181,800   3.73   1,595,600     592,500   2.2e-15
   106   1024     149,200     687,400   4.61     462,600     168,300   1.9e-15
   512    512     258,400     435,800   1.69     182,500     170,100   7.8e-16
   128   1024      96,600     101,150   1.05      68,150      55,950   5.6e-16
    32   4096      87,880      79,820   0.91      76,420      61,700   4.4e-16
```

Where the length carries a prime above the pool the lane-major batch is 3.2x to 7.2x
MKL's lane-major batch (the real batches 2.6x to 6.0x, the prime in the half length) and
1.0x to 1.3x our own contiguous one; MKL does not exploit
the lanes (its lane-major and contiguous times are equal there). At a smooth length the
lane-major batch is 1.05x to 1.26x MKL's lane-major batch but slower than both
contiguous forms, ours by 1.0x to 1.7x: the column pass at a short smooth N over
thousands of lanes is not where it wins, and the contract says the caller brings the
layout, so the max-performance mode will name contiguous there. Our lane-major
spreads read 18 to 53 percent against MKL's 2 to 27: the min is quoted. The canonical
measurement is step 4's race on the results page.

## 1. The request and the data contract

A 1D interleaved batch of K >= 2 transforms in the LANE-MAJOR geometry
(`vfft_config_t.batch`, include/vfft.h): element e of transform t at
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

**The fused strips** are the form the plain chain class runs in (a Bluestein or turned
axis at a prime M keeps the three passes over the plane). A strip of w lanes goes
through a dense per-worker scratch of M x w complexes at pitch w, in L2: the pack from
the caller's plane straight into the scratch, the chain in place in the scratch through
the plain walker at pitch w (the same kernels, tables and values, so the output is
bitwise the plane form's), the fold from the scratch into the caller's spectrum; c2r is
the mirror. The plane is read once and written once, one dispatch serves the execute,
and the plane W is never walked at its pitch: at a power-of-two K every row sits a power
of two apart, so a sub-strip's rows share one cache-set group and every stage of a walk
on the plane streams instead of running in cache (the chain scaled 2.0x at eight threads
that way; the pitch, not the caller's buffers, was the cause). In place the bins of lane
k sit where the reals of lanes 2 k and 2 k + 1 sit (a bin is two doubles, a real one), so
a strip's fold would overwrite reals of strips not yet packed; the input is therefore
consumed whole first, the pack (the unfold) over W as the three passes do it, and each
strip then reads its lanes from W: one more pass over the plane than out of place. The
serial form walks tiles of `sw` lanes the same way; the threaded form gives each worker
its lane range, cut at eight lanes so no cache line is written by two workers, in
sub-strips of `msw`. The scratch is the column descriptor's dense strip scratch, built
for the scrambled chain as the natural chain builds its own; the same scratch lets a
c2c batch's scrambled chain run its threaded blocks dense (the strip copied in, the
walker at pitch w, the strip copied back). At one thread the c2c batch walks the same
dense tiles, the natural strip with its leaf scatter or the scrambled strip, at the
width the serial race banks (`sw`), under the serial-form law (`policy_il.h`): a chain
with a Rader stage walks the plane and races no width, since the engine's cost hides the
walk's aliasing and a tile's copies only add; a smooth chain races the tile widths alone,
the walk no arm, a scrambled chain's widest tile below K; with no width at all (fewer
than eight lanes) the walk serves. For the
real batch `sw = 0` is the whole plane as one fused tile wherever the scratch holds it,
so the three passes over W serve only a Bluestein or turned axis and a plane no tile of
which fits L2.

**An odd N** has no real column engine (the pack trick needs an even length; the odd
real engines are row engines): the request is refused at create with the reason
printed, a named gap, until the real odd column engine is built with piece C of the
Rader spread (`rader_leaf_design.md` section 13).

## 4. Threads

The chain threads through the column MT forms at the plan's T (its own T row, raced as
the column cells race theirs). In the fused strips each worker runs its lane range whole,
the pack and the fold inside it; in the three passes over the plane the elementwise
passes (pack, fold, unfold, unpack) split by lane ranges across the workers, the strips
partition. They thread whenever the chain's threaded form serves and run serially
otherwise. The engagement counters prove a threaded pass ran; a T > 1 result without that
proof is vacuous (the project's rule). The race's arms for a real batch run the whole
executor (the pack and the fold are part of what a width decides), on planes sized for
the bins' extra row.

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

## 8. Rank 2 and 3 (step 6)

A lane-major 2D batch is K planes of N1 x N2 side by side, element (r, c) of transform
t at `z[2 ((r N2 + c) K + t)]`. Every axis is a column pass. The first axis is one column
pass of N1 rows over N2 K lanes, the plane's full width: the 1D batch's descriptor
(`il2d_col` at rn = N2 K), walker, race and dense scratch, unchanged. The last axis is N1
blocks of the 1D lane-major batch at (N2, K): block r is the contiguous N2 K complexes of
row r of the first axis, so the child is the 1D batch's own plan, in role on a private
store (its recipe rides on this cell's row as `rp_*`), run per block in place on the
plane after the first axis (`il/rank2/lm_batch2d_il.h`). The passes commute, so the
backward runs the same order with the backward kernels. The first axis takes the cell's
order class (natural, or the scrambled comb), the row child is natural, as the 2D tier's
row child is. Threaded, the first axis runs its raced column forms and the blocks go to
the workers, each on its own clone of the row child (a plan's scratch is not reentrant;
a clone is accepted only when its chain, verdicts and kernels match the primary), the
whole execute one threaded verdict (`cmt=`) raced with the column pass's arms. Both
placements: the column pass moves src to dst in its first stage, the rows run in place.
The row: the 2D cell at `q = K` on the `lay=lm` axis, the first axis's column tokens on
it. K = 1 is the ordinary 2D plan (the geometry does not apply).

The real 2D batch (`il/rank2/lm_batch2dr_il.h`) is the same shape with the axes in the
real order. r2c runs the last axis first: N1 blocks of the 1D real lane-major batch at
(N2, K), the child in role with its fused strips, out of the caller's reals into the
spectrum's row-blocks (in place: its (N2 + 2) K-double plane per block); then the first
axis, one complex column pass of N1 rows over (N2/2 + 1) K lanes, in place on the
spectrum, natural. c2r is the mirror: the backward column pass, then the blocks through
the 1D c2r batch. Out of place the caller's spectrum must survive, so the column pass
moves it into a scratch of the spectrum's size and the blocks read from there; in place
both passes run on the one padded plane, N1 blocks of (N2 + 2) K doubles. The row is the
real one at rank 2, `t=r2c n=N1xN2 q=K` on the `lay=lm` axis, direction-shared as the 1D
real row is. The named geometry only: at rank 2 the DEFAULT stays contiguous planes.

Rank 3, the first axis over N2 N3 K lanes and then N1 blocks of the rank-2 batch through
the same door, is the next piece.

## 9. What this retires

The two "lane-major lost to transform-contiguous at every measured cell" refusals
(c2c_oop_create_il.h, c2c_ip_create_il.h) are a verdict on the split-convert path of
2026-09, not on a native engine; they go with the door. The V02 refusal of the real
batch's DEFAULT geometry (vfft.c, 2026-10-14) goes when the real door serves. The
split bridge's lane-major crossing was cut with the layout separation; nothing of it
returns.
