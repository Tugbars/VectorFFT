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

## 3. Wisdom

One row per cell in `wisdom2_3d.txt`: `t=c2c n=N1xN2xN3 q=1 ord=scr place=oop lay=il`.

| tokens | owner | meaning |
|---|---|---|
| `chain= blu= forms=` | axis 0 | the column pass's chain, N-arm and forms verdicts, spelled exactly as the 2D row spells them |
| `wl= tf=` | axis 0, the joint race | the banded walk's width (0 = unbanded) and its fusion flag |
| `chain1= blu1= forms1=` | axis 1 (flat arm) | the same verdicts with the axis as suffix |
| `s=` | the joint race | 1 = child, 2 = flat |
| `cmt= cmtt=` | axis 0 | not raced yet (phase 4) |

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
| 3 | NATURAL order (its own cell) and in place | next |
| 4 | MT: plane-parallel child clones vs an axis-0 column/band MT (`cmt`), raced | after 3 |
| 5 | real 3D (r2c/c2r) | after 4 |
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
- `api_matrix_gate`: 3D c2c OOP IL 16³ DEFAULT and SCRAMBLED and 9×15×27
  are served; NATURAL and howmany 2 are refused.
- The plan fingerprint carries `ilnd=[arm ax0=nst/blu/wl ax1=nst/blu]` and
  recurses into the child and row plans.

## 7. Measurement

`bench_1d_vs_mkl --3dil` (env `VFFT_3DIL_CELLS`, `VFFT_3DIL_ROUNDS`): arms
O-NATIVE (this tier), M-inter (DFTI 3D CCE NOT_INPLACE, the yardstick),
M-split (DFTI REAL_REAL NOT_INPLACE, shows CCE is MKL's best), ctl memcpy —
all out of place, median + spread, a delta below the ctl spread is not a
result. The split rank-N tier is not an arm and not a comparison (owner,
2026-09-06: "split is not our concern. IL is what matters"). Numbers live
in `docs/performance/v1_0_results.md` (the 3D section), never here.

## 8. File map

| file | role |
|---|---|
| `transforms/fftnd/fftnd_il.h` | `vfft_ilnd_t`, execute, destroy, the arm builders, the structure race, `_vfft_create_fftnd_il` |
| `transforms/fftnd/fftnd_create.h` | rank-3/4 create dispatch: INTERLEAVED C2C rank 3 → this tier |
| `transforms/fft2d/il2d_col.h`, `il2d_tier.h` | the column-axis descriptor, `_il2d_col_build`, `_il2d_col_exec`, `_il2d_col_free` (lent to this tier; no rank-3 code lives there) |
| `wisdom2/wisdom2_2d_reader.h` | `vw2_ilcol_key_t` (rank, dims, ord, axis), the axis-suffixed chain/forms bank and lookup, `vw2_ilnd_arm_lookup/bank` |
| `vfft_execute.h` | dispatch (`h->ilnd`, inside the rank≥2 INTERLEAVED branch) and destroy |
| `vfft_internal.h` | `struct vfft_ilnd_s *ilnd` on the plan |
