# The strip form of the 3D natural class — axis 0 in cache-resident strips, no move pass

Design, 2026-09-15. Basis: `docs/design/3D_natural_il_design.md` (the cycle
form), `ilnd_natural_fused_design.md` (the scratch-cube form, refuted the
same day: a permuting pass is one extra cube sweep however arranged), the
one-session 3D table of 2026-09-15 (`v1_0_results.md`: natural 18.8 ms vs
scrambled 13.6 ms vs MKL 16.7 ms at 32×32×4096), the survey of the 2D tier's
natural strip arm (`il2d_cols.h` `_il2d_col_pass_nat_range`, the stage
runner's separate scratch pitch, the `t2c`/`n1c` argument contract), and the
owner's direction (2026-09-15): "this totally makes sense, let's do it".

## The idea in one paragraph

The natural class's whole deficit is a cube-sized move pass: the band walk
leaves the planes digit-reversed, and putting them back is one cold sweep of
the cube. MKL never moves a plane because it transforms each axis in STRIPS
— a few columns wide, the full axis long, small enough to live in L1/L2 —
and resolves the digit reversal inside the strip. This form does the same
for axis 0 of the natural class: a strip walk that writes natural order in
place with a strip-sized scratch, then the per-plane work in place. Two
cube sweeps, the scrambled class's count, instead of three or four.

## The strip pass

A strip is columns `[k, k + w)` of the virtual N1 × (N2·N3) plane, every
plane included: N1 rows of `w` complexes, `w · 16` bytes per row, `N1 · w`
complexes in all (128 × 16 = 32 KB). Its scratch is a dense N1 × w block
(pitch `w`), one per worker.

```
forward, strip [k, k+w):
  stage 0        src (pitch rn) -> scr (pitch w), one kernel call PER DIGIT d
                 (the stage kernels advance input and output by the same pitch
                 per digit, so the change of pitch is a caller loop: input at
                 src + 2*(d*rn + k), output at scr + 2*d*w, OGs = 1, the
                 table advanced (R-1)*8 doubles per digit -- the kernels stay)
  stages 1..nst-2  in place on scr (Ls = D*w, Gs = w, OGs = D, count = w)
  leaf           scr block b (Ls = w) -> dst rows perm[b*Rl] + r*(N1/Rl), columns k..k+w
                 (OLs = (N1/Rl)*rn, count = w): NATURAL order, the strip's own rows
backward, the mirror: leaf gathers dst-rows -> scr (Ls = (N1/Rl)*rn, OLs = w), mids in
  place on scr, stage 0 scr -> dst per digit (input scr + 2*d*w, output dst + 2*(d*rn + k)).
```

In place by construction: every row of the strip is read into the scratch
before any row of the strip is written, and strips are disjoint. Same
kernels, same tables, same values as the plane-sized natural pass: only the
scratch addresses differ, so the output is bitwise the cycle form's (the
cycle form runs the same stages on the same columns and the same per-plane
structure). Odd `count` is legal (the kernels carry their tails), so the
last strip may be narrower; the width is otherwise free.

## The form

`nf=2` (strip), beside `nf=1` (cycle):

- forward: the strip pass over all columns, then the per-plane structure
  IN PLACE on every plane (the scrambled class's plane call);
- backward: the per-plane structure in place, then the strip pass reversed;
- both placements from one code path (out of place: the strips `src -> dst`,
  the planes in place on dst);
- threading: strips are the partition (disjoint column ranges, one strip
  scratch per worker), then plane ranges — the PLANE arm's two phases with
  the natural strip pass in the first; the BAND arm does not apply (there
  is no band); the cycle binding (`cycw`, `bufw`) is the cycle form's only;
- memory: T strip scratches of `N1 · w` complexes; no cube, no plane buffer.

The strip width `w` is a raced parameter: candidates {8, 16, 32, 64}
columns, filtered so the strip scratch stays under the L2 budget
(`N1 · w · 16 <= L2 bytes`; at N1 = 256 that admits 8..64, at N1 = 16 all).
Banked as `nsw=` on the rank-3 `ord=nat` row beside `nf=`.

## The race, and what wisdom banks

The natural cell's one-thread race: arms `(structure, wl, cycle)` as today
plus `(structure, w, strip)` for each admitted width (the strip form has no
axis-0 band, so `wl` does not apply to it). Verdict banks `s=`, `wl=` and
`tf=` (the cycle form's), `nf=` and `nsw=`. The threaded race at the plan's
T: serial against (partition, structure) as today for the cycle form, and
`plane/structure/strip` for the strip form; `cmt= cmtt= cmts=` and `cmtf=`
banked; serving form = the threaded one when the plan threads, else the
one-thread one (the structure's rule). `VFFT_ILND_NF=1|2` and
`VFFT_ILND_SW=<w>` pin for a probe, never bank. DEFAULT keeps meaning
scrambled; the natural cell is its own row and is never compared with the
scrambled cell.

## Gates

- `ilnd_probe`, natural passes: the strip form pinned (`NF=2 SW=16`) out of
  place and in place at T=1 is BITWISE the cycle form under the same
  structure pin; at T=8 (plane partition) bitwise the plan's own serial
  with the engagement counter moving; the raced natural passes (8-11) keep
  holding whichever form wins; a warm store replays with no race.
- `api_matrix_gate` unchanged; the scrambled class untouched (its probe
  passes and bench numbers are the regression reference).

## Measurement

`bench_1d_vs_mkl --3dil` with `VFFT_3DIL_ORDER=nat`, the 14 cells, one
thread paced and T=8 unpaced with engagement, scratch store, the create log
kept (`VFFT_IL2D_LOG=1`: every arm's ns, the form and width chosen);
against the same session's scrambled column and the 2026-09-15 natural
table. The expectation to test: the large pow2 cells move from 0.89-0.99x
of MKL to the scrambled class's neighbourhood (about 13 ms against MKL's
16.7 ms at 32×32×4096); the small cells, cache-resident whole, do not move.

## Measured (2026-09-15, two quiet runs, the race's own arm times)

The strip form won the one-thread race at 128³ (flat/strip512 6.24 ms vs
the cycle form's 6.79 ms) and 32×32×4096 (flat/strip256 15.0 vs 16.6 ms;
the bench 0.89x -> 1.17x of MKL), tied at 36×20×28, and lost where the cube
fits L3 (64³, 256×64×16, 64×128×32: the strided strip reads cost more than
the cheap move pass; the race keeps the cycle form). At T=8 it won 10 of 14
cells, on both runs: 32×32×4096 3.07 vs 5.06 ms, 81×27×27 31 vs 49 µs,
36×20×28 10.0 vs 16.5 µs, 45³ 38 vs 51 µs, 256×64×16 116 vs 170 µs, 32×16×64
18 vs 30 µs, 16³ 4.5 vs 5.4 µs — the strips give every worker independent
in-scratch work with no cold scattered plane writes. Against MKL CCE at
T=8 the natural class went from 0.67-0.74x at 16³, 32³, 81×27×27,
32×32×4096 to parity or better at every cell. Width: the arm times fall
monotonically to 256-512 columns at the long cells (one page visit per
plane per strip), so the pool runs to 1024 under the L2 budget. Full table:
`v1_0_results.md`.

## Ruling

Shipped: the strip form is a raced form of the natural class beside the
cycle form (`nf=`, `nsw=`, `cmtf=` on the rank-3 `ord=nat` row), never a
default. Open item, not built here: the axis-0 chain is raced as a bare
column pass and can pick a single-stage axis (natural already, no strip
form to race) that serves slower than a two-stage chain with strips; a
joint chain × form race for the natural cell is its own design.

## Checklist

- [x] 1. This design.
- [x] 2. `_il2d_col_pass_nat_strip` in `il2d_cols.h`: the strip pass with a
      strip-pitched scratch, both directions (the per-digit stage-0 loop, the
      in-place mids, the leaf scatter/gather). No kernel or table change.
- [x] 3. The strip form in `fftnd_il.h`: the form (`nf`), the strip width and
      the per-worker strip scratches, the serial execute (strips then
      planes; backward the mirror), both placements.
- [x] 4. Threading: the strip phase (one scratch per worker) then the plane
      phase; the cycle binding only for the cycle form.
- [x] 5. The race: the form × width axis in the one-thread race (widths
      8..256 under the L2 budget), the strip form's plane arm in the
      threaded race; `nf= nsw= cmtf=` banked and replayed; `VFFT_ILND_NF` /
      `VFFT_ILND_SW` pins.
- [x] 6. `ilnd_probe` passes 12-15 (pinned NF=2 SW=16): T=1 both placements
      bitwise the cycle form; T=8 bitwise own serial, engaged. ALL OK at 13
      cells, 2026-09-15.
- [x] 7. Measure: the 14 natural cells, T=1 and T=8, the create log (two
      quiet runs; a third was discarded, the machine was loaded).
- [x] 8. Rule from the numbers: shipped as a raced form; wins T=8 at 10/14
      cells and T=1 at the large pow2 cells; the cycle form stays where the
      cube fits L3.
- [x] 9. Records: `v1_0_results.md`, `3D_natural_il_design.md` §6, this doc's
      measured section, `vfft.h`'s 3D paragraph, memory.
