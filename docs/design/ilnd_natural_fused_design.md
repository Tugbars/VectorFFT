# The fused natural form of the 3D interleaved tier — designed, built, raced, REFUTED

Design and verdict, 2026-09-15. Basis: `docs/design/3D_natural_il_design.md`
(the cycle form, shipped 2026-09-07; its §6 named this form as the first
lever), `docs/roadmap/fftnd_il_design.md` §2/2a, the long-N3 measurement of
2026-09-15 (`ztt_2d_design.md`: the natural class 1.28-1.34x over the
scrambled class at the pow2 long-N3 cells), and the owner's direction to
develop the arm so the 3D interleaved c2c tier is finished. The build and
the race are recorded in
`docs/research/sub2048_mkl_method/campaign_state/probes/IL3D/` (the patch,
the race logs, the results); the form is not in the library.

## The premise, and the form

The natural class runs the scrambled axis-0 pass with nothing fused, then a
second pass — the per-plane structure along the cycles of the axis-0 plane
permutation, one plane of buffer — so every finished plane lands at its
natural position (the cycle form). The record charged its 17-66% over the
scrambled class to "the band fusion it gives up". The fused form gave the
axis-0 pass its own cube: the scrambled banded walk through a scratch cube
Z (prefix `src -> Z`, per band the suffix in place on Z and the structure
`Z plane q -> dst plane natp[q]`; the backward its mirror; in place the same
walk), no cycles, the band and plane partitions with the redirection, the
form raced beside structure × width (`nf=`) and in the threaded race
(`cmtf=`).

## What the race said

Built and gated (bitwise the cycle form in both placements at 13 cells,
threaded bitwise the serial under both partitions with engagement), then
raced on a cool machine at every 3D cell: the fused form LOST to the cycle
form at every one-thread cell, by 3-27% (64³ flat/8: 582 vs 547 µs; 128³
flat/8: 7.47 vs 6.61 ms; 32×32×4096 flat/32: 19.1 vs 17.0 ms), and at 13 of
14 threaded cells (128³ band/flat 2.44 vs 1.18 ms; 32×32×4096 plane/flat
7.25 vs 4.73 ms); its one threaded win, 32×16×64 by 8%, sits inside that
cell's 145% spread.

Read: the permuting plane pass is one extra cube sweep however it is
arranged. The cycle form re-reads every plane cold; the fused form writes
the cube Z and still writes every destination plane cold, which costs more
than the recovered fusion returns (at 128³ the excess, 0.86 ms, is one cube
write). The natural class's residual over the scrambled class is structural
to a permuting pass with this tier's shape — the cycle form is the cheapest
known way to pay it.

## Ruling

Refuted and deleted the same day (the pool-sunset law: an arm that wins one
cell inside the noise is not kept). Do not rebuild this form. A form that
pays no extra sweep would need bands CLOSED under the axis-0 permutation
(a cycle-closed banding: the suffix stages and the plane pass over a set of
planes that maps onto itself), which is a different design, not a variant
of this one; nothing here builds it.

What stayed in the tree from this work:

- the natural axis-1 pass's scratch choice (`_ilnd_plane_t`): out of place
  one plane is dead — forward the source (a vacated cycle position),
  backward the destination — and the pass runs its pre-leaf stages there;
  `natscr` serves only the fixed points. Same arithmetic (the natural probe
  passes stay bitwise), one plane sweep fewer per plane;
- `ilnd_probe`'s 16×16×4096 cell.

## The 3D interleaved c2c tier, as it stands

Scrambled class: beats MKL CCE at every measured cell (11 of 11 standard,
3 of 3 long-N3). Natural class: the cycle form; wins or ties MKL's natural
output at the small and odd cells, trails at the large pow2 cubes and the
long-N3 cells by the structural sweep above (0.89-0.99x at one thread,
0.70x at T=8 at 32×32×4096). The rows are ZTURN-T by inheritance in both
classes. The tier has no open arm; what remains is the cycle-closed banding
question, a design of its own.
