# Radix 32 and 64 in the interleaved c2c pipeline — a sunset to rule on

Note for the owner, 2026-09-16. Not a plan; the facts behind a ruling.

## What the shipped verdicts use (store, 2026-09-16)

| tier | cells | radices / sides the banked verdicts name |
| --- | --- | --- |
| ZTURN-T, pow2 16..262144 and the odd band | 10 pow2 rows | 4 and 8 only (3 and 5 as the odd mids) |
| Bailey pairs, below 2048 | 14 rows | sides 4..32; 64 only at 768 (64x12) and on 2048/4096 rows that ZTURN-T has since taken |
| the four-step, 2^19..2^22 | 4 rows | row plans of 256..4096: the pairs' 16/32 and ZTURN-T's 4/8 |
| the four-step's children, the 2D column chains | 164 stage uses in the 2D store | 4: 61, 8: 84, 16: 6, 32: 6, 64: 5 |

The pipeline's working set is 4 and 8. Radix 16 and 32 appear in a few
cells, 64 in fewer.

## The two holdouts

- Radix 64 has no 1D user left: no ZTURN-T chain, one odd pair (768 =
  64x12), and superseded 2048/4096 pair rows. Its users are 2D column
  chains in a handful of cells: 8.64.8 at 4096x256 (the 1M cell's child),
  64.x at the tall planes (4096x16, 8192x64, 16384x64, 32768x64), 64.16 at
  1024x1024, and the super-band's 8.64.8.
- Radix 32 carries the pair 32x32 at 1024 (16x32 at 512) and the column
  chain 8.8.32 at 2048x2048 (the 4M cell's child), plus 32.x at a few tall
  2D planes.

## What retiring them would take

1. Drop 32 and 64 from the pair enumerator and from the 2D column chain
   pool ({64, 32, 16, 8, 4} today); the odd radices stay.
2. Re-race the affected cells: 512 and 1024 (ZTURN-T already beats the pair
   at 1024 in the record, so they may simply move), the 2D cells above,
   and through them the four-step's 1M, 2M and 4M splits.
3. Keep a family only if a re-raced cell asks for it back; otherwise the
   codelets, their registry rows and their generator recipes go, whole
   (the clean-library law).

Consumers to check first: the 2D and 3D tiers' own cells (the tall planes
are theirs), the real tier's column chains, and the odd band's mids.

About half a day after the natural leaf lands. Not started.
