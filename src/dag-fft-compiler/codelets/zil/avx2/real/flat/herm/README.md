# real/flat/herm/ — the real flat DIT's last stage: `t2csgh` and `t2csght`

38 files at the odd radices 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27,
29, 31, 37, 41, 43, 47: `radixR_z_t2csgh_avx2.c` (fwd, r2c) and
`radixR_z_t2csght_bwd_avx2.c` (bwd, c2r).

## What they are

The last stage of the c2c flat DIT is `t2csgn` (`../../../flat/`): the
column-stride tail with the in-kernel group loop, one call for the whole
stage, its blocks written in the plane's own block order (the scrambled
class). On the real flat DIT (`src/core/il/real/zrf.h`) that stage's
outputs are the transform's bins, and only half of them are wanted: the
r2c contract is bins 0..N/2, with a bin above N/2 standing for the
conjugate of its mirror N - f.

A block of the last stage holds one bin per leg: leg l is the bin
base + l·(N/R), base < N/R. So the legs sort themselves: legs below R/2 are
bins below N/2 and go to their places; legs above R/2 are bins above N/2
and go, conjugated, to N minus their bin; the middle leg is on one side or
the other by its base. `t2csgh` is `t2csgn` with exactly that store edge:

- the group loop reads the plane by the relative group index, as `t2csgn`
  does, and writes through an ABSOLUTE base table: the natural bases of the
  blocks, scaled to the caller's bins (a deeper level's bin f is the
  transform's bin f·N/Nj);
- the body knows its column's place from the two pointers the wrapper
  passes in the unused slots: the caller's spectrum (`zin_unused`) and its
  mirror base `out + 2N` (`zout_unused`); `OLs` / `OGs` are the leg stride
  and column pitch there;
- legs above R/2 store `xor(v, {0, -0})` at `mirror - 2·(place + l·OLs)`;
  the middle leg tests `2·place < OLs` per column (two columns per vector,
  each its own side).

`t2csght` is the transposed backward twin (`t2csgnt` with the same rule on
its LOAD edge): it reads the half spectrum through the scaled base table and
the mirror, conjugating the mirrored legs, and writes the plane in block
order (a last stage: leg stride 1, column pitch R).

What this buys: the order sweep that moved the plane to the half spectrum
was one more pass over the data, 12% of the transform at N ≤ 2048
(2026-09-30); the last stage now writes its bins where they belong and the
first backward stage reads them from there.

## Where they run

`src/core/il/real/zrf.h`: every level's last forward record (`cf[ns-1]`)
and first backward record (`ct[0]`) carry these kernels with the level's
scaled base table; the mirror base is passed at the call
(`_zrf_call_herm`). Tiled levels run them per tile, the threaded forms
(`zrf_mt.h`) per worker's tile range. The engine refuses a chain whose last
stage is not the group-loop form.

## Emission

`generator/lib/gen/c2c_il.ml` (`gen_radix R --cil-t2csgh` and
`--cil-t2csght --cil-bwd`), the edges in `lib/cx/cx_render.ml`
(`Cx_render.herm`, `herm_hl`); rows in `corpus.ml`. Regeneration: through
`gen_set` to a temporary root (`--root`), never in place; the folder a file
lands in follows its kind (`Corpus.dir_of_file`), and the law is byte
identity against the shipped file. The map of the whole tree, the two
layouts and the rules are in [`../../../README.md`](../../../README.md).
