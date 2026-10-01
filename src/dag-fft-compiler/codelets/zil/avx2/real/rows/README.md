# real/rows/ — the real rows kind `r2zr`: the row pass of a real plane

7 files, `radixN_z_r2zr_avx2.c` at the even radices 4, 6, 8, 10, 12, 16 and
32; forward only. One call is the real N-point DFT of every ROW of a
row-major plane: row k's samples at `zin[k*Ls + j]` (`Ls` = the row pitch in
doubles), its CCE bins 0..N/2 at `zout[2*(k*OLs + p)]` (`OLs` = the row
pitch in complex, at least N/2 + 1). `count` rows, at least two; out of
place.

A lane is a row. The arithmetic is the real leaf's (`cx_real.ml`: the
radix-2 recursion on real samples, real throughout, no fold and no
untangle), four rows per vector. The edges are the row-major ones:

- the load edge takes four rows' samples 4b..4b+3 as four vectors and
  transposes the block (two unpacks per row pair, one turn per sample), so a
  sample across the four rows is one vector; N = 4m + 2 takes its last two
  samples from an overlapping block at N - 4;
- the store edge interleaves (re, im) per row over the slots 0..N/2, the DC
  and the Nyquist slots as (x, 0): slots pair up into one 256-bit store per
  row, a lone last slot leaves as per-row halves.

Four rows per wide iteration, then two at VEX-128; a lone last row runs with
the row before it (the same values written again).

## Why it exists

In the row pass of a real plane the rows are a batch, and across a batch the
real symmetry is plain arithmetic: half the work of the complex transform,
on every lane. Measured on the i9-14900KF (2026-10-01), ns per row over 128
and 4096 rows: N = 16: 3.7 against 9.1 for the real mono called per row;
N = 32: 12.0-12.5 against 16.1-16.4 for the real pair. The transposes cost
1.6-2.1 ns a row inside L1 and nothing once the plane comes from L2. The
monolithic kernel stops at 32: at 64 its 64 live legs spill and the real
pair is the faster row engine (26 against 44-52).

Run by the 2D real tier's row pass (`src/core/il/rank2/`): a row engine of
the plan's own row race, banked on the 2D real row. Emitted by
`generator/lib/gen/real_il.ml` (`gen_radix N --cil-r2zr`) with the row
address forms of `lib/cx/cx_ir.ml` / `cx_render.ml`; rows in `corpus.ml`.
The kernel gate is `gauntlet/r2zr_gate.c` (a naive real DFT; 2 to 64 rows,
tight and padded pitches with guard values).

Regeneration: through `gen_set` to a temporary root (`--root`), never in
place; the folder a file lands in follows its kind (`Corpus.dir_of_file`),
and the law is byte identity against the shipped file. The map of the whole
tree, the two layouts and the rules are in [`../../README.md`](../../README.md).
