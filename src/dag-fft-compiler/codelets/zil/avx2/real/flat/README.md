# real/flat/ — the real flat DIT's leaf `r1c`

38 files, `radixR_z_r1c_avx2.c` (fwd) and `radixR_z_r1c_bwd_avx2.c` (bwd) at
the odd radices the flat DIT's stages have: 3, 5, 7, 9, 11, 13, 15, 17, 19,
21, 23, 25, 27, 29, 31, 37, 41, 43, 47.

The flat DIT's first stage on REAL input. R real legs at stride `Ls`
(leg l of column k at `zin[l*Ls + k]`), `count` contiguous columns, four real
columns per vector. The output is the real R-point DFT of each column, kept
as digit runs:

- digit 0, real, at `zout[k]`;
- digit p in 1..(R-1)/2, complex, at `zout[2*p*OLs + 2*k]` — block p of the
  c2c flat plane at pitch `OLs`, so the c2c stages run on the digit blocks as
  they stand. The second half of block 0 is never written.

The backward kind is the inverse: the digit runs in, R real legs out,
unnormalized (R times the input). Any `count`: a two-column step at VEX-128,
then the lone last column (an odd N has odd runs). Out of place only.

The arithmetic is the conjugate-pair form of the c2c odd kernels on real
input (`generator/lib/cx/cx_real.ml`): pair sums and differences, the cosine
row over the sums for a digit's real part, the sine row over the differences
for its imaginary part. Every operation is real. The kernels are monolithic:
none spills through radix 9; from radix 11 the compiler spills, lightly to
15 and heavily from 21.

The engine's other kind, the Hermitian last stage `t2csgh` / `t2csght`,
sits in [`herm/`](herm/README.md).

Run by `src/core/il/real/zrf.h` under `eng=zrf`: the odd real race
(`bridge/real_bridge.h`) sweeps the chains and banks the winner in the real
shard. Emitted by `generator/lib/gen/real_il.ml` (`gen_radix R --cil-r1c
[--cil-bwd]`) with the digit address forms of `lib/cx/cx_ir.ml` /
`cx_render.ml`; rows in `corpus.ml`. The kernel gate is `gauntlet/r1c_gate.c`
(a naive digit transform, counts 1 to 15, both directions, the unused half
of block 0 checked).

Regeneration: through `gen_set` to a temporary root (`--root`), never in
place; the folder a file lands in follows its kind (`Corpus.dir_of_file`),
and the law is byte identity against the shipped file. The map of the whole
tree, the two layouts and the rules are in [`../../README.md`](../../README.md).
