# rows/ — the 2D row pass's kernels

The kernels of the row pass of the 2D interleaved tier
(`docs/design/il2d_c2c_strategy.md`), also serving the rank-3 tier's planes
through the 2D child. Since 2026-11-08 the batched row kernels live in one
subfolder per construction; the row-loop twins of the two-pass route stay here.

| folder / kind | what it is | wisdom |
|---|---|---|
| `row/` — `row`, `rowb44`, `rowb48` (+ `_bwd`) | the batched row kernel in MKL's row form: ONE row per iteration, a vector holds neighbouring points of the row; stage 1 lane-parallel across the registers, one register transpose per block, lane-varying twiddle constants, stage 2 across the block; natural order in and out, no tail. `rowb<m1><p1>` runs stage 1 as m1 sub-DFTs of p1 through a scratch array (the 256-bit register file is too small for 16 or 32 live vectors). Fits lengths that are a multiple of (complex per vector)^2. | `ro=2` on the 2D row |
| `rowswap/` — `rowswap` (+ `_bwd`) | the batched row kernel with lanes = rows (as `n1ccs`), but each row's vectors are loaded whole and the halves of two rows' vectors are swapped in registers; the leftover point of an odd row goes the `n1ccs` way. | `ro=2` |
| `n1ccs/` — `n1ccs` (+ `_bwd`) | the lane-load construction: `n1c` with lane k = row k at pitch `Gs`, every lane loaded and stored on its own (the loadu2 / storeu2 pairs). Present at every length 2..47 and 64: it is the row kernel where it won, and it alone honours the turn route's turned stores (`OLs` = the scratch pitch, `OGs` = 1). | `ro=2`; `turn=1` |
| `n1tr`, `t2r`, `t2tr_bwd`, `n1r_bwd` and their `tan` forms (here) | the two-pass pair's kinds with a row loop inside the kernel (`count = rows x Ls`, `Ls` lanes per row, `Gs` the row pitch) | `ro=3`, the tile `rbk=` |

**Which kernel serves a length** is the generated `VFFT_IL_ROW_CHOICE` in
`generator/generated/il_registry_<isa>.h` (`vfft_il_row_fn`): the one
`row*` / `rowswap` pair the corpus holds at that radix, else `n1ccs`. The
corpus holds only the winner of the pinned race of 2026-11-08 on the 14900KF
(AVX2, ns per row, min of 21 rounds): `row` at 4, 12, 16; `rowb44` at 32 and
`rowb48` at 64 (2.8x and 1.6x over `n1ccs`); `rowswap` at 3, 9, 11, 13, 15,
22; `n1ccs` at 2, 5, 6, 7, 8, 10, 14, 26 and the odd lengths 17..47. At
AVX-512 nothing was raced: `row` at 16, 32, 64 and `rowb48` at 128 stand on
instruction counts (within 5 % of MKL's row kernels) and SDE correctness;
`n1ccs` serves the other lengths, and `rowswap/` has no AVX-512 folder.

Emitted with `--cil-row [--cil-split m1.p1]`, `--cil-rowswap`, `--cil-n1ccs`
and `--cil-rowloop` (`generator/lib/gen/c2c_il.ml`: `emit_row` and `emit`).

Regeneration: through `gen_set` to a temporary root (`--root`), never in place;
the folder a file lands in follows its kind (`Corpus.dir_of_file`), and the law
is byte identity against the shipped file. The map of the whole tree, the two
layouts and the rules are in [`../README.md`](../README.md).
