# rows/ — the 2D row pass's kernels

82 files, two kinds built on 2026-09-23 for the row pass of the 2D interleaved
tier (`docs/design/il2d_c2c_strategy.md`), both serving the rank-3 tier's planes
through the 2D child.

| kind | files | what it is | wisdom |
|---|---|---|---|
| `n1ccs` (+ `_bwd`) | 62 | the column-stride mono kernel: `n1c` with lane k = row k at pitch `Gs`, loads and stores in pairs; one call transforms every row of a plane whose length is a mono radix (2..47, 64). The turn route stores it transposed. | `ro=2` on the 2D row; `turn=1` |
| `n1tr`, `t2r`, `t2tr_bwd`, `n1r_bwd` and their `tan` forms | 20 | the two-pass pair's kinds with a row loop inside the kernel (`count = rows x Ls`, `Ls` lanes per row, `Gs` the row pitch): the rows of a plane whose length is a pair cell, in chunks that fit a raced tile | `ro=3`, the tile `rbk=` |

Emitted with `--cil-n1ccs` and `--cil-rowloop` (`generator/lib/gen/c2c_il.ml`).
The tangent row-loop forms are raced kernel variants; none has banked in the
shipped 2D shard as of 2026-09-24.

Regeneration: through `gen_set` to a temporary root (`--root`), never in place;
the folder a file lands in follows its kind (`Corpus.dir_of_file`), and the law
is byte identity against the shipped file. The map of the whole tree, the two
layouts and the rules are in [`../README.md`](../README.md).
