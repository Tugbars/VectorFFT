# shared/col/ — the column stage of the N-D tiers

110 files, `n1c` and `t2c` (+ `_bwd`): one radix-R stage over `count` lanes,
a lane being a column of the plane at pitch `Gs`. Twiddles are hoisted out of
the column loop (`t2c` takes one (R-1)-record set for the whole call). The
kernels allow input = output, so a stage runs in place; that is also how the
in-place 1D MONO path uses `n1c`, and how the 2D row pass batches through the
column-stride `n1ccs` twin in [`../../rows/`](../../rows/).

| who runs it | what for | wisdom |
|---|---|---|
| `transforms/fft2d/il2d_cols.h` | the 2D column chain (and the skewed pass `csk`) | `chain=` on the 2D row; per-stage forms `forms=` |
| `transforms/fftnd/fftnd_il.h` | axis 0 and axis 1 of the rank-3 tier | `chain=`, `chain1=` on the 3D row |
| `oop/c2c_ip_create.h` | the in-place 1D MONO cell | `il_route=mono` |

`blocked/` holds the radix-32 and radix-64 forms raced per cell (`b48`, `b84`,
`b88`); the `b416` form was deleted on 2026-09-24 (it never won a cell).

Regeneration: through `gen_set` to a temporary root (`--root`), never in place;
the folder a file lands in follows its kind (`Corpus.dir_of_file`), and the law
is byte identity against the shipped file. The map of the whole tree, the two
layouts and the rules are in [`../../../README.md`](../../../README.md).
