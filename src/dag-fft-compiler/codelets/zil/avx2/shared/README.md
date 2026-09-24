# shared/ — the `n1` kind: a whole small transform in registers

62 files, `radixN_z_n1_avx2.c` and `_n1_bwd_`: no scratch buffer, no twiddle
table. The transform of R points is done entirely in registers, natural in,
natural out.

| direction | who runs it | wisdom |
|---|---|---|
| forward | the MONO route: a small N served by one kernel call | `il_route=mono` on the K=1 row |
| backward | the leaf of the two-pass pair and of chain3 (the R2 stage of `t2t`) | `il_route=2p`, `il_route=chain3` |

The forward kernels at radix 4, 8, 16, 32 and 64 were deleted on 2026-08-23
(no reachability); MONO has used the odd and composite radices since 2026-09-04.
Nothing here is unused.

Regeneration: through `gen_set` to a temporary root (`--root`), never in place;
the folder a file lands in follows its kind (`Corpus.dir_of_file`), and the law
is byte identity against the shipped file. The map of the whole tree, the two
layouts and the rules are in [`../../README.md`](../../README.md).
