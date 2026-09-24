# pair2p/ — the two-pass Bailey pair

161 files. A length N = R1 x R2 runs as two passes: `n1t` (the R1 leaf with a
turned store), then `t2` (the R2 stage with twiddles); backward `t2t` then the
`n1_bwd` leaf from [`../shared/`](../shared/). The transpose between the
passes is folded into the store addresses (R lane swaps), so there is no extra
memory pass. Wins while input, scratch and output fit in L1.

The `_ct` kinds (`n1_ct`, `n1t_ct`, `t2_ct`, `t2t_ct`, 26 files) do the odd
composite radices 15, 21, 25 and 27 as a small Cooley-Tukey inside the kernel,
so the kernel stops spilling registers. Their banked wins were raced against the
odd kernels of before 2026-09-23; a re-race may flip some of them.

| who runs it | wisdom |
|---|---|
| `src/core/oop/il2p.h`, the K=1 door | `il_route=2p`; `il_kv` packs the per-pass kernel variant: the default, a blocked form ([`blocked/`](blocked/)), the tangent interior 3 and its edge variant 4 ([`tangent/`](tangent/)) |

Of the 2,250 banked pair slots on 2026-09-17: 80% the one-pass default, 10%
`_ct`, 9% tangent, 1% blocked. Nothing here is unused; the `log3` twiddle
policy's 52 kernels were deleted on 2026-09-24 (no consumer, loses in
interleaved: registers, not loads, are the limit).

Regeneration: through `gen_set` to a temporary root (`--root`), never in place;
the folder a file lands in follows its kind (`Corpus.dir_of_file`), and the law
is byte identity against the shipped file. The map of the whole tree, the two
layouts and the rules are in [`../README.md`](../README.md).
