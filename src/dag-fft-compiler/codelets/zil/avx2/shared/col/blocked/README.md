# shared/col/blocked/ — the blocked column forms

18 files: `t2cb44` (+ `_bwd`), the radix-16 twiddle stage as 4x4 -- its ONE body
since 2026-10-06 (the one-pass stage parked 40 vectors per iteration under gcc;
the 4x4 form raced 2-5% ahead in the band role and 6-7% in the wide sweep,
dominating 8x2 and 2x8, so the monolithic file is gone; the radix-16 leaf `n1c`
is clean and stays in `../`); `n1cb48`, `n1cb84`, `n1cb88` and `t2cb48`, `t2cb84`,
`t2cb88` (+ `_bwd`),
radix 32 as 4x8 or 8x4 and radix 64 as 8x8, two passes through a small parked
buffer because the one-pass column kernel spills at those radices; and
`n1cb816`, `n1cb448` (+ `_bwd`), radix 128 as 8x16 (two passes) and 4x4x8
(three), the 2D real tier's ONE-KERNEL COLUMN LEAVES: the whole column pass
of a 128-row plane in one call, natural order by construction, raced against
the chain's pass by the r2c column plan (`cx=`; the forward kernels) and by
the c2r column plan (`cx_c2r=`; the backward twins, 2026-10-05) in
`src/core/il/rank2/il2d_real_plan.h` (`vfft_il2p_col_leaf_fn` /
`vfft_il2p_col_leaf_bwd_fn` in `src/core/il/rank1/il2p.h`). The door races
the radix 32 and 64 forms per cell and banks the winner as `forms=` on the 2D
row (`vfft_il2p_col_forms` in `src/core/il/rank1/il2p.h` is the pool). The
`b416` form (radix 64 as 4x16) was deleted on 2026-09-24: it never won a
banked cell.

Regeneration: through `gen_set` to a temporary root (`--root`), never in place;
the folder a file lands in follows its kind (`Corpus.dir_of_file`), and the law
is byte identity against the shipped file. The map of the whole tree, the two
layouts and the rules are in [`../../../../README.md`](../../../../README.md).
