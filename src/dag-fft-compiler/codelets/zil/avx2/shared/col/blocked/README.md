# shared/col/blocked/ — the blocked column forms

12 files: `n1cb48`, `n1cb84`, `n1cb88` and `t2cb48`, `t2cb84`, `t2cb88` (+ `_bwd`),
radix 32 as 4x8 or 8x4 and radix 64 as 8x8, two passes through a small parked
buffer because the one-pass column kernel spills at those radices. The door
races the forms per cell and banks the winner as `forms=` on the 2D row
(`vfft_il2p_col_forms` in `src/core/oop/il2p.h` is the pool). The `b416`
form (radix 64 as 4x16) was deleted on 2026-09-24: it never won a banked cell.

Regeneration: through `gen_set` to a temporary root (`--root`), never in place;
the folder a file lands in follows its kind (`Corpus.dir_of_file`), and the law
is byte identity against the shipped file. The map of the whole tree, the two
layouts and the rules are in [`../../../../README.md`](../../../../README.md).
