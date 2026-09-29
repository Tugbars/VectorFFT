# real/pair/ — the real pair's kinds

48 files at the even pair radices 4, 6, 8, 10, 12, 16, 32 and 64, fwd + bwd:
`t2h` (form A: the Hermitian top stage over the stock `n1t` leaf's packed
view), `r2z` (form B: the real leaf) and `t2m` (form B: its top stage). An
even-N r2c or c2r as two kernels and no fold pass. Run by
`src/core/il/real/zrp.h` when the real door (`il/real/zrp_build.h`) banks
`eng=zrp pair=R1.R2 leaf=n1t|r2z`; the pair and the form are plan input,
raced against zr2c, ZTT-r and the real mono. Emitted by
`generator/lib/gen/c2c_il.ml` with the real address forms of
`lib/cx/cx_real.ml`; rows in `corpus.ml`.

Raced 2026-09-29: form A loses everywhere; form B wins only where the c2c
child of zr2c is MONO (N = 128), so few cells bank it. The pair sat in
`real/` until 2026-09-30, when the real family got one folder per engine.

Regeneration: through `gen_set` to a temporary root (`--root`), never in
place; the folder a file lands in follows its kind (`Corpus.dir_of_file`),
and the law is byte identity against the shipped file. The map of the whole
tree, the two layouts and the rules are in [`../../README.md`](../../README.md).
