# real/mono/ — the real mono kind `rn1`: the whole small real transform in one kernel

66 files, `radixN_z_rn1_avx2.c` (r2c) and `radixN_z_rn1_bwd_avx2.c` (c2r) at
the `n1` radices N = 3..64 (3-17, 19, 21-23, 25-27, 29, 31, 32, 37, 41, 43,
47, 64) and at the primes 53, 59, 61 (no `n1` kernel there: the mono is the
one engine between the chain's largest radix and the real Bluestein's
floor). The `n1` body on real input: the forward loads N reals (a lane is
`(x, 0)`), computes the N-point DFT and stores bins 0..N/2 only (the CCE
half); the backward loads bins 0..N/2 and forms bin N-l as the conjugate of
bin l (a sign flip on the imaginary lane, no load), then stores N real lanes.
Count = 1 runs the VEX-128 tail: the K=1 solo, no child, no fold, no table.
Count >= 2 packs two rows per vector (element l of row k at `zin[l*Ls + k]`).

Run by `src/core/il/real/zrm.h` under `eng=zrm`: the real door races it at
N <= 64 against zr2c (even N) and the odd-real routes (odd N) and banks the
winner in the real shard. Emitted by `generator/lib/gen/c2c_il.ml`
(`gen_radix N --cil-rn1 [--cil-bwd]`) with the real address forms of
`lib/cx/cx_ir.ml` / `cx_render.ml`; rows in `corpus.ml`. The kernel gate is
`gauntlet/rn1_gate.c` (a naive real DFT, count 1 and 2, both directions).

Regeneration: through `gen_set` to a temporary root (`--root`), never in
place; the folder a file lands in follows its kind (`Corpus.dir_of_file`),
and the law is byte identity against the shipped file. The map of the whole
tree, the two layouts and the rules are in [`../../README.md`](../../README.md).
