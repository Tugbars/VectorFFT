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

At 32 and 64 the body is BLOCKED (2026-10-06): `radix32_z_rn1b48` (4x8) and
`radix64_z_rn1b444` (4x4x4, three passes), both directions -- the c2c blocked
construction (`c2c_il.ml` `emit_blocked`) on the mono's own edges: pass 1 the
m real sub-DFTs parked in a function-scope `S[]`, pass 2 the m-point DFTs
keeping the CCE half (the scheduler sees the kept sinks only), the remainder
arm (count = 1, zrm's call) the blocked passes at VEX-128. The kernel race
(count 1 and 2, both directions): 7-14% over the monolithic bodies, spills
443 -> 226 at 32 and 1276 -> 761 at 64; the 8x8 twin at 64 tied at count 1
and lost at count 2. The monolithic 32/64 bodies are deleted (the law:
r32/r64 never monolithic).

Run by `src/core/il/real/zrm.h` under `eng=zrm`: the real door races it at
N <= 64 against zr2c (even N) and the odd-real routes (odd N) and banks the
winner in the real shard. Emitted by `generator/lib/gen/c2c_il.ml`
(`gen_radix N --cil-rn1 [--cil-bwd]`; blocked: `--cil-blocked --cil-split 4.8
--cil-form-tag` at 32, `--cil-blocked --cil-split3 4.4.4 --cil-form-tag` at 64)
with the real address forms of
`lib/cx/cx_ir.ml` / `cx_render.ml`; rows in `corpus.ml`. The kernel gate is
`gauntlet/rn1_gate.c` (a naive real DFT, count 1 and 2, both directions).

Regeneration: through `gen_set` to a temporary root (`--root`), never in
place; the folder a file lands in follows its kind (`Corpus.dir_of_file`),
and the law is byte identity against the shipped file. The map of the whole
tree, the two layouts and the rules are in [`../../README.md`](../../README.md).
