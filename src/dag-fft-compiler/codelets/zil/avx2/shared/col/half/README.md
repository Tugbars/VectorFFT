# shared/col/half/ — the forward column leaf's half-store twins

32 files: `n1ch` at every `n1c` radix but 32 and 64 (where the monolithic
leaf is never served), and `n1cb48h`, `n1cb84h`, `n1cb88h`; forward only. Each
is its plain or blocked forward twin with one difference: the plane stores of
the wide path leave as two 128-bit halves (`vmovupd xmm` + `vextractf128` to
memory) instead of one 256-bit store. The loads, the body and the VEX-128
odd-count tail are the twin's, byte for byte.

## Why they exist

The 2D real r2c column pass runs in place at the CCE row pitch
hp1 = N2/2 + 1, which is odd for every even N2. A column pair is 32 bytes; at
an odd pitch every other row starts 16 bytes off a 32-byte boundary, so a
quarter of the 256-bit stores cross a 64-byte cache line. Inside L1 that costs
little; past L1 a store that splits a line it does not own is expensive.
Measured on the i9-14900KF (2026-10-01): the radix-16 leaf over 513 columns
took 7296 ns writing an odd pitch against 4957 an even one; this twin takes
4325 at the odd pitch. FFTW's AVX double load/store macros are 128-bit halves
for the same reason.

The halves are not a general improvement: they cost 3-30% inside L1 and on
the mid stages (`t2c`, 9-16% at every pitch), and at radix 32/64 the leaf is
level either way. So they are a raced form of the forward leaf only: the 2D
real r2c plan times each leaf form on the whole column pass at the cell's own
pitch and banks the winner by name in `forms=` on the 2D row (`h` beside `-`;
`b48h` / `b84h` / `b88h` beside `b48` / `b84` / `b88`). The pool is
`vfft_il2p_col_forms` in `src/core/il/rank1/il2p.h`. A c2r plan never takes
them: its backward leaf runs first and out of place into its scratch plane, a
different question left as its own piece of work, and the backward of a
half-store form name resolves to the full-store twin.

## Emission

`generator/lib/gen/c2c_il.ml` (`gen_radix R --cil-n1c --cil-st128`, blocked:
`--cil-blocked --cil-split A.B --cil-form-tag --cil-st128`; the generator
refuses `--cil-bwd` with it), the store edge in `lib/cx/cx_render.ml`
(`store128`, the `AZoutLeg` halves); rows in `corpus.ml`. AVX2 only (the twin
is the halves of a 256-bit column pair). Regeneration: through `gen_set` to a
temporary root (`--root`), never in place; the folder a file lands in follows
its kind (`Corpus.dir_of_file`), and the law is byte identity against the
shipped file. The map of the whole tree, the two layouts and the rules are in
[`../../../../README.md`](../../../../README.md).
