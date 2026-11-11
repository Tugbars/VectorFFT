# mono/ — the whole N-point transform in one call (FP32, N = 2 .. 1024)

60 files, `radix<N>_z_mono<A>x<M>[i]_avx2_f32.c` and `_bwd_`. Natural order in and out, the 11-argument z ABI
with every argument but `zin` and `zout` ignored, nothing to build at plan time.

| N | form 1 | form 2 | form 3 |
| ---: | --- | --- | --- |
| 2 | 1x2 | | |
| 4 | 1x4 | | |
| 8 | 2x4 | | |
| 16 | 4x4 | | |
| 32 | 4x8 | 8x4 | |
| 64 | 16x4 | 8x8 | 4x16 |
| 128 | 16x8 | 32x4 | 8x16 |
| 256 | 8x32 | 32x8 | 16x16 |
| 512 | 32x16 | 16x32 | 64x8 |
| 1024 | 32x32 | 64x16 | 16x64 |

Three constructions:

- **N = 2, 4, 8**: the points sit in the lanes of one or two vectors; the stages run inside the vector. Legal in
  place.
- **N = 16 .. 128** (A x M): pass 1 is a DFT of A points across the legs, then a register transpose, then pass 2,
  a DFT of M points. Both passes in registers, every twiddle a constant of the kernel, every load before every
  store: legal in place.
- **N = 256 .. 1024** (A x M): the same two passes as two loops with a plane of N complex between them, the
  twiddles read from a table in the kernel's read-only data. Two kernels per shape and direction:
  `mono<A>x<M>` uses `zout` as the plane and is out of place only; `mono<A>x<M>i` keeps the plane on its own
  stack and is legal in place. The same operations either way.

The K=1 door races them as the MONO route's forms 1, 2, 3 against form 0 (the solo `n1` kind) and the other
engines (`src/core/il/isa/il_solo_registry.h`). The form number is the entry's place in
`Corpus.zil_f32_mono_forms` and is what the wisdom row banks as `il_kv`, so a row of that table is appended to,
never reordered. An in-place plan runs the form's in-place kernel; an out-of-place plan called with one buffer
does too.

Emitted by `c2c_il.ml emit_k1_f32` and `emit_k1_f32_small`
(`gen_radix N --cil-k1 --cil-chain A:M [--cil-plane] [--cil-bwd] [--cil-port 50] --isa avx2_f32`; 16 points and
below at port 0), the DFT bodies built by `lib/cx/cx_lazy.ml`.

Regeneration: through `gen_set f32` to a temporary root (`--root`), never in place; the law is byte identity
against the shipped file.
