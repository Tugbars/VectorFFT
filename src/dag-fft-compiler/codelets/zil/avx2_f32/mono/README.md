# mono/ — the whole N-point transform in one call (FP32, N = 16 .. 1024)

36 files, `radix<N>_z_mono<A>x<M>_avx2_f32.c` and `_bwd_`: N = A x M. Pass 1 is a DFT of A points across the
legs, then a register transpose, then pass 2, a DFT of M points. Natural order in and out, the 11-argument z ABI
with every argument but `zin` and `zout` ignored, nothing to build at plan time.

| N | form 1 | form 2 | form 3 |
| ---: | --- | --- | --- |
| 16 | 4x4 | | |
| 32 | 4x8 | 8x4 | |
| 64 | 16x4 | 8x8 | 4x16 |
| 128 | 16x8 | 32x4 | 8x16 |
| 256 | 8x32 | 32x8 | 16x16 |
| 512 | 32x16 | 16x32 | 64x8 |
| 1024 | 32x32 | 64x16 | 16x64 |

Two constructions:

- **N <= 128**: both passes in registers, every twiddle a constant of the kernel. Every load comes before every
  store, so `zin == zout` is legal: the same kernel serves the in-place plan.
- **N >= 256**: two loops with `zout` as the plane between them (pass 1 stores rows into `zout` and reads its
  twiddles from a table in the kernel's read-only data; pass 2 runs in `zout`). Out of place only.

The K=1 door races them as the MONO route's forms 1, 2, 3 against form 0 (the solo `n1` kind) and the other
engines (`src/core/il/isa/il_solo_registry.h`). The form number is the shape's place in
`Corpus.zil_f32_mono_shapes` and is what the wisdom row banks as `il_kv`, so a row of that table is appended to,
never reordered.

Emitted by `c2c_il.ml emit_k1_f32` (`gen_radix N --cil-k1 --cil-chain A:M [--cil-bwd] --cil-port 50 --isa avx2_f32`;
16 points at port 0), the DFT bodies built by `lib/cx/cx_lazy.ml`. 256-bit only: `avx512_f32` has no such kernels
yet.

Regeneration: through `gen_set f32` to a temporary root (`--root`), never in place; the law is byte identity
against the shipped file.
