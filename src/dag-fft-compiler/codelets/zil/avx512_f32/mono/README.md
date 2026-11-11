# mono/ — the whole N-point transform in one call (FP32 on AVX-512, N = 2 .. 1024)

90 files, `radix<N>_z_mono<A>x<M>[y][i]_avx512_f32.c` and `_bwd_`. The same family as
[`../../avx2_f32/mono/`](../../avx2_f32/mono/) (its README describes the constructions), in two register widths:

- **`mono<A>x<M>y`** — the 256-bit kernel of the AVX2 tree under this ISA's name: 4 complex per vector, with 32
  registers to hold its values instead of 16. Every AVX2 shape, the same form numbers.
- **`mono<A>x<M>`** — 512 bits, 8 complex per vector, from 64 = 8 x 8 up. One DAG to 512 points (64 vectors),
  two loops at 1024.
- **N = 2, 4, 8** (`mono1x2`, `mono1x4`, `mono2x4`) — the AVX2 tree's small kernels, 256-bit text.
- A trailing **`i`** is the in-place twin of a two-loop kernel (its plane on its own stack).

| N | forms, in order |
| ---: | --- |
| 2, 4, 8 | 1x2 / 1x4 / 2x4 |
| 16 | 4x4y |
| 32 | 4x8y, 8x4y |
| 64 | 16x4y, 8x8y, 4x16y, 8x8 |
| 128 | 16x8y, 32x4y, 8x16y, 16x8, 8x16 |
| 256 | 8x32y, 32x8y, 16x16y, 16x16, 32x8, 8x32 |
| 512 | 32x16y, 16x32y, 64x8y, 32x16, 16x32, 64x8 |
| 1024 | 32x32y, 64x16y, 16x64y, 32x32, 64x16, 16x64 |

The planner races all the forms of a size on the machine it runs on. **None of them has been timed**: no
AVX-512 machine has raced them yet, so no shape here is known to be the fast one. They are checked for
correctness under Intel SDE (every kernel against a long-double DFT, and in place where it is legal).

Emitted by `c2c_il.ml emit_k1_f32` / `emit_k1_f32_small`
(`gen_radix N --cil-k1 --cil-chain A:M [--cil-ymm] [--cil-plane] [--cil-bwd] --isa avx512_f32`; the y forms at
`--cil-port 50` as in the AVX2 tree, the 512-bit forms at 0). Regeneration: through `gen_set f32` to a temporary
root, never in place.
