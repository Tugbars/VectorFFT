# Re-run the small 2D planes against FFTW on Zen 4

For the Claude agent on the Zen 4 laptop (Ryzen 5 PRO 8640HS, AVX-512 build). Written 2026-10-29 on the 14900KF.

## What changed

On the 2026-10-28 Zen 4 run (`gauntlet/results/fftw_avx512_2026-10-28_2d`), the smallest 2D c2c planes lost to FFTW: 2x2 to 16x2 at 0.46-0.62, every plane of up to 256 points at a median of 0.76.

On the 14900KF we split one call of 2x2, 4x4 and 4x8 into parts: the public `vfft_execute` entry, the route itself (`_il2d_csk_exec`), and the two bare kernel calls. Median ns, 15 rounds, one thread on logical CPU 2:

| plane | whole call | entry | route logic | kernels | FFTW |
|---|---|---|---|---|---|
| 2x2 | 14.3 | 8.4 | 1.7 | 4.2 | 6.7 |
| 4x4 | 16.3 | 5.9 | 1.9 | 8.5 | 11.0 |
| 4x8 | 23.3 | 5.6 | 2.0 | 15.7 | 19.1 |

Without the entry, our route plus kernels already beat FFTW at all three planes. The entry cost came from 2D c2c plans never getting the lean door of 2026-10-11 (`k1_go` was NULL). Every call went through the full door's signature walk.

**The change:** `src/core/il/il_execute.h` now has `_k2x_il2d_c2c`. `_vfft_k1_bind_exec` binds it for every one-thread interleaved 2D c2c plan. Threaded plans keep the full door.

- **Bit-identical:** 29 plans over every route (csk, csk+rb, csk+rb2, chain, chain+rb/rb2, turn; out of place and in place; 2x2 to 1024x1024), both directions plus the aliased call. 154 comparisons against the full door, all bit-identical.
- **On the 14900KF after the change:**

  | plane | before | after | FFTW |
  |---|---|---|---|
  | 2x2 | 14.3 | 9.6 | 6.3 |
  | 4x4 | 16.3 | 14.2 | 11.9 |
  | 4x8 | 23.3 | 20.6 | 20.1 |

## Separate, not fixed by this change: the stack-alignment lottery

GCC on Windows spills vector registers with unaligned moves at a stack offset that differs per process. A zmm spill splits a cache line in 3 of the 4 possible stack positions.

- **Planes of up to 32 points do not spill.** Their AVX-512 kernels (radix 2/4/8/16 `n1`, `n1c`, `n1ccs`, `t2c` and the row kinds) have no vector stack accesses. Their loss was the entry cost.
- **Planes of 64 to 256 points may still flip.** On 10-28 their two readings disagreed: 16x8 254/156, 8x32 494/381, 16x16 351/282, 128x2 379/460. Some of them run kernels that do spill (the radix-16 column kernel `t2cb44`, the 32/64 kernels).

## What to run

1. **Check the change is in your checkout.** `grep -n "_k2x_il2d_c2c" src/core/il/il_execute.h` must find it. If it doesn't, pull first.
2. **Rebuild the tools.** The gauntlet uses prebuilt binaries in `gauntlet/`, and stale ones test old code. Run `python benchmark_scripts/build_tools.py --fftw-dll <the same FFTW DLL as on 10-28>`. On 10-28 that was `gauntlet/fftw/libfftw3-3.dll`, FFTW 3.3.10 with AVX-512.
3. **Run the cells.** Use a quiet machine on mains power; the 10-28 pow2 run on this laptop ran at half speed. One thread, the default 2D contract.

   ```
   python gauntlet/gauntlet.py run --cmp fftw --name fftw_avx512_2026-10-29_2d_small --cells 2x2,2x4,4x2,2x8,4x4,8x2,2x16,4x8,8x4,16x2,2x32,4x16,8x8,16x4,32x2,2x64,4x32,8x16,16x8,32x4,64x2,2x128,4x64,8x32,16x16,32x8,64x4,128x2,8x512,256x256
   ```

   - The calibrate stage replays this CPU's banked 2D rows from a scratch copy of the store. Nothing should race. Do not pass `--merge` or `--calibrate`.
   - 8x512 is the 10-28 spike (33,065 ns in one engine order, 5,883 in the other).
   - 256x256 is a reference that should not move: the entry is negligible there.

## The 10-28 readings to compare against

Ours and FFTW, both engine orders, in ns; ratio = FFTW's better reading / our better one:

| cell | route | ours a | ours b | FFTW a | FFTW b | ratio |
|---|---|---|---|---|---|---|
| 2x2 | csk+rb | 39 | 33 | 18 | 20 | 0.55 |
| 2x4 | csk+rb | 37 | 37 | 20 | 23 | 0.54 |
| 4x2 | csk+rb | 40 | 38 | 19 | 22 | 0.50 |
| 2x8 | csk+rb | 58 | 47 | 30 | 32 | 0.64 |
| 4x4 | csk+rb | 54 | 67 | 39 | 39 | 0.72 |
| 8x2 | csk+rb | 50 | 45 | 29 | 30 | 0.64 |
| 2x16 | csk+rb | 64 | 62 | 50 | 54 | 0.81 |
| 4x8 | csk+rb | 104 | 104 | 65 | 65 | 0.62 |
| 8x4 | csk+rb | 66 | 61 | 45 | 50 | 0.74 |
| 16x2 | csk+rb | 82 | 76 | 51 | 65 | 0.67 |
| 2x32 | csk | 98 | 93 | 166 | 191 | 1.78 |
| 4x16 | csk+rb2 | 136 | 135 | 114 | 111 | 0.82 |
| 8x8 | csk+rb | 116 | 112 | 81 | 98 | 0.72 |
| 16x4 | csk+rb | 134 | 140 | 88 | 118 | 0.66 |
| 32x2 | csk+rb | 103 | 103 | 80 | 92 | 0.78 |
| 2x64 | csk | 136 | 128 | 161 | 169 | 1.26 |
| 4x32 | chain+rb2 | 216 | 216 | 213 | 229 | 0.99 |
| 8x16 | csk+rb | 219 | 262 | 172 | 200 | 0.79 |
| 16x8 | csk+rb | 254 | 156 | 194 | 201 | 1.24 |
| 32x4 | csk+rb | 194 | 207 | 175 | 192 | 0.90 |
| 64x2 | chain+rb | 214 | 218 | 158 | 188 | 0.74 |
| 2x128 | csk | 240 | 239 | 312 | 329 | 1.31 |
| 4x64 | csk | 242 | 243 | 309 | 329 | 1.28 |
| 8x32 | chain+rb | 494 | 381 | 308 | 370 | 0.81 |
| 16x16 | chain+rb2 | 351 | 282 | 311 | 418 | 1.10 |
| 32x8 | chain+rb | 345 | 347 | 305 | 329 | 0.88 |
| 64x4 | chain+rb | 326 | 307 | 309 | 372 | 1.01 |
| 128x2 | turn | 379 | 460 | 396 | 462 | 1.04 |

## What to report back

- **Per cell:** our two readings, FFTW's two, and the new ratio beside the 10-28 ratio.
- **Planes of up to 32 points:** they should gain roughly the removed entry cost, a constant number of ns. On the 14900KF that was 3-5 ns. This laptop runs at lower clocks, so expect more ns there. Say whether the loss is gone or how much is left.
- **Planes of 64 to 256 points:** say whether our two readings still disagree by more than 20%. If they do, that is the stack lottery, not the entry.
- **The control cell (64x64) and 256x256:** say whether they match 10-28. If they don't, the run's clock differs and the ratios are what to compare.
