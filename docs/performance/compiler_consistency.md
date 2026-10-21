# VectorFFT across compilers: GCC and Intel ICX

The same source built with two compilers, GCC and Intel's ICX, each build racing its own plans
and timed against Intel oneMKL on the same machine in the same session. The question is whether
VectorFFT's speed depends on the compiler that builds it. On every power of two from 64 to 2^23
the two builds run within 5% of each other, and both lead MKL by the same margin.

> **Platform:** Intel Core i9-14900KF (P-core, AVX2), DDR5, single thread  
> **Source:** commit `de62b27e`, the AVX2 build, the gauntlet's flags for both compilers:
> `-O3 -mavx2 -mfma -march=native -mno-avx512f`  
> **Compilers:** GCC 15.2.0 (MinGW-w64, UCRT); Intel oneAPI DPC++/C++ Compiler (ICX) 2025.3.0
> (clang-based, the Windows MSVC runtime, linked with lld)  
> **Competitor:** Intel oneMKL 2025.3, sequential  
> **Method:** the gauntlet's `run --group pow2 --cmp mkl --calibrate`: every length re-raced by
> its own build into a fresh copy of the shipped wisdom, then timed in one process per length,
> both engines in two arms with the engine order flipped, the caller pinned with its SMT sibling
> held, a control cell at 4096. The two runs back to back (control cell: ICX 1.062 to 1.078,
> GCC 1.076 to 1.082 against MKL).

---

## 1D c2c, every power of two from 64 to 2^23

> **Contract:** complex-to-complex FP64, interleaved, natural order, out of place, K = 1, one thread  
> **Records:** [`icx_pow2_mkl_20261021`](../../gauntlet/results/icx_pow2_mkl_20261021/) and
> [`gcc_pow2_mkl_20261021`](../../gauntlet/results/gcc_pow2_mkl_20261021/) (2026-10-21); GCC's
> 128 and 2048 re-timed from that run's own plans in
> [`gcc_pow2_mkl_retime_20261021`](../../gauntlet/results/gcc_pow2_mkl_retime_20261021/), where
> the full run's two arms disagreed by 1.5x

The ns columns are the faster of the two arms; GCC/ICX is GCC's time over ICX's (below 1: GCC
faster); each x is MKL's time over VectorFFT's in the worse of the two arms.

```
         N  plan, ICX build                plan, GCC build                ICX ns     GCC ns  GCC/ICX  x ICX  x GCC
        64  pair 4.16                      pair 4.16                          30         30    1.00    0.99   1.04
       128  pair 8.16                      pair 4.32                          66         64    0.97    1.05   1.07
       256  pair 16.16                     pair 16.16                        138        134    0.97    0.99   1.02
       512  pair 16.32                     pair 16.32                        300        296    0.99    0.99   0.88
      1024  ZTURN-T 4.4.4.4.4              ZTURN-T 4.8.8.4                   727        714    0.98    1.15   1.18
      2048  ZTURN-T 8.4.8.8                ZTURN-T 8.8.8.4                  1623       1605    0.99    1.25   1.26
      4096  ZTURN-T 8.8.8.8 tile 1024      ZTURN-T 8.8.8.8 tile 1024        3607       3543    0.98    1.06   1.07
      8192  ZTURN-T 8.4.4.8.8 tile 1024    ZTURN-T 8.4.4.8.8 tile 1024      7719       7739    1.00    1.10   1.09
     16384  ZTURN-T 8.4.4.4.4.8 tile 2048  ZTURN-T 8.8.8.4.8 tile 2048     17482      17298    0.99    1.08   0.97
     32768  ZTURN-T 8.8.8.8.8 tile 1024    ZTURN-T 8.8.8.8.8 tile 1024     38805      39048    1.01    0.95   0.58*
     65536  ZTURN-T 8.4.4.8.8.8 tile 1024  ZTURN-T 8.4.4.8.8.8 tile 1024   86230      86933    1.01    1.25   1.25
    131072  ZTURN-T 8.4.8.8.8.8 tile 2048  ZTURN-T 8.4.4.4.4.8.8 tile 2048 221487     222800    1.01    1.18   1.17
    262144  ZTURN-T 8.8.8.8.8.8 tile 1024  ZTURN-T 8.8.8.8.8.8 tile 1024  543913     542750    1.00    1.29   1.23
    524288  four-step 1024.512             four-step 2048.256            1431338    1384938    0.97    1.02   1.08
   1048576  four-step 2048.512             four-step 4096.256            3311700    3144950    0.95    1.22   1.29
   2097152  four-step 4096.512             four-step 4096.512            8303887    7915738    0.95    1.27   1.35
   4194304  four-step 1024.4096            four-step 4096.1024          19169338   18441563    0.96    1.24   1.39
   8388608  four-step 4096.2048            four-step 4096.2048          41803887   40587675    0.97    1.43   1.46
```

- **The two builds are level.** GCC/ICX runs from 0.95 to 1.01, median 0.98. The widest gaps,
  3 to 5% in GCC's favour, are the four-step lengths from 2^19 to 2^22, where the two races also
  chose different splits or different inner chains.
- **Both lead MKL by the same margin:** median x 1.12 with ICX and 1.13 with GCC; within a few
  percent of MKL from 64 to 512 (0.88x to 1.07x), ahead at most lengths from 1024, and 1.17x to
  1.46x from 65536 up except at 524288 (1.02x and 1.08x).
- **The planner adapts to the compiler.** The two races chose the same plan at 8 of the 18
  lengths. Where they differ, each build's own choice runs within 5% of the other's, so the
  outcome does not depend on which compiler built the plan's kernels.
- **Accuracy:** every roundtrip error is at most 5.6e-16 with either build.

\* At 32768 with GCC, in both GCC runs (the full run and the re-time), one of the two arms timed
the transform at about 64 µs and the other at about 39 µs; the worse-arm ratio is 0.58. The ICX
build's two arms at 32768 (38.9 and 38.8 µs), and the GCC build in the 2026-10-20 record
([`c2c_1d_pow2_mkl_20261020`](../../gauntlet/results/c2c_1d_pow2_mkl_20261020/), 0.97x), show
no such split. The cause is not established.

---

## How the ICX build was made

`gauntlet/build.py` with `CC` pointing at `icx.exe`, from a scratch copy of the repository at
`de62b27e`, with three local changes the build needed:

- the codelet archive made with `llvm-ar` (beside ICX, in `bin\compiler\`), where `build.py`
  derives the archiver from the compiler's name;
- in `gauntlet/bench_1d_vs_mkl.c`, `strtok_r` mapped to the MSVC runtime's `strtok_s`;
- in the same file, `pace()` sleeping through `Sleep()` instead of `nanosleep`, which the MSVC
  runtime does not provide (the library itself already uses `Sleep()` on Windows).

The library and its codelets built unchanged.
