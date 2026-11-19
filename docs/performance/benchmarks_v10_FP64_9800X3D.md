# VectorFFT v1.0 — FP64 and FP32 results, **Ryzen 7 9800X3D**

Double and single precision on an AMD Ryzen 7 9800X3D, AVX-512 build, one thread, against FFTW 3.3.10. Speedup =
FFTW's time / VectorFFT's time: above 1 means VectorFFT is faster.

- [18 selected transforms](#18-selected-transforms-estimate-plans) in 1D, 2D and 3D, double precision, planned by
  the estimate (`VFFT_FLAG_ESTIMATE`): nothing was measured on this machine and no wisdom was read. Geometric mean
  1.62×, ahead on 16 of 18.
- [Every 1D length from 2 to 4,096](#every-1d-length-from-2-to-4096-estimate-plans), double precision, planned by
  the estimate: geometric mean 2.10×, ahead on 3,939 of the 4,095 lengths.
- [1D powers of two in single precision](#1d-powers-of-two-single-precision-planned-by-measuring), 64 to
  4,194,304 points, planned by measuring: geometric mean 2.27×, ahead on 17 of 17.

This host is not the i9-14900KF of [v1_0_results.md](v1_0_results.md), the Zen 4 laptop of
[v1_0_Zen4_results.md](v1_0_Zen4_results.md), the Xeon 6980P of
[benchmarks_v10_FP64_6980P.md](benchmarks_v10_FP64_6980P.md) or the Xeon 8558 of
[benchmarks_v10_8558.md](benchmarks_v10_8558.md): the numbers are **not comparable** across these pages.

## The machine

| Property | Value |
| --- | --- |
| CPU | AMD Ryzen 7 9800X3D (Zen 5) |
| Cores | 8, two threads each |
| L1d | 48 KiB per core |
| L2 | 1 MiB per core |
| L3 | 96 MiB shared |
| Operating system | Windows |
| Instruction set | AVX-512, the program's default on a CPU that runs it |

The files of the 2026-11-14 runs record neither the CPU nor the instruction set: both are as reported by the person
who ran them. The every-length run's files name the CPU and the AVX-512 build. The cache sizes are the CPU's
specification.

## 18 selected transforms, estimate plans

Complex-to-complex in 1D, 2D and 3D, interleaved, natural order, out of place. The program is
`vectorfft_vs_fftw_selected` of the release package `vectorfft-benchmarks-windows-x64` (the programs of
[`benchmark_scripts/`](../../benchmark_scripts/README.md)) with its default 18 transforms. Every plan comes from
the estimate (`VFFT_FLAG_ESTIMATE`); before timing, the program checked that the plan it ran was the one the
estimate produced. FFTW is the package's own 3.3.10 with its SSE2, AVX, AVX2 and AVX-512 kernels, its plans made
with `FFTW_MEASURE`. Run `vectorfft_vs_fftw_selected_20261114-103124`, one pass of 114 s on 2026-11-14.

The times and the `vs FFTW` column are the worse of the two engine orders (VectorFFT timed first, FFTW timed
first); the last column is the speedup in the other order.

| Transform | Points | VectorFFT | FFTW | vs FFTW | Other order |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1D 1920 | 1,920 | 2.05 µs | 1.70 µs | 0.83× | 1.06× |
| 1D 2048 | 2,048 | 1.57 µs | 2.18 µs | **1.39×** | 1.58× |
| 1D 2560 | 2,560 | 1.84 µs | 2.51 µs | **1.37×** | 1.54× |
| 1D 3072 | 3,072 | 2.47 µs | 3.23 µs | **1.31×** | 1.52× |
| 1D 3840 | 3,840 | 3.67 µs | 3.64 µs | 0.99× | 1.13× |
| 1D 1M | 1,048,576 | 2.01 ms | 6.28 ms | **3.12×** | 3.14× |
| 1D 4M | 4,194,304 | 10.3 ms | 30.8 ms | **3.00×** | 3.05× |
| 2D 512×512 | 262K | 348 µs | 643 µs | **1.85×** | 1.90× |
| 2D 1024×1024 | 1M | 2.09 ms | 2.94 ms | **1.40×** | 1.46× |
| 2D 2048×1024 | 2M | 4.04 ms | 6.29 ms | **1.56×** | 1.57× |
| 2D 2048×2048 | 4M | 10.9 ms | 16.7 ms | **1.53×** | 1.58× |
| 2D 1080×1920 | 2.07M | 4.55 ms | 7.38 ms | **1.62×** | 1.73× |
| 2D 720×1280 | 922K | 1.67 ms | 3.36 ms | **2.01×** | 2.01× |
| 3D 128³ | 2M | 3.46 ms | 6.43 ms | **1.86×** | 1.87× |
| 3D 256×128×128 | 4M | 8.39 ms | 17.1 ms | **2.04×** | 2.13× |
| 3D 256×256×64 | 4M | 8.33 ms | 19.5 ms | **2.34×** | 2.50× |
| 3D 64×256×256 | 4M | 14.0 ms | 16.5 ms | 1.18× | 1.18× |
| 3D 256³ | 16M | 58.8 ms | 87.6 ms | **1.49×** | 1.55× |
| **Geometric mean** | | | | **1.62×** | |

VectorFFT is ahead on 16 of the 18 transforms and behind on two, both in one engine order only.

### Reading the selected transforms

- **Strongest:** 1D at 1M and 4M points, 3.0× to 3.1×, and the 3D volume 256×256×64 at 2.34×. Every 2D plane is
  ahead, 1.40× to 2.01×, the two video frames at 1.62× (1080×1920) and 2.01× (720×1280).
- **Behind:** 1D 1920 (0.83×) and 1D 3840 (0.99×). Each is above 1 in the other engine order (1.06× and 1.13×).
- **3D:** 64×256×256 is the lowest volume at 1.18×, where the same 4M points as 256×256×64 read 2.34×.
- **Noise:** the control transforms were timed at the start and at the end of the run, in both engine orders.
  1D 4096 read 1.61×, 1.33×, 1.66× and 1.63×; 2D 64×64 read 2.20×, 1.74×, 2.17× and 1.82×; 3D 64³ read 2.36×,
  1.96×, 2.33× and 1.86×. VectorFFT's own time moved by 12 % to 26 % between the two engine orders at the 1D
  transforms from 1920 to 3840 points and at the 64×64 and 64³ controls, while FFTW's moved by 3 % or less; from
  512×512 up the two orders agree within 5 %. This is one pass: the figures of the small transforms carry that
  spread.
- **Accuracy:** VectorFFT's output agrees with FFTW's to 3.0e-15 or better at every transform.

## Every 1D length from 2 to 4,096, estimate plans

Complex-to-complex in 1D at every length from 2 to 4,096 points, 4,095 lengths, double precision, interleaved,
natural order, out of place. The program is the AVX-512 build of `vectorfft_vs_fftw_selected` (the programs of
[`benchmark_scripts/`](../../benchmark_scripts/README.md)), given the lengths to time (`--cells`) in 16 runs on
2026-11-18, built from VectorFFT 0.8.0. Every plan comes from the estimate
(`VFFT_FLAG_ESTIMATE`); before timing, the program checked that the plan it ran was the one the estimate produced.
FFTW is the package's own 3.3.10 with its SSE2, AVX, AVX2 and AVX-512 kernels, its plans made with `FFTW_MEASURE`.

Each speedup is the worse of the two engine orders (VectorFFT timed first, FFTW timed first).

| Lengths | Count | Geometric mean | Ahead | Level | Behind | Lowest | Highest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All lengths | 4,095 | **2.10×** | 3,939 | 4 | 152 | 0.68× | 6.61× |
| Powers of two | 12 | **1.12×** | 9 | 1 | 2 | 0.80× | 1.65× |
| Primes | 564 | **1.93×** | 542 | 1 | 21 | 0.77× | 4.98× |
| The other lengths | 3,520 | **2.13×** | 3,388 | 3 | 129 | 0.68× | 6.61× |
| 2 to 64 | 63 | **1.18×** | 42 | 1 | 20 | 0.68× | 2.47× |
| 65 to 512 | 448 | **1.98×** | 421 | 0 | 27 | 0.74× | 6.61× |
| 513 to 4,096 | 3,584 | **2.13×** | 3,476 | 3 | 105 | 0.68× | 6.02× |

### Reading every length

- **Ahead on 3,939 of the 4,095 lengths**, level on 4 and behind on 152. 3,273 lengths read 1.5× or more and 2,131
  read 2× or more.
- **Highest:** 323 (6.61×), 3,266 (6.02×), 2,414 (6.01×), 2,698 (5.94×) and 437 (5.81×).
- **Lowest:** most of the slowest lengths have one prime factor between 151 and 449: 562 = 2 × 281 and 566 = 2 × 283
  (0.68×), 604 = 4 × 151 (0.70×), 1,347 = 3 × 449 (0.72×), then 628, 302, 879, 878, 1,345 and the prime 439 (0.73×
  to 0.77×). The others are short lengths, 12 (0.68×), 15 (0.70×) and 48 (0.76×), and two smooth ones, 896 = 2⁷ × 7
  (0.76×) and 1,600 = 2⁶ × 5² (0.77×).
- **Powers of two:** 1.12×. Behind at 8 (0.98×) and 16 (0.80×); from 32 up 1.08× to 1.65×, with 512 at 1.00×.
- **Short lengths:** from 2 to 64, 1.18×, behind on 20 of the 63.
- **Accuracy:** VectorFFT's output agrees with FFTW's to 2.0e-15 or better at every length (the largest difference,
  1.95e-15, at 2,971).

## 1D powers of two, single precision, planned by measuring

Complex-to-complex in 1D at the 17 powers of two from 64 to 4,194,304 points, single precision (FP32),
interleaved, natural order, out of place. The program is `vectorfft_vs_fftw_pow2_fp32` of
[`benchmark_scripts/`](../../benchmark_scripts/README.md). The FP32 library has no estimate: every plan is the
planner's, measured on this machine and kept in its wisdom folder, or read back from it. FFTW is FFTW 3.3.10 in
single precision, its plans made with `FFTW_MEASURE`. Run `vectorfft_vs_fftw_pow2_fp32_20261114-230111`, one pass
of 49 s on 2026-11-14.

The times and the speedup are the worse of the two engine orders (VectorFFT timed first, FFTW timed first).

| Transform | Points | VectorFFT | FFTW | vs FFTW |
| --- | ---: | ---: | ---: | ---: |
| 1D 64 | 64 | 0.01 µs | 0.03 µs | **2.08×** |
| 1D 128 | 128 | 0.03 µs | 0.05 µs | **1.88×** |
| 1D 256 | 256 | 0.06 µs | 0.10 µs | **1.72×** |
| 1D 512 | 512 | 0.12 µs | 0.20 µs | **1.75×** |
| 1D 1024 | 1,024 | 0.29 µs | 0.45 µs | **1.55×** |
| 1D 2048 | 2,048 | 0.65 µs | 1.05 µs | **1.62×** |
| 1D 4096 | 4,096 | 1.51 µs | 2.43 µs | **1.60×** |
| 1D 8192 | 8,192 | 3.72 µs | 7.77 µs | **2.09×** |
| 1D 16384 | 16,384 | 8.69 µs | 14.1 µs | **1.62×** |
| 1D 32768 | 32,768 | 19.8 µs | 32.6 µs | **1.65×** |
| 1D 65536 | 65,536 | 47.5 µs | 73.9 µs | **1.56×** |
| 1D 131072 | 131,072 | 131 µs | 181 µs | **1.39×** |
| 1D 262144 | 262,144 | 276 µs | 407 µs | **1.47×** |
| 1D 512K | 524,288 | 473 µs | 2.94 ms | **6.22×** |
| 1D 1M | 1,048,576 | 984 µs | 5.87 ms | **5.96×** |
| 1D 2M | 2,097,152 | 2.01 ms | 12.2 ms | **6.08×** |
| 1D 4M | 4,194,304 | 4.32 ms | 25.5 ms | **5.89×** |
| **Geometric mean** | | | | **2.27×** |

VectorFFT is ahead on all 17 lengths.

### Reading the powers of two

- **Up to 262,144 points:** 1.39× to 2.09×, every length ahead.
- **From 524,288 points:** 5.9× to 6.2×. The step is FFTW's: from 262,144 to 524,288 points its time grows 7.2×
  (407 µs to 2.94 ms) while VectorFFT's grows 1.7× (276 µs to 473 µs); above that, both double with each doubling
  of the length. This is one pass, not yet repeated.
- **Noise:** the control transform, 1D 4096, read 1.28× and 1.55× in its two timings during the run.
