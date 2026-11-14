# VectorFFT v1.0 — FP64 results, **Ryzen 7 9800X3D**

Double precision on an AMD Ryzen 7 9800X3D, AVX-512 build, one thread, against FFTW 3.3.10. Speedup = FFTW's
time / VectorFFT's time: above 1 means VectorFFT is faster.

One run, planned by the estimate (`VFFT_FLAG_ESTIMATE`): nothing was measured on this machine and no wisdom was
read.

- [18 selected transforms](#18-selected-transforms-estimate-plans) in 1D, 2D and 3D: geometric mean 1.62×, ahead
  on 16 of 18.

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

The run's files record neither the CPU nor the instruction set: both are as reported by the person who ran it,
and the cache sizes are the CPU's specification.

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
