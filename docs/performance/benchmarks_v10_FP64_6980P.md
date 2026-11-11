# VectorFFT v1.0 — FP64 results, **Granite Rapids (6980P)**

Double precision on an Intel Xeon 6980P, AVX-512 build, one thread, against Intel MKL (sequential). Speedup =
MKL's time / VectorFFT's time: above 1 means VectorFFT is faster.

Two runs, both planned by the estimate (`VFFT_FLAG_ESTIMATE`): nothing was measured on this machine and no
wisdom was read.

- [18 selected transforms](#18-selected-transforms-estimate-plans) in 1D, 2D and 3D: geometric mean 1.40×.
- [1D, every power of two from 64 to 4M](#1d-every-power-of-two-from-64-to-4m-estimate-plans): ahead on 16 of
  17 lengths.

This host is not the i9-14900KF of [v1_0_results.md](v1_0_results.md) or the Zen 4 laptop of
[v1_0_Zen4_results.md](v1_0_Zen4_results.md): the numbers are **not comparable** across these pages. The
single-precision results of the same host are in
[benchmarks_v10_FP32.md](benchmarks_v10_FP32.md#granite-rapids-6980p).

## The machine

| Property | Value |
| --- | --- |
| CPU | Intel Xeon 6980P (Granite Rapids) |
| vCPUs | 2, a shared KVM slice; no turbo visible to the guest |
| L1d | 48 KiB per core |
| L2 | 2 MiB per core |
| L3 | 504 MiB shared (the host's whole L3, not a share reserved for this machine) |

## 18 selected transforms, estimate plans

Complex-to-complex in 1D, 2D and 3D, interleaved, natural order, out of place. The program is
`vectorfft_vs_mkl_selected --avx512` from [`benchmark_scripts/`](../../benchmark_scripts/README.md) with its
default 18 transforms. Every plan comes from the estimate (`VFFT_FLAG_ESTIMATE`): nothing was measured on this
machine and no wisdom was read. Each figure is the median of 3 passes; each pass takes the worse of the two
engine orders.

| Transform | Points | VectorFFT | MKL | vs MKL |
| --- | ---: | ---: | ---: | ---: |
| 1D 1920 | 1,920 | 4.0 µs | 3.4 µs | 0.84× |
| 1D 2048 | 2,048 | 2.7 µs | 3.4 µs | 1.24× |
| 1D 2560 | 2,560 | 3.7 µs | 4.5 µs | 1.21× |
| 1D 3072 | 3,072 | 5.9 µs | 5.6 µs | 1.00× |
| 1D 3840 | 3,840 | 6.8 µs | 6.5 µs | 0.96× |
| 1D 1M | 1,048,576 | 8.02 ms | 20.2 ms | **2.51×** |
| 1D 4M | 4,194,304 | 40.6 ms | 93.2 ms | **2.29×** |
| 2D 512×512 | 262K | 1.21 ms | 2.06 ms | **1.73×** |
| 2D 1024×1024 | 1M | 6.70 ms | 11.3 ms | **1.70×** |
| 2D 2048×1024 | 2M | 15.7 ms | 34.6 ms | **2.22×** |
| 2D 2048×2048 | 4M | 38.6 ms | 67.3 ms | **1.81×** |
| 2D 1080×1920 | 2.07M | 16.3 ms | 22.7 ms | **1.44×** |
| 2D 720×1280 | 922K | 6.09 ms | 8.21 ms | **1.34×** |
| 3D 128³ | 2M | 15.0 ms | 20.2 ms | **1.35×** |
| 3D 256×128×128 | 4M | 35.8 ms | 44.6 ms | **1.30×** |
| 3D 256×256×64 | 4M | 35.5 ms | 42.4 ms | 1.21× |
| 3D 64×256×256 | 4M | 49.9 ms | 56.9 ms | 1.14× |
| 3D 256³ | 16M | 228 ms | 237 ms | 1.04× |
| **Geometric mean** | | | | **1.40×** |

VectorFFT is ahead on 15 of the 18 transforms, level on one (1D 3072) and behind on two.

### Reading the selected transforms

- **Strongest:** 1D at 1M and 4M points, 2.3× to 2.5×, and every 2D plane, 1.3× to 2.2×.
- **Behind:** 1D 1920 (0.84×; the three passes read 0.82×, 0.84× and 1.01×) and 1D 3840 (0.96×; below 1 in all
  three passes). With 3072, these are the selection's three 1D lengths with a factor of 3.
- **3D:** the lead shrinks as the volume grows, from 1.35× at 128³ to 1.04× at 256³ (the three passes there
  read 0.98×, 1.06× and 1.04×).
- **Noise:** this is a shared virtual machine. The control cell (1D 4096) read 1.23× to 2.68× within the third
  pass, so a single-pass figure can be far off; the table holds the medians of the three passes.

## 1D, every power of two from 64 to 4M, estimate plans

1D complex-to-complex, interleaved, natural order, out of place, against Intel oneMKL 2026.0.1 (sequential). The
program is `vectorfft_vs_mkl_pow2` from [`benchmark_scripts/`](../../benchmark_scripts/README.md). Every plan
comes from the estimate; before timing, the program checked that the plan it ran was the one the estimate
produced. Each figure is the median of 4 samples (2 passes, both engine orders in each); the last column is the
lowest and the highest of the four speedups.

| N | Route | VectorFFT | MKL | vs MKL | Range |
| ---: | --- | ---: | ---: | ---: | ---: |
| 64 | pair | 0.041 µs | 0.036 µs | 0.88× | 0.86–0.90× |
| 128 | pair | 0.093 µs | 0.104 µs | 1.12× | 1.04–1.28× |
| 256 | pair | 0.166 µs | 0.312 µs | **1.89×** | 1.32–1.98× |
| 512 | pair | 0.38 µs | 0.467 µs | 1.23× | 1.16–1.28× |
| 1024 | ZTURN-T | 0.899 µs | 1.26 µs | **1.41×** | 1.36–1.44× |
| 2048 | ZTURN-T | 2.10 µs | 2.56 µs | 1.22× | 1.19–1.44× |
| 4096 | ZTURN-T | 5.16 µs | 6.18 µs | 1.20× | 1.17–1.25× |
| 8192 | ZTURN-T | 12.8 µs | 16.0 µs | 1.25× | 1.22–1.28× |
| 16K | ZTURN-T | 28.2 µs | 30.4 µs | 1.08× | 1.05–1.52× |
| 32K | ZTURN-T | 58.3 µs | 117 µs | **2.00×** | 1.37–3.55× |
| 64K | ZTURN-T | 196 µs | 388 µs | **1.98×** | 1.64–2.22× |
| 128K | ZTURN-T | 644 µs | 1.17 ms | **1.82×** | 1.66–1.98× |
| 256K | ZTURN-T | 1.71 ms | 2.78 ms | **1.63×** | 1.43–1.88× |
| 512K | four-step | 3.56 ms | 6.24 ms | **1.75×** | 1.70–2.11× |
| 1M | four-step | 6.54 ms | 11.4 ms | **1.75×** | 1.12–2.09× |
| 2M | four-step | 14.3 ms | 30.1 ms | **2.11×** | 1.70–2.41× |
| 4M | four-step | 26.8 ms | 78.3 ms | **2.92×** | 1.71–3.05× |

VectorFFT is ahead on 16 of the 17 lengths. The program's own score, the geometric mean with the worse engine
order in each pass, was 1.45× in the first pass and 1.41× in the second.

### Reading the powers of two

- **Behind:** N = 64 (0.88×), in both passes.
- **Steadiest:** 128 to 8192, where the four samples agree most closely: 1.12× to 1.41×, and 1.89× at 256.
- **From 32K up** the samples spread widely, MKL's times more than VectorFFT's: 32K read 1.37× to 3.55× and 1M
  read 1.12× to 2.09×. The 2.92× at 4M is not a figure to quote: its samples run from 1.71× to 3.05×.
- **Noise:** the control cell read 1.18× to 1.22× through the runs. The two tables on this page come from
  separate runs: they agree at 2048 (1.24× and 1.22×) and differ at 1M (2.51× and 1.75×) and 4M (2.29× and
  2.92×), which is the uncertainty of the large lengths on this machine.
