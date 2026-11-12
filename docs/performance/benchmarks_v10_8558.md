# VectorFFT v1.0 — FP64 in-place results, **Emerald Rapids (Xeon Platinum 8558)**

Double precision, **in place on both libraries** (one buffer, the input overwritten by the output), one thread,
against Intel MKL (sequential). Every run was made twice: with the AVX-512 build against MKL, and with the AVX2
build against MKL held to AVX2. Speedup = MKL's time / VectorFFT's time: above 1 means VectorFFT is faster.

- [1D, every power of two from 64 to 4M](#1d-every-power-of-two-from-64-to-4m-estimate-plans), estimate plans:
  geometric mean 1.57× at AVX-512 (ahead on 16 of 17 lengths), 1.02× at AVX2 (9 of 17).
- [1D, 64 to 256K, the planner's plans](#1d-64-to-256k-the-planners-plans), beside the estimate's: 1.31× against
  1.27× at AVX-512, 1.04× against 1.01× at AVX2.
- [2D, power-of-two planes](#2d-power-of-two-planes-estimate-plans), estimate plans: 1.20× at AVX-512 (19 of 22
  planes), 1.41× at AVX2 (21 of 22).
- [3D, power-of-two volumes](#3d-power-of-two-volumes-estimate-plans), estimate plans: 1.24× at AVX-512 and 1.31× at
  AVX2, ahead on 15 of 16 volumes at both.

This host is not the i9-14900KF of [v1_0_results.md](v1_0_results.md), the Zen 4 laptop of
[v1_0_Zen4_results.md](v1_0_Zen4_results.md) or the Xeon 6980P of
[benchmarks_v10_FP64_6980P.md](benchmarks_v10_FP64_6980P.md): the numbers are **not comparable** across these
pages. The other pages time out of place; this one times in place.

## The machine

| Property | Value |
| --- | --- |
| CPU | Intel Xeon Platinum 8558 (Emerald Rapids), inferred: see below |
| Seen by the guest | "Intel Xeon Processor @ 2.10GHz", a cloud virtual machine |
| L1d | 48 KiB per core |
| L2 | 2 MiB per core |
| L3 | 260 MiB shared |

The hypervisor hides the model name. Among 5th-generation Xeons (Emerald Rapids) only the 48-core 8558 family has
a 260 MB L3, and of its three parts (8558 at 2.1 GHz, 8558U at 2.0 GHz, 8558P at 2.7 GHz) the 2.10 GHz base clock
the guest reports matches the Platinum 8558. A cloud-only part with the same cache would look identical from inside
the machine, so the model is an inference, not a reading.

## 1D, every power of two from 64 to 4M, estimate plans

1D complex-to-complex, interleaved, natural order, in place on both libraries. Every plan comes from the estimate
(`VFFT_FLAG_ESTIMATE`): nothing was measured on this machine and no wisdom was read. Each figure is the median of 4
samples (2 passes, both engine orders in each).

| N | Route | AVX-512: VectorFFT | MKL | vs MKL | AVX2: VectorFFT | MKL (AVX2) | vs MKL |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | pair | 0.051 µs | 0.047 µs | 0.92× | 0.064 µs | 0.063 µs | 0.99× |
| 128 | pair | 0.108 µs | 0.127 µs | 1.18× | 0.142 µs | 0.123 µs | 0.87× |
| 256 | pair | 0.263 µs | 0.370 µs | **1.41×** | 0.272 µs | 0.282 µs | 1.04× |
| 512 | pair | 0.562 µs | 0.582 µs | 1.04× | 0.589 µs | 0.664 µs | 1.13× |
| 1024 | ZTURN-T | 1.14 µs | 1.35 µs | 1.18× | 1.47 µs | 1.53 µs | 1.04× |
| 2048 | ZTURN-T | 2.95 µs | 3.37 µs | 1.14× | 3.77 µs | 3.26 µs | 0.86× |
| 4096 | ZTURN-T | 6.75 µs | 7.65 µs | 1.13× | 7.80 µs | 9.26 µs | 1.19× |
| 8192 | ZTURN-T | 15.9 µs | 19.3 µs | 1.22× | 17.8 µs | 16.5 µs | 0.93× |
| 16K | ZTURN-T | 33.2 µs | 35.8 µs | 1.08× | 40.9 µs | 41.2 µs | 1.01× |
| 32K | ZTURN-T | 80.2 µs | 93.4 µs | 1.17× | 91.4 µs | 88.0 µs | 0.96× |
| 64K | ZTURN-T | 284 µs | 297 µs | 1.04× | 313 µs | 328 µs | 1.05× |
| 128K | ZTURN-T | 803 µs | 1.70 ms | **2.11×** | 796 µs | 920 µs | 1.16× |
| 256K | ZTURN-T | 1.58 ms | 4.06 ms | **2.58×** | 1.70 ms | 1.56 ms | 0.92× |
| 512K | four-step | 3.50 ms | 8.45 ms | **2.42×** | 3.95 ms | 3.29 ms | 0.83× |
| 1M | four-step | 7.29 ms | 18.2 ms | **2.49×** | 8.19 ms | 7.75 ms | 0.95× |
| 2M | four-step | 14.9 ms | 52.5 ms | **3.52×** | 17.4 ms | 20.6 ms | 1.18× |
| 4M | four-step | 31.6 ms | 156 ms | **4.93×** | 35.4 ms | 50.6 ms | **1.43×** |
| **Geometric mean** | | | | **1.57×** | | | 1.02× |

The program's own score, the geometric mean with the worse engine order in each pass, was 1.45× and 1.56× at
AVX-512 and 0.95× and 0.96× at AVX2.

### Reading the 1D powers of two

- **From 128K up, the AVX-512 lead is mostly MKL's.** MKL in place at AVX-512 takes 1.8× to 3.1× its own time at
  AVX2 there (1.70 ms against 920 µs at 128K, 156 ms against 50.6 ms at 4M), while VectorFFT's time barely changes
  with the width (803 µs against 796 µs at 128K). Against MKL's faster path at each length (the lower of its two
  times), the AVX-512 build is ahead 1.11× (14 of 17): behind at 64 (0.92×), 256K (0.99×) and 512K (0.94×), ahead
  1.38× at 2M and 1.60× at 4M.
- **AVX2 is about even:** behind at 128, 2048, 8192 and 256K to 1M (0.83× to 0.93×), ahead at 4096 (1.19×), 2M
  (1.18×) and 4M (1.43×).
- **Noise:** the control cell read 0.88× to 1.51× within one AVX2 pass, so a difference under about 10% at one
  length is within this machine's noise.

## 1D, 64 to 256K, the planner's plans

The same transforms with the plans of the library's default planner, which measures: each length was planned on
this machine first, in place. The estimate's speedups are those of the run above, at the same lengths. Each figure
is the median of 4 samples.

**AVX-512, against MKL:**

| N | Route | VectorFFT | MKL | Planner's plans | Estimate's plans |
| ---: | --- | ---: | ---: | ---: | ---: |
| 64 | pair | 0.051 µs | 0.045 µs | 0.88× | 0.92× |
| 128 | pair | 0.098 µs | 0.117 µs | 1.19× | 1.18× |
| 256 | pair | 0.212 µs | 0.261 µs | 1.23× | **1.41×** |
| 512 | pair | 0.460 µs | 0.591 µs | 1.29× | 1.04× |
| 1024 | ZTURN-T | 1.13 µs | 1.40 µs | 1.24× | 1.18× |
| 2048 | ZTURN-T | 2.96 µs | 3.30 µs | 1.12× | 1.14× |
| 4096 | ZTURN-T | 6.55 µs | 6.69 µs | 1.02× | 1.13× |
| 8192 | ZTURN-T | 15.6 µs | 18.8 µs | 1.21× | 1.22× |
| 16K | ZTURN-T | 31.9 µs | 36.2 µs | 1.14× | 1.08× |
| 32K | ZTURN-T | 71.2 µs | 86.4 µs | 1.21× | 1.17× |
| 64K | ZTURN-T | 252 µs | 297 µs | 1.18× | 1.04× |
| 128K | ZTURN-T | 720 µs | 1.83 ms | **2.54×** | **2.11×** |
| 256K | ZTURN-T | 1.59 ms | 4.45 ms | **2.79×** | **2.58×** |
| **Geometric mean** | | | | **1.31×** | 1.27× |

**AVX2, against MKL held to AVX2:**

| N | Route | VectorFFT | MKL (AVX2) | Planner's plans | Estimate's plans |
| ---: | --- | ---: | ---: | ---: | ---: |
| 64 | pair | 0.063 µs | 0.067 µs | 1.07× | 0.99× |
| 128 | pair | 0.109 µs | 0.142 µs | **1.31×** | 0.87× |
| 256 | pair | 0.276 µs | 0.271 µs | 0.98× | 1.04× |
| 512 | pair | 0.638 µs | 0.628 µs | 0.99× | 1.13× |
| 1024 | ZTURN-T | 1.52 µs | 1.51 µs | 0.99× | 1.04× |
| 2048 | ZTURN-T | 3.65 µs | 3.44 µs | 0.94× | 0.86× |
| 4096 | ZTURN-T | 8.11 µs | 8.52 µs | 1.05× | 1.19× |
| 8192 | ZTURN-T | 18.5 µs | 18.5 µs | 1.00× | 0.93× |
| 16K | ZTURN-T | 39.9 µs | 45.6 µs | 1.14× | 1.01× |
| 32K | ZTURN-T | 91.5 µs | 102.5 µs | 1.12× | 0.96× |
| 64K | ZTURN-T | 320 µs | 335 µs | 1.05× | 1.05× |
| 128K | ZTURN-T | 818 µs | 944 µs | 1.15× | 1.16× |
| 256K | ZTURN-T | 1.82 ms | 1.45 ms | 0.79× | 0.92× |
| **Geometric mean** | | | | 1.04× | 1.01× |

The program's score with the planner's plans was 1.15× and 1.30× at AVX-512 and 1.00× and 0.98× at AVX2.

At AVX2 the planner's plans were ahead on 7 of the 13 lengths. The samples behind each, and the plan the planner
saved:

| N | vs MKL | Range of the 4 samples | Samples ahead | Saved plan |
| ---: | ---: | ---: | ---: | --- |
| 64 | 1.07× | 0.88–1.11× | 3 of 4 | pair 4 × 16 |
| 128 | 1.31× | 0.89–1.60× | 3 of 4 | pair 16 × 8 |
| 4096 | 1.05× | 0.84–1.17× | 3 of 4 | ZTURN-T 8.8.8.8 |
| 16K | 1.14× | 1.10–1.31× | 4 of 4 | ZTURN-T 8.8.8.4.8 |
| 32K | 1.12× | 0.90–1.31× | 3 of 4 | ZTURN-T 8.4.4.8.8.4 |
| 64K | 1.05× | 0.96–1.14× | 3 of 4 | ZTURN-T 8.4.8.4.8.8 |
| 128K | 1.15× | 1.08–1.32× | 4 of 4 | ZTURN-T 8.8.4.8.8.8 |

### Reading the planner's runs

- **The estimate is within a few percent of the planner on a machine it was never fitted on:** 1.27× against 1.31×
  at AVX-512, 1.01× against 1.04× at AVX2. Both take the same route at every length (the pair up to 512, ZTURN-T
  from 1024). The radix orders differ at 5 of the 7 lengths above (the estimate's at AVX2 for this machine's
  caches: pair 4 × 32 at 128, ZTURN-T 8.4.8.8.8 at 16K, 8.8.8.8.8 at 32K, 8.4.4.8.8.8 at 64K, 8.4.8.8.8.8 at
  128K), and the times stay close.
- **Solid at AVX2:** 16K and 128K, ahead in all four samples. The other five above have one sample at or below 1.
- **Behind:** 256K at AVX2 with both plans (0.79× planned, 0.92× estimate), and 64 at AVX-512 with both (0.88×,
  0.92×; MKL takes 45 ns there). At AVX2 the planner's plans also trail at 256 to 2048 (0.94× to 0.99×).
- **The planner's and the estimate's columns come from separate runs.** The control cell ranged 0.78× to 1.35×
  across the passes, so a gap under about 10% between the two columns at one length is noise; 128 at AVX2 (1.31×
  against 0.87×) is the one large gap.

## 2D, power-of-two planes, estimate plans

2D complex-to-complex, interleaved, natural order, in place on both libraries, one thread. Every plan comes from the
estimate (`VFFT_FLAG_ESTIMATE`).

| Plane | Points | AVX-512: VectorFFT | MKL | vs MKL | AVX2: VectorFFT | MKL (AVX2) | vs MKL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16×16 | 256 | 0.175 µs | 0.202 µs | 1.15× | 0.238 µs | 0.241 µs | 1.01× |
| 32×32 | 1K | 0.996 µs | 1.06 µs | 1.06× | 1.36 µs | 1.38 µs | 1.02× |
| 64×64 | 4K | 6.35 µs | 7.83 µs | 1.23× | 8.90 µs | 11.9 µs | **1.34×** |
| 128×128 | 16K | 35.2 µs | 30.5 µs | 0.87× | 50.6 µs | 51.4 µs | 1.02× |
| 256×256 | 64K | 214 µs | 225 µs | 1.05× | 246 µs | 270 µs | 1.10× |
| 64×1024 | 64K | 162 µs | 247 µs | **1.53×** | 225 µs | 243 µs | 1.08× |
| 1024×64 | 64K | 313 µs | 249 µs | 0.80× | 333 µs | 407 µs | 1.22× |
| 128×512 | 64K | 184 µs | 213 µs | 1.15× | 222 µs | 311 µs | **1.40×** |
| 512×128 | 64K | 238 µs | 352 µs | **1.48×** | 388 µs | 322 µs | 0.83× |
| 512×512 | 256K | 1.19 ms | 1.54 ms | 1.29× | 1.39 ms | 1.90 ms | **1.36×** |
| 256×1024 | 256K | 1.23 ms | 1.37 ms | 1.11× | 1.43 ms | 1.94 ms | **1.36×** |
| 1024×256 | 256K | 1.38 ms | 1.86 ms | **1.35×** | 1.63 ms | 2.04 ms | 1.25× |
| 128×4096 | 512K | 2.75 ms | 2.57 ms | 0.94× | 3.16 ms | 3.59 ms | 1.14× |
| 4096×128 | 512K | 2.95 ms | 3.59 ms | 1.22× | 3.85 ms | 4.37 ms | 1.14× |
| 1024×1024 | 1M | 6.04 ms | 7.45 ms | 1.23× | 6.98 ms | 10.6 ms | **1.52×** |
| 512×2048 | 1M | 5.02 ms | 7.25 ms | **1.44×** | 5.53 ms | 8.55 ms | **1.55×** |
| 2048×512 | 1M | 6.48 ms | 7.12 ms | 1.10× | 6.69 ms | 9.91 ms | **1.48×** |
| 1024×2048 | 2M | 12.4 ms | 15.4 ms | 1.24× | 13.6 ms | 29.2 ms | **2.15×** |
| 2048×1024 | 2M | 12.5 ms | 16.1 ms | 1.28× | 14.1 ms | 33.8 ms | **2.40×** |
| 2048×2048 | 4M | 28.6 ms | 39.1 ms | **1.37×** | 31.2 ms | 81.6 ms | **2.62×** |
| 1024×4096 | 4M | 29.5 ms | 36.2 ms | 1.23× | 32.7 ms | 89.1 ms | **2.72×** |
| 4096×1024 | 4M | 26.3 ms | 45.3 ms | **1.72×** | 30.2 ms | 63.3 ms | **2.10×** |
| **Geometric mean** | | | | **1.20×** | | | **1.41×** |

### Reading the 2D planes

- **AVX-512: ahead on 19 of 22 planes.** Against MKL's faster path at each plane (the lower of its two times; its
  AVX2 path is the faster one only at 64×1024 and 512×128), the AVX-512 build is still ahead 1.20×, on the same 19
  planes. Behind at 1024×64 (0.80×), 128×128 (0.87×) and 128×4096 (0.94×).
- **AVX2: ahead on 21 of 22.** From 1M points up, MKL at AVX2 takes 1.2× to 2.5× its own time at AVX-512, while
  VectorFFT's grows 3% to 16%: the 1.5× to 2.7× there is largely MKL's AVX2 path. Behind at 512×128 (0.83×), where
  VectorFFT's AVX2 time is 1.63× its AVX-512 time against 1.03× to 1.44× on the other planes.
- **The tall narrow planes are uneven across the widths:** 1024×64 is 0.80× at AVX-512 and 1.22× at AVX2; 512×128 is
  1.48× and 0.83×.

## 3D, power-of-two volumes, estimate plans

3D complex-to-complex, interleaved, natural order, in place on both libraries, one thread. Every plan comes from the
estimate (`VFFT_FLAG_ESTIMATE`). Each figure is the median of 4 samples. The largest error against MKL's spectrum
was 5.4e-16.

| Volume | Points | AVX-512: VectorFFT | MKL | vs MKL | AVX2: VectorFFT | MKL (AVX2) | vs MKL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8×8×8 | 512 | 0.732 µs | 0.497 µs | 0.68× | 0.662 µs | 0.547 µs | 0.83× |
| 16×16×16 | 4K | 5.85 µs | 6.71 µs | 1.15× | 7.07 µs | 10.8 µs | **1.53×** |
| 32×32×32 | 32K | 55.2 µs | 69.9 µs | 1.27× | 70.6 µs | 104 µs | **1.47×** |
| 16×32×64 | 32K | 59.0 µs | 71.0 µs | 1.20× | 81.8 µs | 97.4 µs | 1.19× |
| 64×32×16 | 32K | 71.8 µs | 81.5 µs | 1.13× | 75.5 µs | 98.7 µs | **1.31×** |
| 64×64×64 | 256K | 857 µs | 1.32 ms | **1.54×** | 1.16 ms | 1.47 ms | 1.27× |
| 32×64×128 | 256K | 871 µs | 1.42 ms | **1.62×** | 1.10 ms | 1.56 ms | **1.41×** |
| 128×64×32 | 256K | 847 µs | 1.27 ms | **1.50×** | 1.00 ms | 1.34 ms | **1.33×** |
| 64×64×256 | 1M | 4.11 ms | 5.82 ms | **1.42×** | 5.29 ms | 6.92 ms | **1.31×** |
| 256×64×64 | 1M | 4.40 ms | 5.76 ms | **1.31×** | 5.06 ms | 7.03 ms | **1.39×** |
| 128×128×128 | 2M | 9.26 ms | 12.1 ms | **1.30×** | 11.4 ms | 15.7 ms | **1.37×** |
| 128×128×256 | 4M | 26.0 ms | 31.3 ms | 1.20× | 25.7 ms | 33.6 ms | **1.31×** |
| 256×128×128 | 4M | 21.1 ms | 31.4 ms | **1.49×** | 24.2 ms | 35.6 ms | **1.47×** |
| 256×256×64 | 4M | 19.9 ms | 26.2 ms | **1.32×** | 23.1 ms | 31.0 ms | **1.34×** |
| 64×256×256 | 4M | 29.2 ms | 32.0 ms | 1.10× | 30.9 ms | 41.1 ms | **1.33×** |
| 256×256×256 | 16M | 166 ms | 171 ms | 1.03× | 157 ms | 210 ms | **1.34×** |
| **Geometric mean** | | | | 1.24× | | | **1.31×** |

The program's own score, the geometric mean with the worse engine order in each pass, was 1.16× and 1.19× at
AVX-512 and 1.24× and 1.26× at AVX2.

### Reading the 3D volumes

- **Behind only at 8×8×8** (0.68× and 0.83×), where MKL takes half a microsecond: per-call cost, not the transform.
- **Strongest at AVX-512 in the middle,** 1.50× to 1.62× at 256K points; weakest at 256³ (1.03×) and 64×256×256
  (1.10×).
- **MKL's AVX-512 path is its faster one on every volume,** so the 1.24× holds against MKL's better path too.
- **The AVX2 lead is not mostly MKL slowing down:** MKL at AVX2 takes 1.06× to 1.61× its own AVX-512 time (1.22× in
  geometric mean), VectorFFT 0.90× to 1.39× (1.15×).
- **VectorFFT's AVX2 build is faster than its own AVX-512 build on three volumes:** 8×8×8, 128×128×256 (25.7 against
  26.0 ms) and 256³ (157 against 166 ms). At 256³ that makes the AVX-512 result the page's weakest 3D figure.
