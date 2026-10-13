# VectorFFT vs FFTW

VectorFFT against [FFTW](https://www.fftw.org/): the 1D complex transform at every length to
4,096, and the real-input transforms in 2D and 3D. Each section is one contract: a plot, a short
reading of it, and the speedups by size band. Every run is a [gauntlet](../../gauntlet/) record
under [`gauntlet/results/`](../../gauntlet/results/) (csv, calibration log, the wisdom it used,
FFTW's own wisdom). The 1D plot is drawn from the run's csv by
[`src/tools/plots/gen_gflops.py`](../../src/tools/plots/gen_gflops.py), the 2D matrix from the
run's report by [`src/tools/plots/gen_2d_pow2.py`](../../src/tools/plots/gen_2d_pow2.py), the 3D
matrices from the csv by [`src/tools/plots/gen_3d_pow2.py`](../../src/tools/plots/gen_3d_pow2.py).

> **Platform:** Intel Core i9-14900KF (P-core, AVX2), DDR5, GCC 15.2, single thread  
> **Competitor:** FFTW 3.3.10, the vcpkg build with its AVX and AVX2 code paths; plans made with FFTW_MEASURE, each run keeping its own FFTW wisdom  
> **Method:** one process per cell (a length, a plane or a volume); both engines timed in two arms with the
> engine order flipped; a control cell re-timed every 100 cells; every speedup is FFTW's time over
> VectorFFT's in the **worse** of the two arms

---

## 1D c2c, every N from 2 to 4,096

> **Contract:** complex-to-complex FP64, interleaved, natural order, out of place, K = 1  
> **Cells:** every length N from 2 to 4,096: 4,095 transforms, no size skipped  
> **Record:** [`c2c_1d_2_4096_fftw_20261013`](../../gauntlet/results/c2c_1d_2_4096_fftw_20261013/) (2026-10-13), every length's plan replayed from a scratch copy of the shipped wisdom, the 309 lengths missing from it raced at create in the run's own store; and [`rader_class_fftw_20261013`](../../gauntlet/results/rader_class_fftw_20261013/) (2026-10-13/14), the 2,157 composites with a prime factor of 53 or more re-raced cold after the Rader leaf, every plan raced at create with the leaf chain among the arms, drawn over the first record

![Throughput, VectorFFT vs FFTW, 1D c2c, every N from 2 to 4096](../../src/tools/plots/vectorfft-fftw-c2c.svg)

One line per engine, one point per length (log scale: equal speed ratios are equal vertical
gaps). VectorFFT is ahead in both engine orders at 3,734 of the 4,095 lengths, median 1.63x,
geomean 1.71x; FFTW leads in both orders at 286, and at the other 75 only one of the two orders
came out slower. The split is by the largest prime factor of N. Composites built from primes up
to 47 run 2.15x faster than FFTW at the median. A composite with a prime factor of 53 or more,
2,157 lengths, runs that prime as a Rader stage of the flat chain where the race chose it, 1,412
lengths at a median of 1.99x, and as a prime-size convolution on the whole length at the other
745, median 1.31x: the class reads 1.64x at the median and FFTW leads in both orders at 24 of
its lengths (2429 = 7 x 347 at 0.66x, 3521 = 7 x 503 at 0.70x). Before the Rader leaf, in the
September record, the same class read 1.12x with FFTW ahead at 754 lengths and 2101 = 11 x 191
at 0.41x. The 563 primes sit at 1.33x. Below 17 FFTW's codelets lead at 12 of the 15 lengths
(0.58x at 12 to 0.87x at 7). The powers of two sit at 1.05x at the median, 1.37x at 4,096. Every
output agrees with FFTW's to 3.7e-15 in the September record and to 1.4e-15 in the class
re-race.

| Lengths | Cells | Median speedup | At or above parity | Best |
|---|---|---|---|---|
| 2..64 | 63 | 1.12x | 65% | 3.34x (N=61) |
| 65..256 | 192 | 1.68x | 88% | 4.55x (N=172) |
| 257..1,024 | 768 | 1.61x | 89% | 4.69x (N=779) |
| 1,025..2,048 | 1,024 | 1.63x | 92% | 4.50x (N=2021) |
| 2,049..4,096 | 2,048 | 1.64x | 93% | 4.40x (N=3149) |
| **all, 2..4,096** | **4,095** | **1.63x** | **91%** | 4.69x (N=779) |

| Family | Cells | Median speedup | At or above parity |
|---|---|---|---|
| Powers of two | 12 | 1.05x | 75% |
| Primes (3..4,093) | 563 | 1.33x | 84% |
| Composites, largest prime factor ≤ 47 | 1,363 | 2.15x | 83% |
| Composites, largest prime factor ≥ 53 | 2,157 | 1.64x | 98% |

---

## 2D r2c, every power-of-two plane up to 4M points

> **Contract:** 2D real-to-complex FP64, N1 x N2 reals in, the N1 x (N2/2 + 1) CCE bins out, interleaved, out of place, K = 1  
> **Cells:** every plane N1 x N2 with each side a power of two from 2 to 8,192 and at most 2^22 points: 146 planes; the 13 planes with N2 = 2 are refused by VectorFFT's 2D r2c and not timed  
> **Record:** [`vsfftw_2d_r2c`](../../gauntlet/results/vsfftw_2d_r2c/) (2026-10-11), every plane raced at create in the run's own store

![2D r2c speedup matrix, VectorFFT vs FFTW, every 2^a x 2^b plane up to 4M points](../../src/tools/plots/vectorfft-fftw-2d-r2c.svg)

One cell per plane, the speedup over FFTW in the worse of the two engine orders. VectorFFT is
ahead at 112 of the 146 planes, median 1.11x, geomean 1.12x. FFTW leads in both engine orders at
24 planes, almost all of them thin: a last axis of 4 (512x4 0.74x, 8192x4 0.81x, 256x4 0.86x) or
N1 = 16 with a long N2 (16x2048 0.77x, 16x512 and 16x1024 0.86x); at the other 10 planes behind,
only one of the two orders came out slower. Every output agrees with FFTW's to 5.6e-16; the
control plane (64x64) read 1.21-1.29x.

| Plane size | Cells | Median speedup | At or above parity | Best |
|---|---|---|---|---|
| up to 256 points | 21 | 1.24x | 90% | 1.61x (4x64) |
| 257..4,096 | 34 | 1.11x | 79% | 1.50x (16x64) |
| 4,097..65,536 | 46 | 1.05x | 61% | 1.43x (1024x8) |
| 65,537..4M | 45 | 1.10x | 84% | 2.01x (8192x512) |
| **all, 146 planes** | **146** | **1.11x** | **77%** | 2.01x (8192x512) |

---

## 3D r2c, every power-of-two volume from 4 to 256 per side

> **Contract:** 3D real-to-complex FP64, N1 x N2 x N3 reals in, the N1 x N2 x (N3/2 + 1) CCE bins out, interleaved, out of place, K = 1  
> **Cells:** every volume N1 x N2 x N3 with each side a power of two from 4 to 256 and at most 2^20 points (323 volumes), and five volumes with sides that are not powers of two (27x9x15, 36x20x28, 32x32x27, 64x64x15, 16x64x63): 328 volumes  
> **Record:** [`vsfftw_3d_r2c`](../../gauntlet/results/vsfftw_3d_r2c/) (2026-10-11), every volume's plan replayed from the shipped wisdom

![3D r2c speedup matrices, VectorFFT vs FFTW, one N2 x N3 matrix per N1](../../src/tools/plots/vectorfft-fftw-3d-r2c.svg)

One matrix per N1; each cell is the speedup over FFTW in the worse of the two engine orders, and a
cell slower than FFTW is outlined (the five volumes with odd sides are not on the matrices).
VectorFFT is ahead at 281 of the 328 volumes, median 1.16x, geomean 1.15x. FFTW leads in both
engine orders at 32 volumes: 13 whose last two sides are both 4 or 8 (0.72-0.88x), 15 larger
volumes with N1 = 64 or 128 (0.81-0.98x) and 4 others; at the other 15 volumes behind, only one of
the two orders came out slower. Of the volumes with odd sides, three are ahead (16x64x63 1.47x,
32x32x27 1.41x, 36x20x28 1.07x) and two behind (27x9x15 0.95x, 64x64x15 0.81x). Every output
agrees with FFTW's to 5.7e-16; the control volume (64x64x64) read 1.09-1.19x.

| Volume | Cells | Median speedup | At or above parity | Best |
|---|---|---|---|---|
| up to 4,096 points | 84 | 1.23x | 81% | 1.52x (4x8x32) |
| 4,097..65,536 | 142 | 1.18x | 93% | 1.50x (256x4x64) |
| 65,537..1M | 97 | 1.10x | 80% | 1.51x (256x128x16) |
| **all power-of-two, 323 volumes** | **323** | **1.16x** | **86%** | 1.52x (4x8x32) |
