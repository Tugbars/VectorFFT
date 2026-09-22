<img width="1568" height="649" alt="preview (1)" src="https://github.com/user-attachments/assets/511964c1-6402-43db-927b-98f0be4f81c9" />

---

## Benchmark Results

> **Full performance record** — every gauntlet run, multi-threaded scaling, the
> 2D/3D tiers, accuracy and hardware caveats — is
> [`docs/performance/v1_0_results.md`](docs/performance/v1_0_results.md).
> The runs themselves (csv, calibration logs, banked wisdom) are in
> [`gauntlet/results/`](gauntlet/results/), and [`gauntlet/`](gauntlet/) is the
> tool that reproduces them on your own machine.

> **Platform:** Intel Core i9-14900KF (P-core, AVX2), DDR5, GCC 15.2, single thread  
> **Competitor:** Intel oneMKL 2025.3 (sequential, `mkl_set_num_threads(1)`)  
> **Contract:** 1D complex-to-complex FP64, interleaved, natural order, out of place, K = 1  
> **Cells:** every length N from 2 to 4,096 — 4,095 transforms, no size skipped

### 1D throughput, every N from 2 to 4,096 — VectorFFT vs Intel MKL

![Throughput, VectorFFT vs MKL, every N from 2 to 4096](src/tools/plots/vectorfft-gflops.svg)

One line per engine, one point per length, joined in N. Each point is the median
of two timing arms with the engine order flipped; every ratio below takes the
**worse** of the two arms, so a win is never an artefact of order. The saw-tooth is
the arithmetic of the lengths themselves: powers of two and smooth composites reach
60-77 GFLOPS, primes and rough composites run at 5-10 GFLOPS on both engines, and on
those VectorFFT sits above MKL by a steady offset (log scale: equal speed ratios are
equal vertical gaps).

| Lengths | Cells | Median speedup | At or above parity | Best |
|---|---|---|---|---|
| 2..16 | 15 | 1.39x | 87% | 2.83x (N=2) |
| 17..64 | 48 | 1.53x | 94% | 4.18x (N=61) |
| 65..256 | 192 | 1.53x | 97% | 6.88x (N=89) |
| 257..1,024 | 768 | 1.28x | 90% | 4.02x (N=508) |
| 1,025..2,048 | 1,024 | 1.20x | 88% | 3.74x (N=1946) |
| 2,049..4,096 | 2,048 | 1.16x | 84% | 3.73x (N=3827) |
| **all, 2..4,096** | **4,095** | **1.23x** | **87%** | 6.88x (N=89) |

| Family | Cells | Median speedup | At or above parity |
|---|---|---|---|
| Powers of two | 12 | 1.13x | 75% |
| Primes | 563 | 1.17x | 87% |
| Other composites | 3,520 | 1.24x | 87% |

## Accuracy

![Precision, VectorFFT vs MKL, every N from 2 to 4096](src/tools/plots/vectorfft-precision.svg)

The forward transform of every length N from 2 to 4,096, VectorFFT and MKL on the
same random input, against a scalar DFT accumulated in 80-bit long double
(the gauntlet's `verify` verb, [`gauntlet/k1_fwd_ref_probe.c`](gauntlet/k1_fwd_ref_probe.c);
the records are in [`gauntlet/results/`](gauntlet/results/)). The error is the
relative L2 norm ||y - X|| / ||X||, in units of 1e-16 (FP64 epsilon = 2.2).

Both libraries deliver 14 to 15 correct digits at every length, and MKL is the
tighter of the two: its error stays within 3 epsilon everywhere, VectorFFT's median
is 1.5x MKL's and its worst lengths reach 13 epsilon. On the powers of two, which
run on the ZTURN-T kernels, the two are level; the gap is on the mixed-radix and prime
lengths, which run on the stage kernels, so that is where the twiddle path will be
looked at. It is measured here so it can be worked on, not hidden.

| Lengths | Cells | VectorFFT median | MKL median | VectorFFT max | MKL max |
|---|---|---|---|---|---|
| 2..16 | 15 | 1.67 | 1.15 | 6.74 | 1.52 |
| 17..64 | 48 | 5.32 | 1.97 | 18.00 | 2.79 |
| 65..256 | 192 | 6.67 | 2.50 | 25.07 | 5.24 |
| 257..1,024 | 768 | 6.95 | 3.14 | 25.66 | 5.22 |
| 1,025..2,048 | 1,024 | 6.01 | 3.99 | 28.94 | 5.45 |
| 2,049..4,096 | 2,048 | 6.09 | 4.62 | 27.46 | 6.07 |
| **all, 2..4,096** | **4,095** | **6.11** | **4.04** | **28.94** | **6.07** |

| Family | Cells | VectorFFT median | MKL median | VectorFFT max | MKL max |
|---|---|---|---|---|---|
| Powers of two | 12 | 1.80 | 1.45 | 2.73 | 2.36 |
| Primes | 564 | 6.15 | 4.64 | 28.94 | 5.58 |
| Other composites | 3,520 | 6.11 | 3.56 | 27.46 | 6.07 |

Elementwise maximum error, relative to the largest output, same reference: VectorFFT
median 8.3, max 41.4; MKL median 4.2, max 8.5. Roundtrip, backward of the forward
divided by N against the input, elementwise maximum: VectorFFT median 15.6, max 103;
MKL median 12.4, max 24.7 (all in units of 1e-16).

---

## Acknowledgments

- [FFTW](http://www.fftw.org/) by Matteo Frigo and Steven G. Johnson -- the gold standard for decades. VectorFFT's prime-radix butterflies (R=11, 13, 17, 19) are derived from FFTW's genfft algebraic output, then re-scheduled using Sethi-Ullman register allocation with explicit spill management to minimize register pressure on AVX2 (16 YMM) and AVX-512 (32 ZMM).
- [VkFFT](https://github.com/DTolm/VkFFT) by Dmitrii Tolmachev -- inspiration for the benchmarking methodology and presentation style.
