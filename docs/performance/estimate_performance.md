# Estimate planning: performance

`VFFT_FLAG_ESTIMATE` ([`include/vfft.h`](../../include/vfft.h)) plans a 1D complex transform from
formulas: the length, the vector width, and this machine's cache sizes and core count. It measures
nothing at create and reads or writes no wisdom. This page answers two questions. How far are its
plans from the plans the default planner measures? And how do they compare with MKL?

> **Platform:** Intel Core i9-14900KF (P-core, AVX2), DDR5, GCC 15.2, single thread  
> **Contract (both sections):** complex-to-complex FP64, interleaved, natural order, out of place, K = 1, one thread

---

## Estimate against the measured plan: powers of two from 64 to 4,194,304

> **Arms:** the measured plan, replayed from a scratch copy of the shipped wisdom; the estimate
> plan; the measured plan created a second time, as a control  
> **Method:** the three plans in one process, the caller pinned to one P-core at high priority;
> 15 rounds with the arm order rotated, 200 ms apart; each time is the arm's median over the
> rounds; each ratio is the median of the 15 per-round ratios, with their range in brackets  
> **Tool:** [`src/estimate/estimate_race.c`](../../src/estimate/estimate_race.c) (2026-10-22)

| N | Measured (ns) | Estimate (ns) | Estimate ÷ measured | Control ÷ measured |
|---|---:|---:|---|---|
| 64 | 29.6 | 29.8 | 1.004 [0.981..1.017] | 1.000 [0.978..1.006] |
| 128 | 64.2 | 65.3 | 1.017 [1.012..1.022] | 1.001 [0.996..1.008] |
| 256 | 134 | 134 | 1.002 [0.995..1.005] | 1.001 [0.995..1.016] |
| 512 | 336 | 336 | 0.999 [0.991..1.024] | 1.001 [0.991..1.014] |
| 1,024 | 716 | 751 | 1.048 [1.024..1.102] | 1.001 [0.980..1.014] |
| 2,048 | 1,651 | 1,653 | 0.990 [0.959..1.072] | 1.008 [0.963..1.032] |
| 4,096 | 3,604 | 3,682 | 1.009 [0.954..1.077] | 1.012 [0.913..1.084] |
| 8,192 | 8,684 | 8,138 | 0.938 [0.895..0.983] | 0.937 [0.877..0.988] |
| 16,384 | 18,180 | 18,201 | 0.999 [0.963..1.080] | 0.995 [0.933..1.055] |
| 32,768 | 39,705 | 40,719 | 1.020 [0.974..1.054] | 0.993 [0.937..1.019] |
| 65,536 | 95,152 | 94,883 | 0.999 [0.913..1.060] | 1.003 [0.856..1.052] |
| 131,072 | 262,108 | 262,225 | 1.002 [0.913..1.045] | 1.010 [0.908..1.077] |
| 262,144 | 651,470 | 649,942 | 1.011 [0.935..1.061] | 0.999 [0.942..1.109] |
| 524,288 | 1,505,115 | 1,501,828 | 0.998 [0.958..1.045] | 0.994 [0.967..1.028] |
| 1,048,576 | 3,325,014 | 3,379,896 | 0.998 [0.913..1.097] | 0.996 [0.860..1.100] |
| 2,097,152 | 7,952,178 | 8,035,033 | 1.019 [0.960..1.086] | 1.032 [0.934..1.132] |
| 4,194,304 | 17,663,650 | 17,459,750 | 1.004 [0.940..1.071] | 0.995 [0.924..1.069] |

A ratio above 1 means the estimate plan is slower. Over the 17 lengths the estimate plans take
1.004 times the measured plans' time (geometric mean), and 15 of the 17 sit within 2%. Up to
262,144 the two plans are identical at 9 of the 13 lengths. From 524,288 both plans split the
length into the same two factors and run the first factor with the same stages. One length is a real loss: at 1,024
the estimate plan is 4.8% slower. The measured plan there starts with a radix-4 stage (4·8·8·4);
the estimate's rule always starts with radix 8 (8·4·4·8). The 0.938 at 8,192 is not a gain: the
control read 0.937 in the same rounds, so it is the measured arm's noise in that run. 8,388,608 is
left out, because building its measured plan from the shipped wisdom measures at create.

---

## Estimate against MKL: every N from 128 to 4,096

> **Competitor:** Intel oneMKL 2025.3 (sequential)  
> **Record:** [`est_c2c_128_4096_mkl_20261022`](../../gauntlet/results/est_c2c_128_4096_mkl_20261022/)
> (2026-10-22). Every length's plan is the estimate's: its wisdom rows were written by
> [`src/estimate/estimate_store.c`](../../src/estimate/estimate_store.c), the same rows
> `VFFT_FLAG_ESTIMATE` builds its plan from, and the gauntlet's bench replayed them  
> **Method:** one process per length; both engines timed in two arms with the engine order
> flipped; a control cell at 4,096 re-timed every 100 cells (82 readings, 1.076 to 1.084 between
> the 10th and 90th percentile, one reading at 0.65); every speedup is MKL's time over
> VectorFFT's in the **worse** of the two arms

VectorFFT is ahead in both engine orders at 3,500 of the 3,969 lengths, median 1.33x, geomean
1.47x. MKL leads in both orders at 215; at the other 254, only one of the two orders came out
slower. Composites built from primes up to 47 run 1.99x faster than MKL at the median. Composites
with a prime factor of 53 or more, and the primes themselves, both sit at 1.21x. The six powers of
two read 1.03x at the median, from 0.90x at 128 to 1.27x at 2,048. The worst lengths are
3,328 = 2^8 x 13 at 0.59x and 1,098 = 2 x 3^2 x 61 at 0.60x. The roundtrip error stays at or
below 1.8e-15 at every length.

| Lengths | Cells | Median speedup | At or above parity | Best |
|---|---|---|---|---|
| 128..511 | 384 | 1.60x | 97% | 4.50x (N=254) |
| 512..1,024 | 513 | 1.31x | 87% | 4.22x (N=893) |
| 1,025..2,048 | 1,024 | 1.31x | 87% | 3.88x (N=1788) |
| 2,049..4,096 | 2,048 | 1.27x | 88% | 3.96x (N=3827) |
| **all, 128..4,096** | **3,969** | **1.33x** | **88%** | 4.50x (N=254) |

| Family | Cells | Median speedup | At or above parity |
|---|---|---|---|
| Powers of two | 6 | 1.03x | 50% |
| Primes (131..4,093) | 533 | 1.21x | 86% |
| Composites, largest prime factor ≤ 47 | 1,276 | 1.99x | 93% |
| Composites, largest prime factor ≥ 53 | 2,154 | 1.21x | 86% |
