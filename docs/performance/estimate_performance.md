# Estimate planning: performance

`VFFT_FLAG_ESTIMATE` ([`include/vfft.h`](../../include/vfft.h)) plans a complex transform in one,
two or three dimensions from formulas: the lengths, the vector width, and this machine's cache
sizes and core count. It measures nothing at create and reads or writes no wisdom. This page
answers two questions. How far are its plans from the plans the default planner measures? And how
do they compare with MKL? Over 58 power-of-two configurations in one, two and three dimensions, an
estimate plan takes, at the median, 1.003 times the time of the measured plan.

> **Platform:** Intel Core i9-14900KF (P-core, AVX2), DDR5, GCC 15.2, single thread  
> **Contract (every section):** complex-to-complex FP64, interleaved, natural order, out of place, K = 1, one thread

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

## Estimate against the measured plan: 2D and 3D, powers of two

> **Arms and method:** as in the 1D section: the measured plan replayed from a scratch copy of the
> shipped wisdom, the estimate plan, and the measured plan created a second time as a control, in
> one process pinned to one P-core at high priority, 15 rounds  
> **Tool:** [`src/estimate/estimate_race_nd.c`](../../src/estimate/estimate_race_nd.c) (2026-10-23, the
> library at commit 1dd3d03b)

| N1 × N2 | Measured (ns) | Estimate (ns) | Estimate ÷ measured | Control ÷ measured |
|---|---:|---:|---|---|
| 64 × 64 | 6,401 | 5,174 | 0.809 [0.791..0.815] | 1.017 [1.007..1.022] |
| 128 × 128 | 25,476 | 28,675 | 1.125 [1.055..1.194] | 1.016 [0.973..1.075] |
| 256 × 256 | 115,138 | 115,647 | 1.005 [0.954..1.095] | 1.012 [0.960..1.103] |
| 512 × 512 | 653,461 | 649,059 | 0.994 [0.939..1.012] | 1.017 [0.978..1.097] |
| 1024 × 1024 | 3,140,313 | 3,179,756 | 1.006 [0.922..1.098] | 1.004 [0.979..1.078] |
| 2048 × 2048 | 15,808,300 | 18,067,000 | 1.139 [1.075..1.238] | 1.001 [0.921..1.079] |
| 16 × 1024 | 17,283 | 17,939 | 1.034 [0.998..1.041] | 0.999 [0.960..1.038] |
| 1024 × 16 | 26,986 | 29,295 | 1.084 [1.048..1.092] | 1.017 [1.012..1.028] |
| 512 × 16 | 13,910 | 13,540 | 0.970 [0.927..1.054] | 0.999 [0.922..1.083] |
| 1024 × 8 | 11,348 | 11,381 | 1.005 [0.985..1.032] | 0.979 [0.958..1.021] |
| 2048 × 4 | 12,097 | 11,994 | 0.991 [0.978..1.011] | 0.997 [0.977..1.020] |
| 4096 × 256 | 3,801,080 | 3,356,700 | 0.882 [0.860..0.943] | 1.019 [0.956..1.051] |
| 32 × 4096 | 279,798 | 314,198 | 1.123 [1.062..1.152] | 1.025 [1.013..1.107] |
| 8 × 8192 | 102,057 | 109,153 | 1.070 [1.022..1.189] | 1.010 [0.956..1.072] |
| 256 × 4096 | 3,092,371 | 3,098,904 | 1.007 [0.929..1.071] | 1.013 [0.938..1.063] |
| 128 × 512 | 121,753 | 129,866 | 1.065 [0.973..1.114] | 1.006 [0.947..1.045] |
| 512 × 128 | 125,570 | 110,472 | 0.881 [0.839..0.919] | 0.926 [0.885..0.961] |
| 64 × 1024 | 120,748 | 126,561 | 1.048 [0.954..1.082] | 1.001 [0.907..1.097] |
| 1024 × 64 | 112,670 | 112,791 | 1.000 [0.878..1.098] | 1.003 [0.884..1.119] |

| N1 × N2 × N3 | Measured (ns) | Estimate (ns) | Estimate ÷ measured | Control ÷ measured |
|---|---:|---:|---|---|
| 16 × 16 × 16 | 5,352 | 5,195 | 0.970 [0.908..0.990] | 0.974 [0.904..0.991] |
| 32 × 32 × 32 | 52,066 | 49,126 | 0.944 [0.916..1.038] | 0.978 [0.902..1.076] |
| 64 × 64 × 64 | 637,449 | 637,709 | 1.001 [0.970..1.060] | 0.987 [0.930..1.048] |
| 128 × 128 × 128 | 7,205,660 | 7,804,140 | 1.082 [0.977..1.104] | 0.993 [0.896..1.068] |
| 8 × 64 × 64 | 60,522 | 56,748 | 0.938 [0.930..0.946] | 1.001 [0.994..1.006] |
| 64 × 8 × 64 | 44,068 | 46,924 | 1.065 [1.006..1.148] | 0.994 [0.928..1.071] |
| 512 × 4 × 4 | 16,328 | 16,912 | 1.036 [1.008..1.062] | 1.002 [0.971..1.027] |
| 1024 × 16 × 16 | 587,887 | 599,067 | 1.018 [0.978..1.128] | 0.996 [0.932..1.065] |
| 4096 × 16 × 16 | 3,615,486 | 3,620,282 | 1.005 [0.963..1.048] | 0.992 [0.966..1.108] |
| 8192 × 4 × 4 | 414,866 | 414,195 | 0.998 [0.981..1.054] | 1.003 [0.986..1.079] |
| 4 × 256 × 256 | 712,215 | 662,006 | 0.930 [0.878..1.033] | 1.012 [0.912..1.133] |
| 32 × 64 × 128 | 729,968 | 635,318 | 0.870 [0.823..0.997] | 0.999 [0.904..1.142] |
| 64 × 512 × 128 | 15,526,800 | 15,029,660 | 0.967 [0.909..1.001] | 1.002 [0.968..1.023] |
| 256 × 128 × 128 | 15,583,080 | 16,172,560 | 1.038 [0.991..1.075] | 0.989 [0.956..1.098] |
| 32 × 1024 × 128 | 16,486,140 | 15,905,460 | 0.966 [0.922..1.048] | 0.981 [0.909..1.076] |
| 512 × 64 × 128 | 18,351,825 | 18,522,550 | 1.009 [0.968..1.060] | 1.003 [0.938..1.055] |
| 1024 × 64 × 64 | 17,859,525 | 16,014,025 | 0.896 [0.796..0.948] | 0.999 [0.843..1.073] |
| 2048 × 32 × 64 | 20,378,200 | 18,856,600 | 0.920 [0.795..0.985] | 0.963 [0.868..1.024] |
| 16 × 256 × 512 | 10,246,657 | 8,911,500 | 0.867 [0.832..1.047] | 0.993 [0.817..1.195] |
| 16 × 256 × 256 | 3,398,481 | 3,487,505 | 1.021 [0.980..1.110] | 1.009 [0.921..1.217] |
| 8 × 512 × 128 | 1,427,166 | 1,415,796 | 0.992 [0.955..1.086] | 1.021 [0.945..1.027] |
| 32 × 256 × 128 | 3,002,046 | 3,665,365 | 1.220 [1.194..1.301] | 0.996 [0.926..1.106] |

A ratio above 1 means the estimate plan is slower. Over the 41 configurations the estimate plans
take 0.997 times the measured plans' time (geometric mean; 2D 1.009, 3D 0.986), 1.005 at the median.
22 sit within 5%, 10 are more than 5% faster and 9 more than 5% slower; the control, the measured
plan created twice, reads between 0.93 and 1.03. In three dimensions the estimate always transforms
each plane with a 2D plan and then the first axis. Where the measured planner chose another
structure the estimate plan is as often faster (1,024 × 64 × 64 at 0.896) as slower
(64 × 8 × 64 at 1.065). The largest losses are 32 × 256 × 128 (1.220), 2,048 × 2,048 (1.139),
128 × 128 (1.125) and 32 × 4,096 (1.123); in the three 2D ones the estimate picks another column
chain than the measured plan. The 0.881 at 512 × 128 is partly the run's noise: its control read
0.926.

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
