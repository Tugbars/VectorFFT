# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D r2c interleaved, natural, out of place, K=8 lane-major_r2c_k8_lm_mt8_est  cells: 7 listed, 7 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 22.917..25.447

threaded plans that ran serial (engaged = 0 at both flips): 1: 2x3x4x5x6x7x8


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x3x4x5x6x7x8  2            lm     -             367812    1901300    5.11     67.1  4.5e-16
   4x4x4x4x4x4x4  2^2          lm     -              56680    1400186   23.97    161.9  0.0e+00
   4x4x4x4x8x8x8  2^2          lm     -             367750    4954218   12.98    242.4  3.2e-16
   6x6x6x6x6x6x6  2.3          lm     -             712625   12870400   17.55    284.3  7.6e-16
   8x8x8x8x8x8x4  2^3          lm     -            7294638   70878763    9.63    115.0  3.1e-16
   8x8x8x8x8x8x8  2^3          lm     -           16039888   93807750    5.70    109.8  4.8e-16
 10x8x6x4x6x8x10  2.5          lm     -            4909575   45119506    9.13    148.8  6.0e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                               7      0      0   5.11   9.63  23.97   10.49
 ALL                              7      0      0   5.11   9.63  23.97   10.49
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0   5.11   9.63  23.97    9.73
 even column                      2      0      0   9.13  13.34  17.55   12.66
 ALL                              7      0      0   5.11   9.63  23.97   10.49
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   5      0      0   5.70   9.63  17.55   10.27
 4097..65536                      2      0      0   5.11  14.54  23.97   11.07
 ALL                              7      0      0   5.11   9.63  23.97   10.49
```


worst 10: 2x3x4x5x6x7x8 (lm 5.11), 8x8x8x8x8x8x8 (lm 5.70), 10x8x6x4x6x8x10 (lm 9.13), 8x8x8x8x8x8x4 (lm 9.63), 4x4x4x4x8x8x8 (lm 12.98), 6x6x6x6x6x6x6 (lm 17.55), 4x4x4x4x4x4x4 (lm 23.97)
best 5: 4x4x4x4x4x4x4 (lm 23.97), 6x6x6x6x6x6x6 (lm 17.55), 4x4x4x4x8x8x8 (lm 12.98), 8x8x8x8x8x8x4 (lm 9.63), 10x8x6x4x6x8x10 (lm 9.13)
