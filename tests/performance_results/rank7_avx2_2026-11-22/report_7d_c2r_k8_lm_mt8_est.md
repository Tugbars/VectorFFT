# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2r interleaved, natural, out of place, K=8 lane-major_c2r_k8_lm_mt8_est  cells: 7 listed, 7 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 10.057..11.945

threaded plans that ran serial (engaged = 0 at both flips): 1: 2x3x4x5x6x7x8


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x3x4x5x6x7x8  2            lm     -             591250    1139569    1.87     41.7  1.2e-15
   4x4x4x4x4x4x4  2^2          lm     -              65687     769840   10.33    139.7  5.6e-16
   4x4x4x4x8x8x8  2^2          lm     -             504987    3124337    5.73    176.5  8.9e-16
   6x6x6x6x6x6x6  2.3          lm     -            1949762    7119668    3.59    103.9  2.0e-15
   8x8x8x8x8x8x4  2^3          lm     -           10249488   35215031    2.95     81.8  1.0e-15
   8x8x8x8x8x8x8  2^3          lm     -           19427225   49389956    2.30     90.7  1.0e-15
 10x8x6x4x6x8x10  2.5          lm     -            7573075   19355912    2.10     96.4  1.6e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                               7      0      0   1.87   2.95  10.33    3.44
 ALL                              7      0      0   1.87   2.95  10.33    3.44
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0   1.87   2.95  10.33    3.76
 even column                      2      0      0   2.10   2.84   3.59    2.74
 ALL                              7      0      0   1.87   2.95  10.33    3.44
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   5      0      0   2.10   2.95   5.73    3.11
 4097..65536                      2      0      0   1.87   6.10  10.33    4.39
 ALL                              7      0      0   1.87   2.95  10.33    3.44
```


worst 10: 2x3x4x5x6x7x8 (lm 1.87), 10x8x6x4x6x8x10 (lm 2.10), 8x8x8x8x8x8x8 (lm 2.30), 8x8x8x8x8x8x4 (lm 2.95), 6x6x6x6x6x6x6 (lm 3.59), 4x4x4x4x8x8x8 (lm 5.73), 4x4x4x4x4x4x4 (lm 10.33)
best 5: 4x4x4x4x4x4x4 (lm 10.33), 4x4x4x4x8x8x8 (lm 5.73), 6x6x6x6x6x6x6 (lm 3.59), 8x8x8x8x8x8x4 (lm 2.95), 8x8x8x8x8x8x8 (lm 2.30)
