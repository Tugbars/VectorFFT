# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D r2c interleaved, natural, out of place, K=4 lane-major_r2c_k4_lm_mt8_est  cells: 7 listed, 7 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 27.257..27.829

threaded plans that ran serial (engaged = 0 at both flips): 1: 2x3x4x5x6x7x8


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x3x4x5x6x7x8  2            lm     -             169800    1244983    6.78     72.7  6.2e-16
   4x4x4x4x4x4x4  2^2          lm     -              31613     863628   27.09    145.1  0.0e+00
   4x4x4x4x8x8x8  2^2          lm     -             193500    3289300   16.62    230.3  3.1e-16
   6x6x6x6x6x6x6  2.3          lm     -             318638    6679137   19.92    317.9  8.0e-16
   8x8x8x8x8x8x4  2^3          lm     -            2021500   38566906   18.60    207.5  4.2e-16
   8x8x8x8x8x8x8  2^3          lm     -            5811662   54075712    9.27    151.6  3.7e-16
 10x8x6x4x6x8x10  2.5          lm     -            1443787   22312556   15.11    253.0  6.3e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                               7      0      0   6.78  16.62  27.09   14.84
 ALL                              7      0      0   6.78  16.62  27.09   14.84
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0   6.78  16.62  27.09   13.94
 even column                      2      0      0  15.11  17.52  19.92   17.35
 ALL                              7      0      0   6.78  16.62  27.09   14.84
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   5      0      0   9.27  16.62  19.92   15.39
 4097..65536                      2      0      0   6.78  16.93  27.09   13.55
 ALL                              7      0      0   6.78  16.62  27.09   14.84
```


worst 10: 2x3x4x5x6x7x8 (lm 6.78), 8x8x8x8x8x8x8 (lm 9.27), 10x8x6x4x6x8x10 (lm 15.11), 4x4x4x4x8x8x8 (lm 16.62), 8x8x8x8x8x8x4 (lm 18.60), 6x6x6x6x6x6x6 (lm 19.92), 4x4x4x4x4x4x4 (lm 27.09)
best 5: 4x4x4x4x4x4x4 (lm 27.09), 6x6x6x6x6x6x6 (lm 19.92), 8x8x8x8x8x8x4 (lm 18.60), 4x4x4x4x8x8x8 (lm 16.62), 10x8x6x4x6x8x10 (lm 15.11)
