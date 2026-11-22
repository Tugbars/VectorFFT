# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D r2c interleaved, natural, out of place, K=4 lane-major_r2c_k4_lm_est  cells: 10 listed, 10 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 13.235..13.502


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            lm     -               3209     116631   36.31     28.7  0.0e+00
     2x4x6x8x10x12  2            lm     -             173510    3115985   17.64     82.3  4.8e-16
       4x4x4x4x4x4  2^2          lm     -              18780     618628   32.29     52.3  0.0e+00
       4x4x4x8x8x8  2^2          lm     -             119107    2445310   20.20     82.5  2.5e-16
       6x6x6x6x6x6  2.3          lm     -             201010    3382300   16.52     72.0  7.9e-16
       8x8x8x4x4x4  2^3          lm     -             159873    3883420   24.26     61.5  2.7e-16
       8x8x8x8x8x8  2^3          lm     -            1131025   15075918   13.32     83.4  3.2e-16
 10x10x10x10x10x10  2.5          lm     -            7331837   64951374    8.84     54.4  6.1e-16
     16x4x4x4x4x16  2^4          lm     -             276737    3540381   12.78     75.8  3.6e-16
      53x4x4x4x4x4  53           lm     -             505489    8868989   17.43     33.8  8.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              10      0      0  12.78  17.53  36.31   18.38
 ALL                             10      0      0  12.78  17.53  36.31   18.38
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      7      0      0  12.78  20.20  36.31   20.87
 even column                      2      0      0   8.84  12.68  16.52   12.08
 prime column                     1      0      0  17.43  17.43  17.43   17.43
 ALL                             10      0      0  12.78  17.53  36.31   18.38
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      6      0      0  12.78  17.53  24.26   17.80
 > 65536 points                   2      0      0   8.84  11.08  13.32   10.85
 257..1024                        1      0      0  36.31  36.31  36.31   36.31
 1025..4096                       1      0      0  32.29  32.29  32.29   32.29
 ALL                             10      0      0  12.78  17.53  36.31   18.38
```


worst 10: 10x10x10x10x10x10 (lm 8.84), 16x4x4x4x4x16 (lm 12.78), 8x8x8x8x8x8 (lm 13.32), 6x6x6x6x6x6 (lm 16.52), 53x4x4x4x4x4 (lm 17.43), 2x4x6x8x10x12 (lm 17.64), 4x4x4x8x8x8 (lm 20.20), 8x8x8x4x4x4 (lm 24.26), 4x4x4x4x4x4 (lm 32.29), 2x4x2x4x2x4 (lm 36.31)
best 5: 2x4x2x4x2x4 (lm 36.31), 4x4x4x4x4x4 (lm 32.29), 8x8x8x4x4x4 (lm 24.26), 4x4x4x8x8x8 (lm 20.20), 2x4x6x8x10x12 (lm 17.64)
