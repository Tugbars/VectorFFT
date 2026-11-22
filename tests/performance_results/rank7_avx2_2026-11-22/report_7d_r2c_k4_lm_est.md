# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D r2c interleaved, natural, out of place, K=4 lane-major_r2c_k4_lm_est  cells: 7 listed, 7 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 34.451..34.757


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x3x4x5x6x7x8  2            lm     -             183475    4236370   23.06     67.2  6.2e-16
   4x4x4x4x4x4x4  2^2          lm     -              83833    2907953   34.03     54.7  0.0e+00
   4x4x4x4x8x8x8  2^2          lm     -             587475   11905481   20.19     75.9  3.1e-16
   6x6x6x6x6x6x6  2.3          lm     -            1466925   24108394   16.21     69.1  8.0e-16
   8x8x8x8x8x8x4  2^3          lm     -            9848725  127599919   12.86     42.6  4.2e-16
   8x8x8x8x8x8x8  2^3          lm     -           18909925  190436538   10.04     46.6  3.7e-16
 10x8x6x4x6x8x10  2.5          lm     -            6987875   78057693   11.13     52.3  6.3e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                               7      0      0  10.04  16.21  34.03   16.75
 ALL                              7      0      0  10.04  16.21  34.03   16.75
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0  10.04  20.19  34.03   18.29
 even column                      2      0      0  11.13  13.67  16.21   13.43
 ALL                              7      0      0  10.04  16.21  34.03   16.75
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   5      0      0  10.04  12.86  20.19   13.63
 4097..65536                      2      0      0  23.06  28.55  34.03   28.01
 ALL                              7      0      0  10.04  16.21  34.03   16.75
```


worst 10: 8x8x8x8x8x8x8 (lm 10.04), 10x8x6x4x6x8x10 (lm 11.13), 8x8x8x8x8x8x4 (lm 12.86), 6x6x6x6x6x6x6 (lm 16.21), 4x4x4x4x8x8x8 (lm 20.19), 2x3x4x5x6x7x8 (lm 23.06), 4x4x4x4x4x4x4 (lm 34.03)
best 5: 4x4x4x4x4x4x4 (lm 34.03), 2x3x4x5x6x7x8 (lm 23.06), 4x4x4x4x8x8x8 (lm 20.19), 6x6x6x6x6x6x6 (lm 16.21), 8x8x8x8x8x8x4 (lm 12.86)
