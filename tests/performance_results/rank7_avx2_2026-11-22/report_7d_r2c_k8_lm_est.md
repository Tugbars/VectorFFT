# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D r2c interleaved, natural, out of place, K=8 lane-major_r2c_k8_lm_est  cells: 7 listed, 7 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 39.237..39.938


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x3x4x5x6x7x8  2            lm     -             372988    8776837   23.36     66.2  4.5e-16
   4x4x4x4x4x4x4  2^2          lm     -             147033    5836466   39.35     62.4  0.0e+00
   4x4x4x4x8x8x8  2^2          lm     -            1232950   25191906   19.83     72.3  3.2e-16
   6x6x6x6x6x6x6  2.3          lm     -            3953912   61804556   14.63     51.2  7.6e-16
   8x8x8x8x8x8x4  2^3          lm     -           21463275  352190356   16.38     39.1  3.1e-16
   8x8x8x8x8x8x8  2^3          lm     -           39648638  571057125   14.14     44.4  4.8e-16
 10x8x6x4x6x8x10  2.5          lm     -           15708562  256281062   15.97     46.5  6.0e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                               7      0      0  14.14  16.38  39.35   19.27
 ALL                              7      0      0  14.14  16.38  39.35   19.27
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0  14.14  19.83  39.35   21.14
 even column                      2      0      0  14.63  15.30  15.97   15.28
 ALL                              7      0      0  14.14  16.38  39.35   19.27
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   5      0      0  14.14  15.97  19.83   16.07
 4097..65536                      2      0      0  23.36  31.35  39.35   30.32
 ALL                              7      0      0  14.14  16.38  39.35   19.27
```


worst 10: 8x8x8x8x8x8x8 (lm 14.14), 6x6x6x6x6x6x6 (lm 14.63), 10x8x6x4x6x8x10 (lm 15.97), 8x8x8x8x8x8x4 (lm 16.38), 4x4x4x4x8x8x8 (lm 19.83), 2x3x4x5x6x7x8 (lm 23.36), 4x4x4x4x4x4x4 (lm 39.35)
best 5: 4x4x4x4x4x4x4 (lm 39.35), 2x3x4x5x6x7x8 (lm 23.36), 4x4x4x4x8x8x8 (lm 19.83), 8x8x8x8x8x8x4 (lm 16.38), 10x8x6x4x6x8x10 (lm 15.97)
