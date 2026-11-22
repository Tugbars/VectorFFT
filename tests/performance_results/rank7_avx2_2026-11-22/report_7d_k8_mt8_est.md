# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2c interleaved, natural, out of place, K=8 contiguous_k8_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 5.674..14.878

threaded plans that ran serial (engaged = 0 at both flips): 0


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -               2042       9332    3.99     17.6  1.5e-16
  2x2x2x2x2x2x53  2            -      -              21421      86637    2.61     74.3  1.1e-15
   2x3x4x5x6x7x8  2            -      -              81400     556762    3.91    303.1  4.6e-16
   2x4x8x2x4x8x2  2            -      -              21550     252875    6.01    197.7  3.7e-16
   3x3x3x3x3x3x3  3            -      -              12833      84943    3.77     75.6  5.5e-16
   3x5x7x3x5x7x3  3            -      -              82500     800706    4.07    240.8  5.7e-16
   4x4x4x4x4x4x4  2^2          -      -              35693     304970    5.67    257.1  2.2e-16
   4x4x4x4x8x8x8  2^2          -      -             253200    1180869    3.82    352.0  6.3e-16
   5x5x5x5x5x5x5  5            -      -             194663     976225    3.64    260.9  4.5e-16
   6x6x6x6x6x6x6  2.3          -      -            1457888    2900318    1.93    139.0  9.1e-16
   8x8x8x8x8x8x4  2^3          -      -            7844500   21982537    2.52    106.9  1.3e-15
   8x8x8x8x8x8x8  2^3          -      -           18480437   46275968    2.11     95.3  2.0e-15
 10x8x6x4x6x8x10  2.5          -      -            7865675   17261756    2.15     92.9  6.4e-16
  53x2x2x2x2x2x2  53           -      -              32789     251065    5.50     48.5  1.1e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   2.11   3.79   5.67    3.46
 ALL                             14      0      0   2.11   3.79   5.67    3.46
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   2.11   3.86   6.01    3.60
 prime column                     4      0      0   3.64   3.92   5.50    4.19
 even column                      2      0      0   1.93   2.04   2.15    2.04
 ALL                             14      0      0   2.11   3.79   5.67    3.46
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   1.93   2.34   3.82    2.60
 4097..65536                      4      0      0   3.91   4.87   6.01    4.83
 1025..4096                       3      0      0   2.61   3.77   5.50    3.78
 <= 256 points                    1      0      0   3.99   3.99   3.99    3.99
 ALL                             14      0      0   2.11   3.79   5.67    3.46
```


worst 10: 6x6x6x6x6x6x6 (- 1.93), 8x8x8x8x8x8x8 (- 2.11), 10x8x6x4x6x8x10 (- 2.15), 8x8x8x8x8x8x4 (- 2.52), 2x2x2x2x2x2x53 (- 2.61), 5x5x5x5x5x5x5 (- 3.64), 3x3x3x3x3x3x3 (- 3.77), 4x4x4x4x8x8x8 (- 3.82), 2x3x4x5x6x7x8 (- 3.91), 2x2x2x2x2x2x2 (- 3.99)
best 5: 2x4x8x2x4x8x2 (- 6.01), 4x4x4x4x4x4x4 (- 5.67), 53x2x2x2x2x2x2 (- 5.50), 3x5x7x3x5x7x3 (- 4.07), 2x2x2x2x2x2x2 (- 3.99)
