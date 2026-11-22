# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2c interleaved, natural, out of place, K=4 contiguous_k4_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 8.147..8.588


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -               3632      33390    9.12      4.9  1.6e-16
  2x2x2x2x2x2x53  2            -      -              58397     258136    4.37     13.6  1.3e-15
   2x3x4x5x6x7x8  2            -      -             299033    1589725    5.31     41.3  5.0e-16
   2x4x8x2x4x8x2  2            -      -              74028     645660    8.71     28.8  3.0e-16
   3x3x3x3x3x3x3  3            -      -              29669     232186    7.73     16.4  6.4e-16
   3x5x7x3x5x7x3  3            -      -             316560    1820783    5.71     31.4  6.1e-16
   4x4x4x4x4x4x4  2^2          -      -             126570    1090367    8.53     36.2  2.0e-16
   4x4x4x4x8x8x8  2^2          -      -             909225    4520862    4.96     49.0  6.4e-16
   5x5x5x5x5x5x5  5            -      -             745913    3680593    4.93     34.0  5.0e-16
   6x6x6x6x6x6x6  2.3          -      -            2705750   10334244    3.77     37.4  9.1e-16
   8x8x8x8x8x8x4  2^3          -      -           12359425   44890637    3.62     33.9  1.4e-15
   8x8x8x8x8x8x8  2^3          -      -           25487863   77381124    3.02     34.6  1.8e-15
 10x8x6x4x6x8x10  2.5          -      -           11017138   30042981    2.70     33.1  6.3e-16
  53x2x2x2x2x2x2  53           -      -             116544     842763    6.72      6.8  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   3.02   5.14   8.71    5.27
 ALL                             14      0      0   3.02   5.14   8.71    5.27
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   3.02   5.14   9.12    5.51
 prime column                     4      0      0   4.93   6.22   7.73    6.18
 even column                      2      0      0   2.70   3.24   3.77    3.19
 ALL                             14      0      0   3.02   5.14   8.71    5.27
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   2.70   3.70   4.96    3.74
 4097..65536                      4      0      0   5.31   7.12   8.71    6.89
 1025..4096                       3      0      0   4.37   6.72   7.73    6.10
 <= 256 points                    1      0      0   9.12   9.12   9.12    9.12
 ALL                             14      0      0   3.02   5.14   8.71    5.27
```


worst 10: 10x8x6x4x6x8x10 (- 2.70), 8x8x8x8x8x8x8 (- 3.02), 8x8x8x8x8x8x4 (- 3.62), 6x6x6x6x6x6x6 (- 3.77), 2x2x2x2x2x2x53 (- 4.37), 5x5x5x5x5x5x5 (- 4.93), 4x4x4x4x8x8x8 (- 4.96), 2x3x4x5x6x7x8 (- 5.31), 3x5x7x3x5x7x3 (- 5.71), 53x2x2x2x2x2x2 (- 6.72)
best 5: 2x2x2x2x2x2x2 (- 9.12), 2x4x8x2x4x8x2 (- 8.71), 4x4x4x4x4x4x4 (- 8.53), 3x3x3x3x3x3x3 (- 7.73), 53x2x2x2x2x2x2 (- 6.72)
