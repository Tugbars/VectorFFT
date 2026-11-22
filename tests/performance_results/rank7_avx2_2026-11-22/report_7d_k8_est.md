# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2c interleaved, natural, out of place, K=8 contiguous_k8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 7.752..8.172


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -               7415      66637    8.96      4.8  1.5e-16
  2x2x2x2x2x2x53  2            -      -             116874     516367    4.13     13.6  1.1e-15
   2x3x4x5x6x7x8  2            -      -             599275    3176437    5.28     41.2  4.6e-16
   2x4x8x2x4x8x2  2            -      -             149380    1291486    8.64     28.5  3.7e-16
   3x3x3x3x3x3x3  3            -      -              59375     464401    7.82     16.3  5.5e-16
   3x5x7x3x5x7x3  3            -      -             636575    3641606    5.60     31.2  5.7e-16
   4x4x4x4x4x4x4  2^2          -      -             268547    2182777    8.12     34.2  2.2e-16
   4x4x4x4x8x8x8  2^2          -      -            2031425    9396700    4.60     43.9  6.3e-16
   5x5x5x5x5x5x5  5            -      -            1503675    7372237    4.88     33.8  4.5e-16
   6x6x6x6x6x6x6  2.3          -      -            6552375   21319381    3.25     30.9  9.1e-16
   8x8x8x8x8x8x4  2^3          -      -           25014063   90046643    3.37     33.5  1.3e-15
   8x8x8x8x8x8x8  2^3          -      -           50993837  155338593    3.04     34.5  2.0e-15
 10x8x6x4x6x8x10  2.5          -      -           22601400   60065819    2.65     32.3  6.4e-16
  53x2x2x2x2x2x2  53           -      -             233500    1686486    7.19      6.8  1.1e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   3.04   5.08   8.64    5.12
 ALL                             14      0      0   3.04   5.08   8.64    5.12
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   3.04   4.94   8.96    5.33
 prime column                     4      0      0   4.88   6.39   7.82    6.26
 even column                      2      0      0   2.65   2.95   3.25    2.94
 ALL                             14      0      0   3.04   5.08   8.64    5.12
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   2.65   3.31   4.88    3.55
 4097..65536                      4      0      0   5.28   6.86   8.64    6.75
 1025..4096                       3      0      0   4.13   7.19   7.82    6.14
 <= 256 points                    1      0      0   8.96   8.96   8.96    8.96
 ALL                             14      0      0   3.04   5.08   8.64    5.12
```


worst 10: 10x8x6x4x6x8x10 (- 2.65), 8x8x8x8x8x8x8 (- 3.04), 6x6x6x6x6x6x6 (- 3.25), 8x8x8x8x8x8x4 (- 3.37), 2x2x2x2x2x2x53 (- 4.13), 4x4x4x4x8x8x8 (- 4.60), 5x5x5x5x5x5x5 (- 4.88), 2x3x4x5x6x7x8 (- 5.28), 3x5x7x3x5x7x3 (- 5.60), 53x2x2x2x2x2x2 (- 7.19)
best 5: 2x2x2x2x2x2x2 (- 8.96), 2x4x8x2x4x8x2 (- 8.64), 4x4x4x4x4x4x4 (- 8.12), 3x3x3x3x3x3x3 (- 7.82), 53x2x2x2x2x2x2 (- 7.19)
