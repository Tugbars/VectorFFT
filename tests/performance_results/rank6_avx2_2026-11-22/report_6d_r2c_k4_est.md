# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D r2c interleaved, natural, out of place, K=4 contiguous_r2c_k4_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 12.019..12.139


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               3992     116753   29.00     11.5  0.0e+00
     2x4x6x8x10x12  2            -      -             169920    2436035   14.30     42.0  4.4e-16
       3x3x3x3x3x3  3            -      -               8923     155631   17.33      7.8  5.2e-16
       3x5x7x7x5x3  3            -      -              98833    1512110   15.29     15.0  4.1e-16
       4x4x4x4x4x4  2^2          -      -              19237     579481   30.09     25.6  0.0e+00
      4x4x4x4x4x53  2^2          -      -             686311    2884855    4.13     12.4  5.8e-16
       4x4x4x8x8x8  2^2          -      -             103213    2014873   19.41     47.6  3.2e-16
       5x5x5x5x5x5  5            -      -             128387    1499792   11.55     17.0  4.3e-16
       6x6x6x6x6x6  2.3          -      -             202460    2839680   13.96     35.7  6.0e-16
       8x8x8x4x4x4  2^3          -      -             161480    3568420   22.04     30.4  2.6e-16
       8x8x8x8x8x8  2^3          -      -             911238   10981937   11.95     51.8  3.4e-16
       9x9x9x9x9x9  3^2          -      -            3764363   20409850    5.38     26.9  7.6e-16
 10x10x10x10x10x10  2.5          -      -            6304925   30238456    4.77     31.6  6.4e-16
     12x10x8x6x4x2  2^2.3        -      -             800860    7699150    9.61      8.9  5.4e-16
     16x4x4x4x4x16  2^4          -      -             248025    2534631   10.12     42.3  3.5e-16
      53x4x4x4x4x4  53           -      -             505856    8031211   15.87     16.9  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               16      0      0   4.77  14.13  29.00   12.68
 ALL                             16      0      0   4.77  14.13  29.00   12.68
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   4.13  16.86  30.09   15.07
 prime column                     4      0      0  11.55  15.58  17.33   14.85
 even column                      3      0      0   4.77   9.61  13.96    8.62
 odd column                       1      0      0   5.38   5.38   5.38    5.38
 ALL                             16      0      0   4.77  14.13  29.00   12.68
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   9.61  14.13  22.04   12.54
 > 65536 points                   3      0      0   4.77   5.38  11.95    6.74
 257..1024                        2      0      0  17.33  23.17  29.00   22.42
 1025..4096                       1      0      0  30.09  30.09  30.09   30.09
 ALL                             16      0      0   4.77  14.13  29.00   12.68
```


worst 10: 4x4x4x4x4x53 (- 4.13), 10x10x10x10x10x10 (- 4.77), 9x9x9x9x9x9 (- 5.38), 12x10x8x6x4x2 (- 9.61), 16x4x4x4x4x16 (- 10.12), 5x5x5x5x5x5 (- 11.55), 8x8x8x8x8x8 (- 11.95), 6x6x6x6x6x6 (- 13.96), 2x4x6x8x10x12 (- 14.30), 3x5x7x7x5x3 (- 15.29)
best 5: 4x4x4x4x4x4 (- 30.09), 2x4x2x4x2x4 (- 29.00), 8x8x8x4x4x4 (- 22.04), 4x4x4x8x8x8 (- 19.41), 3x3x3x3x3x3 (- 17.33)
