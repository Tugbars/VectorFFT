# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2c interleaved, natural, out of place, K=4 contiguous_k4_mt8_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 3.012..4.168

threaded plans that ran serial (engaged = 0 at both flips): 0


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               2585      17896    5.72     35.7  2.2e-16
     2x4x6x8x10x12  2            -      -              41550     699010   13.65    343.6  5.8e-16
       3x3x3x3x3x3  3            -      -               4704      20497    3.49     29.5  4.7e-16
       3x5x7x7x5x3  3            -      -              31727     296981    7.92     93.3  4.7e-16
       4x4x4x4x4x4  2^2          -      -               8164      79305    7.34    120.4  2.3e-16
      4x4x4x4x4x53  2^2          -      -             128156     549539    3.71    133.2  1.1e-15
       4x4x4x8x8x8  2^2          -      -              26580     299426    8.47    369.8  3.0e-16
       5x5x5x5x5x5  5            -      -              35945     216421    4.38    121.1  4.4e-16
       6x6x6x6x6x6  2.3          -      -              45820     353585    4.76    315.9  6.5e-16
       8x8x8x4x4x4  2^3          -      -              34767     310980    5.94    282.8  3.0e-16
       8x8x8x8x8x8  2^3          -      -             244575     897118    3.11    385.9  5.5e-16
       9x9x9x9x9x9  3^2          -      -            1384187    3946769    2.74    146.0  8.3e-16
 10x10x10x10x10x10  2.5          -      -            3755675    7073581    1.85    106.1  5.4e-16
     12x10x8x6x4x2  2^2.3        -      -              76330     937965    8.41    187.0  5.0e-16
     16x4x4x4x4x16  2^4          -      -              58875     426244    5.35    356.2  3.9e-16
      53x4x4x4x4x4  53           -      -             132867     773161    3.87    128.5  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               16      0      0   2.74   5.06   8.47    5.03
 ALL                             16      0      0   2.74   5.06   8.47    5.03
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   3.11   5.83  13.65    6.04
 prime column                     4      0      0   3.49   4.12   7.92    4.65
 even column                      3      0      0   1.85   4.76   8.41    4.20
 odd column                       1      0      0   2.74   2.74   2.74    2.74
 ALL                             16      0      0   2.74   5.06   8.47    5.03
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   3.87   5.65  13.65    6.12
 > 65536 points                   3      0      0   1.85   2.74   3.11    2.50
 257..1024                        2      0      0   3.49   4.61   5.72    4.47
 1025..4096                       1      0      0   7.34   7.34   7.34    7.34
 ALL                             16      0      0   2.74   5.06   8.47    5.03
```


worst 10: 10x10x10x10x10x10 (- 1.85), 9x9x9x9x9x9 (- 2.74), 8x8x8x8x8x8 (- 3.11), 3x3x3x3x3x3 (- 3.49), 4x4x4x4x4x53 (- 3.71), 53x4x4x4x4x4 (- 3.87), 5x5x5x5x5x5 (- 4.38), 6x6x6x6x6x6 (- 4.76), 16x4x4x4x4x16 (- 5.35), 2x4x2x4x2x4 (- 5.72)
best 5: 2x4x6x8x10x12 (- 13.65), 4x4x4x8x8x8 (- 8.47), 12x10x8x6x4x2 (- 8.41), 3x5x7x7x5x3 (- 7.92), 4x4x4x4x4x4 (- 7.34)
