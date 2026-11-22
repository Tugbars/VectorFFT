# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2c interleaved, natural, out of place, K=4 contiguous_k4_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 3.344..3.512


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               5713      61777   10.63     16.1  2.2e-16
     2x4x6x8x10x12  2            -      -             294030    1400870    4.74     48.6  5.8e-16
       3x3x3x3x3x3  3            -      -               9715      74677    7.65     14.3  4.7e-16
       3x5x7x7x5x3  3            -      -             110007     668564    6.07     26.9  4.7e-16
       4x4x4x4x4x4  2^2          -      -              29148     254307    8.70     33.7  2.3e-16
      4x4x4x4x4x53  2^2          -      -             874167    1891494    1.96     19.5  1.1e-15
       4x4x4x8x8x8  2^2          -      -             193560     961743    4.97     50.8  3.0e-16
       5x5x5x5x5x5  5            -      -             131968     681990    5.14     33.0  4.4e-16
       6x6x6x6x6x6  2.3          -      -             345580    1516080    4.37     41.9  6.5e-16
       8x8x8x4x4x4  2^3          -      -             253900    1820833    7.17     38.7  3.0e-16
       8x8x8x8x8x8  2^3          -      -            1953725    6841981    3.34     48.3  5.5e-16
       9x9x9x9x9x9  3^2          -      -            5818587   16416706    2.81     34.7  8.3e-16
 10x10x10x10x10x10  2.5          -      -           11969988   27849475    2.33     33.3  5.4e-16
     12x10x8x6x4x2  2^2.3        -      -             576600    4640440    7.98     24.8  5.0e-16
     16x4x4x4x4x16  2^4          -      -             470012    2066575    4.32     44.6  3.9e-16
      53x4x4x4x4x4  53           -      -             729467    3545755    4.84     23.4  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               16      0      0   2.33   4.90   8.70    4.92
 ALL                             16      0      0   2.33   4.90   8.70    4.92
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   1.96   4.86  10.63    5.08
 prime column                     4      0      0   4.84   5.60   7.65    5.83
 even column                      3      0      0   2.33   4.37   7.98    4.33
 odd column                       1      0      0   2.81   2.81   2.81    2.81
 ALL                             16      0      0   2.33   4.90   8.70    4.92
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   4.32   4.90   7.98    4.87
 > 65536 points                   3      0      0   2.33   2.81   3.34    2.80
 257..1024                        2      0      0   7.65   9.14  10.63    9.02
 1025..4096                       1      0      0   8.70   8.70   8.70    8.70
 ALL                             16      0      0   2.33   4.90   8.70    4.92
```


worst 10: 4x4x4x4x4x53 (- 1.96), 10x10x10x10x10x10 (- 2.33), 9x9x9x9x9x9 (- 2.81), 8x8x8x8x8x8 (- 3.34), 16x4x4x4x4x16 (- 4.32), 6x6x6x6x6x6 (- 4.37), 2x4x6x8x10x12 (- 4.74), 53x4x4x4x4x4 (- 4.84), 4x4x4x8x8x8 (- 4.97), 5x5x5x5x5x5 (- 5.14)
best 5: 2x4x2x4x2x4 (- 10.63), 4x4x4x4x4x4 (- 8.70), 12x10x8x6x4x2 (- 7.98), 3x3x3x3x3x3 (- 7.65), 8x8x8x4x4x4 (- 7.17)
