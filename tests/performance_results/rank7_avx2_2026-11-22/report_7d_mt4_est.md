# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2c interleaved, natural, out of place, K=1_mt4_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 4.979..6.453

threaded plans that ran serial (engaged = 0 at both flips): 3: 2x2x2x2x2x2x2, 2x2x2x2x2x2x53, 3x3x3x3x3x3x3


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -                869       6793    7.46      5.2  1.8e-16
  2x2x2x2x2x2x53  2            -      -              13567      38851    2.69     14.7  8.7e-16
   2x3x4x5x6x7x8  2            -      -              53712     195955    3.47     57.4  4.7e-16
   2x4x8x2x4x8x2  2            -      -              14273      89939    5.93     37.3  2.7e-16
   3x3x3x3x3x3x3  3            -      -               6821      26090    3.59     17.8  5.1e-16
   3x5x7x3x5x7x3  3            -      -              40060     176187    3.36     62.0  4.6e-16
   4x4x4x4x4x4x4  2^2          -      -              14397      86560    5.03     79.7  2.6e-16
   4x4x4x4x8x8x8  2^2          -      -              57293     291263    4.85    194.5  7.4e-16
   5x5x5x5x5x5x5  5            -      -              72236     391470    4.87     87.9  4.8e-16
   6x6x6x6x6x6x6  2.3          -      -             193950     880344    4.38    130.6  8.3e-16
   8x8x8x8x8x8x4  2^3          -      -             567763    2595381    4.30    184.7  1.3e-15
   8x8x8x8x8x8x8  2^3          -      -            1666000    4788012    2.78    132.2  2.1e-15
 10x8x6x4x6x8x10  2.5          -      -             583250    2088724    3.52    156.5  5.6e-16
  53x2x2x2x2x2x2  53           -      -               9624      69154    6.74     20.7  1.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   2.78   4.34   6.74    4.30
 ALL                             14      0      0   2.78   4.34   6.74    4.30
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   2.69   4.57   7.46    4.32
 prime column                     4      0      0   3.36   4.23   6.74    4.46
 even column                      2      0      0   3.52   3.95   4.38    3.93
 ALL                             14      0      0   2.78   4.34   6.74    4.30
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   2.78   4.34   4.87    4.04
 4097..65536                      4      0      0   3.36   4.25   5.93    4.32
 1025..4096                       3      0      0   2.69   3.59   6.74    4.02
 <= 256 points                    1      0      0   7.46   7.46   7.46    7.46
 ALL                             14      0      0   2.78   4.34   6.74    4.30
```


worst 10: 2x2x2x2x2x2x53 (- 2.69), 8x8x8x8x8x8x8 (- 2.78), 3x5x7x3x5x7x3 (- 3.36), 2x3x4x5x6x7x8 (- 3.47), 10x8x6x4x6x8x10 (- 3.52), 3x3x3x3x3x3x3 (- 3.59), 8x8x8x8x8x8x4 (- 4.30), 6x6x6x6x6x6x6 (- 4.38), 4x4x4x4x8x8x8 (- 4.85), 5x5x5x5x5x5x5 (- 4.87)
best 5: 2x2x2x2x2x2x2 (- 7.46), 53x2x2x2x2x2x2 (- 6.74), 2x4x8x2x4x8x2 (- 5.93), 4x4x4x4x4x4x4 (- 5.03), 5x5x5x5x5x5x5 (- 4.87)
