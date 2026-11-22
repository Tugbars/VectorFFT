# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2c interleaved, natural, out of place, K=1_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 6.308..7.737

threaded plans that ran serial (engaged = 0 at both flips): 4: 2x2x2x2x2x2x2, 2x2x2x2x2x2x53, 2x4x8x2x4x8x2, 3x3x3x3x3x3x3


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -                874       6807    7.62      5.1  1.8e-16
  2x2x2x2x2x2x53  2            -      -              14446      39893    2.74     13.8  8.7e-16
   2x3x4x5x6x7x8  2            -      -              44727     219295    4.36     69.0  4.7e-16
   2x4x8x2x4x8x2  2            -      -              18411      90308    4.87     28.9  2.7e-16
   3x3x3x3x3x3x3  3            -      -               6959      26842    3.59     17.4  5.1e-16
   3x5x7x3x5x7x3  3            -      -              34343     196217    4.49     72.3  4.6e-16
   4x4x4x4x4x4x4  2^2          -      -              13363      87243    5.83     85.8  2.6e-16
   4x4x4x4x8x8x8  2^2          -      -              58233     318023    4.68    191.3  7.4e-16
   5x5x5x5x5x5x5  5            -      -              57924     280416    2.51    109.6  4.8e-16
   6x6x6x6x6x6x6  2.3          -      -             107663     633581    4.60    235.2  8.3e-16
   8x8x8x8x8x8x4  2^3          -      -             286112    1421394    4.47    366.5  1.3e-15
   8x8x8x8x8x8x8  2^3          -      -            1096687    2616687    2.02    200.8  2.1e-15
 10x8x6x4x6x8x10  2.5          -      -             351963    1485525    3.92    259.4  5.6e-16
  53x2x2x2x2x2x2  53           -      -               5952      46828    6.54     33.4  1.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   2.51   4.48   6.54    4.20
 ALL                             14      0      0   2.51   4.48   6.54    4.20
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   2.02   4.58   7.62    4.27
 prime column                     4      0      0   2.51   4.04   6.54    4.03
 even column                      2      0      0   3.92   4.26   4.60    4.25
 ALL                             14      0      0   2.51   4.48   6.54    4.20
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   2.02   4.20   4.68    3.52
 4097..65536                      4      0      0   4.36   4.68   5.83    4.86
 1025..4096                       3      0      0   2.74   3.59   6.54    4.01
 <= 256 points                    1      0      0   7.62   7.62   7.62    7.62
 ALL                             14      0      0   2.51   4.48   6.54    4.20
```


worst 10: 8x8x8x8x8x8x8 (- 2.02), 5x5x5x5x5x5x5 (- 2.51), 2x2x2x2x2x2x53 (- 2.74), 3x3x3x3x3x3x3 (- 3.59), 10x8x6x4x6x8x10 (- 3.92), 2x3x4x5x6x7x8 (- 4.36), 8x8x8x8x8x8x4 (- 4.47), 3x5x7x3x5x7x3 (- 4.49), 6x6x6x6x6x6x6 (- 4.60), 4x4x4x4x8x8x8 (- 4.68)
best 5: 2x2x2x2x2x2x2 (- 7.62), 53x2x2x2x2x2x2 (- 6.54), 4x4x4x4x4x4x4 (- 5.83), 2x4x8x2x4x8x2 (- 4.87), 4x4x4x4x8x8x8 (- 4.68)
