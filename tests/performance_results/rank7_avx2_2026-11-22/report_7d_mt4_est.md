# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2c interleaved, natural, out of place, K=1_mt4_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 4.903..6.508

threaded plans that ran serial (engaged = 0 at both flips): 3: 2x2x2x2x2x2x2, 2x2x2x2x2x2x53, 3x3x3x3x3x3x3


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -                862       6890    7.62      5.2  1.8e-16
  2x2x2x2x2x2x53  2            -      -              13939      39636    2.55     14.3  8.7e-16
   2x3x4x5x6x7x8  2            -      -              53794     198670    3.62     57.3  4.7e-16
   2x4x8x2x4x8x2  2            -      -              14248      92707    4.85     37.4  2.7e-16
   3x3x3x3x3x3x3  3            -      -               6922      26665    3.58     17.5  5.1e-16
   3x5x7x3x5x7x3  3            -      -              40053     173941    4.26     62.0  4.6e-16
   4x4x4x4x4x4x4  2^2          -      -              14908      93061    5.00     76.9  2.6e-16
   4x4x4x4x8x8x8  2^2          -      -              58173     297940    4.91    191.5  7.4e-16
   5x5x5x5x5x5x5  5            -      -              72072     387798    5.38     88.1  4.8e-16
   6x6x6x6x6x6x6  2.3          -      -             194588     868287    4.12    130.2  8.3e-16
   8x8x8x8x8x8x4  2^3          -      -             579437    2677925    4.06    181.0  1.3e-15
   8x8x8x8x8x8x8  2^3          -      -            1640313    4726087    2.80    134.2  2.1e-15
 10x8x6x4x6x8x10  2.5          -      -             580662    2177825    3.55    157.2  5.6e-16
  53x2x2x2x2x2x2  53           -      -               9784      70639    7.10     20.3  1.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   2.80   4.19   7.10    4.33
 ALL                             14      0      0   2.80   4.19   7.10    4.33
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   2.55   4.45   7.62    4.19
 prime column                     4      0      0   3.58   4.82   7.10    4.91
 even column                      2      0      0   3.55   3.83   4.12    3.82
 ALL                             14      0      0   2.80   4.19   7.10    4.33
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   2.80   4.09   5.38    4.05
 4097..65536                      4      0      0   3.62   4.55   5.00    4.39
 1025..4096                       3      0      0   2.55   3.58   7.10    4.02
 <= 256 points                    1      0      0   7.62   7.62   7.62    7.62
 ALL                             14      0      0   2.80   4.19   7.10    4.33
```


worst 10: 2x2x2x2x2x2x53 (- 2.55), 8x8x8x8x8x8x8 (- 2.80), 10x8x6x4x6x8x10 (- 3.55), 3x3x3x3x3x3x3 (- 3.58), 2x3x4x5x6x7x8 (- 3.62), 8x8x8x8x8x8x4 (- 4.06), 6x6x6x6x6x6x6 (- 4.12), 3x5x7x3x5x7x3 (- 4.26), 2x4x8x2x4x8x2 (- 4.85), 4x4x4x4x8x8x8 (- 4.91)
best 5: 2x2x2x2x2x2x2 (- 7.62), 53x2x2x2x2x2x2 (- 7.10), 5x5x5x5x5x5x5 (- 5.38), 4x4x4x4x4x4x4 (- 5.00), 4x4x4x4x8x8x8 (- 4.91)
