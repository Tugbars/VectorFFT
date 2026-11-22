# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2c interleaved, natural, out of place, K=1_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 5.901..7.145

threaded plans that ran serial (engaged = 0 at both flips): 4: 2x2x2x2x2x2x2, 2x2x2x2x2x2x53, 2x4x8x2x4x8x2, 3x3x3x3x3x3x3


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -                856       6847    7.75      5.2  1.8e-16
  2x2x2x2x2x2x53  2            -      -              14359      38984    2.56     13.9  8.7e-16
   2x3x4x5x6x7x8  2            -      -              44657     220482    4.42     69.1  4.7e-16
   2x4x8x2x4x8x2  2            -      -              17541      92195    4.68     30.4  2.7e-16
   3x3x3x3x3x3x3  3            -      -               6942      26040    3.56     17.5  5.1e-16
   3x5x7x3x5x7x3  3            -      -              34030     200691    5.63     73.0  4.6e-16
   4x4x4x4x4x4x4  2^2          -      -              13193      92467    5.83     86.9  2.6e-16
   4x4x4x4x8x8x8  2^2          -      -              57880     318133    5.05    192.5  7.4e-16
   5x5x5x5x5x5x5  5            -      -              57432     244676    3.76    110.5  4.8e-16
   6x6x6x6x6x6x6  2.3          -      -             109700     514806    3.30    230.9  8.3e-16
   8x8x8x8x8x8x4  2^3          -      -             285425    1326512    3.85    367.4  1.3e-15
   8x8x8x8x8x8x8  2^3          -      -            1015712    2331575    2.03    216.8  2.1e-15
 10x8x6x4x6x8x10  2.5          -      -             374663    1546931    3.57    243.7  5.6e-16
  53x2x2x2x2x2x2  53           -      -               5784      46083    5.80     34.4  1.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   2.56   4.13   5.83    4.17
 ALL                             14      0      0   2.56   4.13   5.83    4.17
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   2.03   4.55   7.75    4.19
 prime column                     4      0      0   3.56   4.70   5.80    4.57
 even column                      2      0      0   3.30   3.44   3.57    3.43
 ALL                             14      0      0   2.56   4.13   5.83    4.17
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   2.03   3.67   5.05    3.47
 4097..65536                      4      0      0   4.42   5.16   5.83    5.10
 1025..4096                       3      0      0   2.56   3.56   5.80    3.75
 <= 256 points                    1      0      0   7.75   7.75   7.75    7.75
 ALL                             14      0      0   2.56   4.13   5.83    4.17
```


worst 10: 8x8x8x8x8x8x8 (- 2.03), 2x2x2x2x2x2x53 (- 2.56), 6x6x6x6x6x6x6 (- 3.30), 3x3x3x3x3x3x3 (- 3.56), 10x8x6x4x6x8x10 (- 3.57), 5x5x5x5x5x5x5 (- 3.76), 8x8x8x8x8x8x4 (- 3.85), 2x3x4x5x6x7x8 (- 4.42), 2x4x8x2x4x8x2 (- 4.68), 4x4x4x4x8x8x8 (- 5.05)
best 5: 2x2x2x2x2x2x2 (- 7.75), 4x4x4x4x4x4x4 (- 5.83), 53x2x2x2x2x2x2 (- 5.80), 3x5x7x3x5x7x3 (- 5.63), 4x4x4x4x8x8x8 (- 5.05)
