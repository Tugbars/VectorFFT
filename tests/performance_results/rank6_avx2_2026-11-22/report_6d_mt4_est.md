# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2c interleaved, natural, out of place, K=1_mt4_est  cells: 17 listed, 17 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 3.692..3.800

threaded plans that ran serial (engaged = 0 at both flips): 2: 2x4x2x4x2x4, 3x3x3x3x3x3


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               1224      12858   10.36     18.8  1.4e-16
     2x4x6x8x10x12  2            -      -              51223     164556    3.02     69.7  5.5e-16
       3x3x3x3x3x3  3            -      -               2043      10112    4.74     17.0  5.3e-16
       3x5x7x7x5x3  3            -      -              17103      81653    4.66     43.3  5.6e-16
       4x4x4x4x4x4  2^2          -      -               5292      23319    4.12     46.4  2.2e-16
      4x4x4x4x4x53  2^2          -      -              63850     126317    1.89     66.8  1.1e-15
       4x4x4x8x8x8  2^2          -      -              30348      87740    2.83     81.0  2.4e-16
       5x5x5x5x5x5  5            -      -              22829      92290    3.01     47.7  5.1e-16
       6x6x6x6x6x6  2.3          -      -              30205     135702    4.38    119.8  7.6e-16
       8x8x8x4x4x4  2^3          -      -              27852     153071    5.34     88.2  3.5e-16
       8x8x8x8x8x8  2^3          -      -             110887     429538    3.86    212.8  4.9e-16
       9x9x9x9x9x9  3^2          -      -             359650    1286087    3.06    140.5  8.2e-16
 10x10x10x10x10x10  2.5          -      -             567150    1898712    2.99    175.7  6.2e-16
     12x10x8x6x4x2  2^2.3        -      -              39888     326853    7.50     89.5  5.8e-16
     16x4x4x4x4x16  2^4          -      -              32887     143420    4.22    159.4  3.3e-16
 16x16x16x16x16x16  2^4          -      -           37284212   43887981    1.15     54.0  4.5e-15
      53x4x4x4x4x4  53           -      -              54986     257736    4.49     77.6  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               17      0      0   1.89   4.12   7.50    3.77
 ALL                             17      0      0   1.89   4.12   7.50    3.77
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      9      0      0   1.15   3.86  10.36    3.45
 prime column                     4      0      0   3.01   4.57   4.74    4.16
 even column                      3      0      0   2.99   4.38   7.50    4.61
 odd column                       1      0      0   3.06   3.06   3.06    3.06
 ALL                             17      0      0   1.89   4.12   7.50    3.77
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   2.83   4.30   7.50    3.87
 > 65536 points                   4      0      0   1.15   3.02   3.86    2.52
 257..1024                        2      0      0   4.74   7.55  10.36    7.01
 1025..4096                       1      0      0   4.12   4.12   4.12    4.12
 ALL                             17      0      0   1.89   4.12   7.50    3.77
```


worst 10: 16x16x16x16x16x16 (- 1.15), 4x4x4x4x4x53 (- 1.89), 4x4x4x8x8x8 (- 2.83), 10x10x10x10x10x10 (- 2.99), 5x5x5x5x5x5 (- 3.01), 2x4x6x8x10x12 (- 3.02), 9x9x9x9x9x9 (- 3.06), 8x8x8x8x8x8 (- 3.86), 4x4x4x4x4x4 (- 4.12), 16x4x4x4x4x16 (- 4.22)
best 5: 2x4x2x4x2x4 (- 10.36), 12x10x8x6x4x2 (- 7.50), 8x8x8x4x4x4 (- 5.34), 3x3x3x3x3x3 (- 4.74), 3x5x7x7x5x3 (- 4.66)
