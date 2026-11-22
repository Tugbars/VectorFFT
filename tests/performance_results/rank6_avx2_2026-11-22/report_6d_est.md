# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2c interleaved, natural, out of place, K=1_est  cells: 17 listed, 17 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 3.529..3.649


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               1262      15474   12.15     18.3  1.4e-16
     2x4x6x8x10x12  2            -      -              68707     346439    5.04     52.0  5.5e-16
       3x3x3x3x3x3  3            -      -               2141      18702    8.59     16.2  5.3e-16
       3x5x7x7x5x3  3            -      -              26949     167298    6.20     27.5  5.6e-16
       4x4x4x4x4x4  2^2          -      -               7421      63656    8.56     33.1  2.2e-16
      4x4x4x4x4x53  2^2          -      -             239433     467145    1.85     17.8  1.1e-15
       4x4x4x8x8x8  2^2          -      -              45084     239253    5.21     54.5  2.4e-16
       5x5x5x5x5x5  5            -      -              32858     170155    5.11     33.1  5.1e-16
       6x6x6x6x6x6  2.3          -      -              79800     374800    4.66     45.3  7.6e-16
       8x8x8x4x4x4  2^3          -      -              66257     454991    6.85     37.1  3.5e-16
       8x8x8x8x8x8  2^3          -      -             437937    1614169    3.66     53.9  4.9e-16
       9x9x9x9x9x9  3^2          -      -            1115175    3664881    3.28     45.3  8.2e-16
 10x10x10x10x10x10  2.5          -      -            2115663    6190868    2.92     47.1  6.2e-16
     12x10x8x6x4x2  2^2.3        -      -             145781    1157578    7.89     24.5  5.8e-16
     16x4x4x4x4x16  2^4          -      -             111683     515496    4.57     46.9  3.3e-16
 16x16x16x16x16x16  2^4          -      -           79608475  103138468    1.29     25.3  4.5e-15
      53x4x4x4x4x4  53           -      -             172528     888262    5.14     24.7  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               17      0      0   1.85   5.11   8.59    4.79
 ALL                             17      0      0   1.85   5.11   8.59    4.79
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      9      0      0   1.29   5.04  12.15    4.49
 prime column                     4      0      0   5.11   5.67   8.59    6.12
 even column                      3      0      0   2.92   4.66   7.89    4.75
 odd column                       1      0      0   3.28   3.28   3.28    3.28
 ALL                             17      0      0   1.85   5.11   8.59    4.79
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   4.57   5.13   7.89    4.96
 > 65536 points                   4      0      0   1.29   3.10   3.66    2.59
 257..1024                        2      0      0   8.59  10.37  12.15   10.22
 1025..4096                       1      0      0   8.56   8.56   8.56    8.57
 ALL                             17      0      0   1.85   5.11   8.59    4.79
```


worst 10: 16x16x16x16x16x16 (- 1.29), 4x4x4x4x4x53 (- 1.85), 10x10x10x10x10x10 (- 2.92), 9x9x9x9x9x9 (- 3.28), 8x8x8x8x8x8 (- 3.66), 16x4x4x4x4x16 (- 4.57), 6x6x6x6x6x6 (- 4.66), 2x4x6x8x10x12 (- 5.04), 5x5x5x5x5x5 (- 5.11), 53x4x4x4x4x4 (- 5.14)
best 5: 2x4x2x4x2x4 (- 12.15), 3x3x3x3x3x3 (- 8.59), 4x4x4x4x4x4 (- 8.56), 12x10x8x6x4x2 (- 7.89), 8x8x8x4x4x4 (- 6.85)
