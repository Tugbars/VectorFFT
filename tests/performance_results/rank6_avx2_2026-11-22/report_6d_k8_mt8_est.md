# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2c interleaved, natural, out of place, K=8 contiguous_k8_mt8_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 1.941..2.329

threaded plans that ran serial (engaged = 0 at both flips): 0


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               3135      22631    6.61     58.8  3.0e-16
     2x4x6x8x10x12  2            -      -              80613     477837    3.84    354.2  5.5e-16
       3x3x3x3x3x3  3            -      -               5859      22563    3.27     47.3  4.7e-16
       3x5x7x7x5x3  3            -      -              30491     259997    4.35    194.2  5.0e-16
       4x4x4x4x4x4  2^2          -      -              12395     104109    4.77    158.6  3.3e-16
      4x4x4x4x4x53  2^2          -      -             258112     675294    1.59    132.3  1.1e-15
       4x4x4x8x8x8  2^2          -      -              51412     322843    3.67    382.4  3.4e-16
       5x5x5x5x5x5  5            -      -              37447     257337    4.29    232.5  4.4e-16
       6x6x6x6x6x6  2.3          -      -              90812     572337    3.59    318.7  8.4e-16
       8x8x8x4x4x4  2^3          -      -              67812     624631    5.04    289.9  3.3e-16
       8x8x8x8x8x8  2^3          -      -             983087    2140887    1.89    192.0  5.9e-16
       9x9x9x9x9x9  3^2          -      -            4075700    6901325    1.67     99.2  8.4e-16
 10x10x10x10x10x10  2.5          -      -            8442150   16138831    1.80     94.4  5.5e-16
     12x10x8x6x4x2  2^2.3        -      -             148162    1255144    5.81    192.7  5.7e-16
     16x4x4x4x4x16  2^4          -      -             127200     689081    3.62    329.7  3.0e-16
      53x4x4x4x4x4  53           -      -             195688    1043600    3.43    174.5  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               16      0      0   1.67   3.64   5.81    3.40
 ALL                             16      0      0   1.67   3.64   5.81    3.40
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   1.59   3.75   6.61    3.54
 prime column                     4      0      0   3.27   3.86   4.35    3.80
 even column                      3      0      0   1.80   3.59   5.81    3.35
 odd column                       1      0      0   1.67   1.67   1.67    1.67
 ALL                             16      0      0   1.67   3.64   5.81    3.40
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   3.43   3.75   5.81    3.75
 > 65536 points                   3      0      0   1.67   1.80   1.89    1.79
 257..1024                        2      0      0   3.27   4.94   6.61    4.65
 1025..4096                       1      0      0   4.77   4.77   4.77    4.77
 ALL                             16      0      0   1.67   3.64   5.81    3.40
```


worst 10: 4x4x4x4x4x53 (- 1.59), 9x9x9x9x9x9 (- 1.67), 10x10x10x10x10x10 (- 1.80), 8x8x8x8x8x8 (- 1.89), 3x3x3x3x3x3 (- 3.27), 53x4x4x4x4x4 (- 3.43), 6x6x6x6x6x6 (- 3.59), 16x4x4x4x4x16 (- 3.62), 4x4x4x8x8x8 (- 3.67), 2x4x6x8x10x12 (- 3.84)
best 5: 2x4x2x4x2x4 (- 6.61), 12x10x8x6x4x2 (- 5.81), 8x8x8x4x4x4 (- 5.04), 4x4x4x4x4x4 (- 4.77), 3x5x7x7x5x3 (- 4.35)
