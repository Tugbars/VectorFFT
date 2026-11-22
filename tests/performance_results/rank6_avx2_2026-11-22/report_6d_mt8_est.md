# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2c interleaved, natural, out of place, K=1_mt8_est  cells: 17 listed, 17 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 4.104..5.386

threaded plans that ran serial (engaged = 0 at both flips): 3: 2x4x2x4x2x4, 3x3x3x3x3x3, 4x4x4x4x4x4


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               1193      12249    9.49     19.3  1.4e-16
     2x4x6x8x10x12  2            -      -              42412     176364    3.01     84.2  5.5e-16
       3x3x3x3x3x3  3            -      -               2034       9971    4.62     17.0  5.3e-16
       3x5x7x7x5x3  3            -      -              15097      82090    5.16     49.0  5.6e-16
       4x4x4x4x4x4  2^2          -      -               6868      23415    2.97     35.8  2.2e-16
      4x4x4x4x4x53  2^2          -      -              77408     162300    1.65     55.1  1.1e-15
       4x4x4x8x8x8  2^2          -      -              19959      88921    3.78    123.1  2.4e-16
       5x5x5x5x5x5  5            -      -              13417      57597    4.07     81.1  5.1e-16
       6x6x6x6x6x6  2.3          -      -              23998     112495    3.90    150.8  7.6e-16
       8x8x8x4x4x4  2^3          -      -              15030      89895    4.69    163.5  3.5e-16
       8x8x8x8x8x8  2^3          -      -              66325     360693    4.76    355.7  4.9e-16
       9x9x9x9x9x9  3^2          -      -             234688     980431    3.09    215.3  8.2e-16
 10x10x10x10x10x10  2.5          -      -             359300    1441556    3.45    277.4  6.2e-16
     12x10x8x6x4x2  2^2.3        -      -              35605     240653    6.19    100.2  5.8e-16
     16x4x4x4x4x16  2^4          -      -              20390      99515    4.43    257.1  3.3e-16
 16x16x16x16x16x16  2^4          -      -           33860637   40375275    1.08     59.5  4.5e-15
      53x4x4x4x4x4  53           -      -              31108     187188    5.03    137.2  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               17      0      0   1.65   4.07   6.19    3.80
 ALL                             17      0      0   1.65   4.07   6.19    3.80
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      9      0      0   1.08   3.78   9.49    3.37
 prime column                     4      0      0   4.07   4.83   5.16    4.70
 even column                      3      0      0   3.45   3.90   6.19    4.37
 odd column                       1      0      0   3.09   3.09   3.09    3.09
 ALL                             17      0      0   1.65   4.07   6.19    3.80
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   3.01   4.25   6.19    3.98
 > 65536 points                   4      0      0   1.08   3.27   4.76    2.72
 257..1024                        2      0      0   4.62   7.06   9.49    6.62
 1025..4096                       1      0      0   2.97   2.97   2.97    2.97
 ALL                             17      0      0   1.65   4.07   6.19    3.80
```


worst 10: 16x16x16x16x16x16 (- 1.08), 4x4x4x4x4x53 (- 1.65), 4x4x4x4x4x4 (- 2.97), 2x4x6x8x10x12 (- 3.01), 9x9x9x9x9x9 (- 3.09), 10x10x10x10x10x10 (- 3.45), 4x4x4x8x8x8 (- 3.78), 6x6x6x6x6x6 (- 3.90), 5x5x5x5x5x5 (- 4.07), 16x4x4x4x4x16 (- 4.43)
best 5: 2x4x2x4x2x4 (- 9.49), 12x10x8x6x4x2 (- 6.19), 3x5x7x7x5x3 (- 5.16), 53x4x4x4x4x4 (- 5.03), 8x8x8x8x8x8 (- 4.76)
