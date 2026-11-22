# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2c interleaved, natural, out of place, K=1_mt8_est  cells: 17 listed, 17 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 2.927..5.453

threaded plans that ran serial (engaged = 0 at both flips): 3: 2x4x2x4x2x4, 3x3x3x3x3x3, 4x4x4x4x4x4


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               1182      12213   10.03     19.5  1.4e-16
     2x4x6x8x10x12  2            -      -              42402     175816    3.94     84.2  5.5e-16
       3x3x3x3x3x3  3            -      -               2029      10238    4.74     17.1  5.3e-16
       3x5x7x7x5x3  3            -      -              15364      84903    5.32     48.2  5.6e-16
       4x4x4x4x4x4  2^2          -      -               7249      24226    3.28     33.9  2.2e-16
      4x4x4x4x4x53  2^2          -      -              71142     137640    1.63     60.0  1.1e-15
       4x4x4x8x8x8  2^2          -      -              19500      91756    4.51    126.0  2.4e-16
       5x5x5x5x5x5  5            -      -              13558      54759    2.73     80.3  5.1e-16
       6x6x6x6x6x6  2.3          -      -              23205      96672    3.16    155.9  7.6e-16
       8x8x8x4x4x4  2^3          -      -              14515     100846    5.10    169.3  3.5e-16
       8x8x8x8x8x8  2^3          -      -              67800     366562    4.82    348.0  4.9e-16
       9x9x9x9x9x9  3^2          -      -             232463     977843    3.67    217.4  8.2e-16
 10x10x10x10x10x10  2.5          -      -             361850    1255950    3.43    275.4  6.2e-16
     12x10x8x6x4x2  2^2.3        -      -              34993     239553    5.19    102.0  5.8e-16
     16x4x4x4x4x16  2^4          -      -              20523     108378    4.32    255.5  3.3e-16
 16x16x16x16x16x16  2^4          -      -           34453300   40977550    1.07     58.4  4.5e-15
      53x4x4x4x4x4  53           -      -              31436     204643    4.63    135.8  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               17      0      0   1.63   4.32   5.32    3.80
 ALL                             17      0      0   1.63   4.32   5.32    3.80
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      9      0      0   1.07   4.32  10.03    3.63
 prime column                     4      0      0   2.73   4.69   5.32    4.23
 even column                      3      0      0   3.16   3.43   5.19    3.83
 odd column                       1      0      0   3.67   3.67   3.67    3.67
 ALL                             17      0      0   1.63   4.32   5.32    3.80
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   2.73   4.42   5.32    3.84
 > 65536 points                   4      0      0   1.07   3.55   4.82    2.84
 257..1024                        2      0      0   4.74   7.38  10.03    6.89
 1025..4096                       1      0      0   3.28   3.28   3.28    3.28
 ALL                             17      0      0   1.63   4.32   5.32    3.80
```


worst 10: 16x16x16x16x16x16 (- 1.07), 4x4x4x4x4x53 (- 1.63), 5x5x5x5x5x5 (- 2.73), 6x6x6x6x6x6 (- 3.16), 4x4x4x4x4x4 (- 3.28), 10x10x10x10x10x10 (- 3.43), 9x9x9x9x9x9 (- 3.67), 2x4x6x8x10x12 (- 3.94), 16x4x4x4x4x16 (- 4.32), 4x4x4x8x8x8 (- 4.51)
best 5: 2x4x2x4x2x4 (- 10.03), 3x5x7x7x5x3 (- 5.32), 12x10x8x6x4x2 (- 5.19), 8x8x8x4x4x4 (- 5.10), 8x8x8x8x8x8 (- 4.82)
