# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D r2c interleaved, natural, out of place, K=8 contiguous_r2c_k8_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 9.969..10.886


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               8262     233679   26.23     11.2  0.0e+00
     2x4x6x8x10x12  2            -      -             341788    4884406   14.11     41.8  5.6e-16
       3x3x3x3x3x3  3            -      -              17231     311788   18.08      8.0  4.8e-16
       3x5x7x7x5x3  3            -      -             197995    3029468   15.23     15.0  4.6e-16
       4x4x4x4x4x4  2^2          -      -              38793    1161387   29.89     25.3  0.0e+00
      4x4x4x4x4x53  2^2          -      -            1411962    5785037    4.04     12.1  6.3e-16
       4x4x4x8x8x8  2^2          -      -             206525    4036025   19.53     47.6  3.4e-16
       5x5x5x5x5x5  5            -      -             255580    3006463   11.69     17.0  4.2e-16
       6x6x6x6x6x6  2.3          -      -             404663    5701887   14.01     35.8  6.8e-16
       8x8x8x4x4x4  2^3          -      -             323713    7146424   21.90     30.4  2.3e-16
       8x8x8x8x8x8  2^3          -      -            2038575   22288456   10.73     46.3  3.2e-16
       9x9x9x9x9x9  3^2          -      -            8129150   41541456    5.11     24.9  7.7e-16
 10x10x10x10x10x10  2.5          -      -           13872450   60982425    4.37     28.7  5.8e-16
     12x10x8x6x4x2  2^2.3        -      -            1591700   15517712    9.72      9.0  5.7e-16
     16x4x4x4x4x16  2^4          -      -             499063    5086169   10.13     42.0  3.3e-16
      53x4x4x4x4x4  53           -      -            1016737   16105231   15.69     16.8  1.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               16      0      0   4.37  14.06  26.23   12.41
 ALL                             16      0      0   4.37  14.06  26.23   12.41
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   4.04  16.82  29.89   14.61
 prime column                     4      0      0  11.69  15.46  18.08   14.99
 even column                      3      0      0   4.37   9.72  14.01    8.41
 odd column                       1      0      0   5.11   5.11   5.11    5.11
 ALL                             16      0      0   4.37  14.06  26.23   12.41
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   9.72  14.06  21.90   12.51
 > 65536 points                   3      0      0   4.37   5.11  10.73    6.21
 257..1024                        2      0      0  18.08  22.16  26.23   21.78
 1025..4096                       1      0      0  29.89  29.89  29.89   29.89
 ALL                             16      0      0   4.37  14.06  26.23   12.41
```


worst 10: 4x4x4x4x4x53 (- 4.04), 10x10x10x10x10x10 (- 4.37), 9x9x9x9x9x9 (- 5.11), 12x10x8x6x4x2 (- 9.72), 16x4x4x4x4x16 (- 10.13), 8x8x8x8x8x8 (- 10.73), 5x5x5x5x5x5 (- 11.69), 6x6x6x6x6x6 (- 14.01), 2x4x6x8x10x12 (- 14.11), 3x5x7x7x5x3 (- 15.23)
best 5: 4x4x4x4x4x4 (- 29.89), 2x4x2x4x2x4 (- 26.23), 8x8x8x4x4x4 (- 21.90), 4x4x4x8x8x8 (- 19.53), 3x3x3x3x3x3 (- 18.08)
