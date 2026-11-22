# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2r interleaved, natural, out of place, K=4 contiguous_c2r_k4_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 7.087..7.527


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               3996     109385   27.32     11.5  4.4e-16
     2x4x6x8x10x12  2            -      -             205790    2469585   11.82     34.7  1.1e-15
       3x3x3x3x3x3  3            -      -              13341     146934   10.91      5.2  9.4e-16
       3x5x7x7x5x3  3            -      -             191918    1391776    7.23      7.7  1.2e-15
       4x4x4x4x4x4  2^2          -      -              23382     536590   22.76     21.0  5.6e-16
      4x4x4x4x4x53  2^2          -      -            1034422    2838178    2.72      8.3  1.5e-15
       4x4x4x8x8x8  2^2          -      -             147527    1949176   13.09     33.3  6.7e-16
       5x5x5x5x5x5  5            -      -             255532    1349024    5.24      8.5  9.4e-16
       6x6x6x6x6x6  2.3          -      -             252460    2683240   10.45     28.7  1.6e-15
       8x8x8x4x4x4  2^3          -      -             203340    3213386   15.79     24.2  6.7e-16
       8x8x8x8x8x8  2^3          -      -            1379138   10059156    6.86     34.2  8.9e-16
       9x9x9x9x9x9  3^2          -      -            6120250   18520750    3.01     16.5  2.4e-15
 10x10x10x10x10x10  2.5          -      -            7018000   27873112    3.88     28.4  1.7e-15
     12x10x8x6x4x2  2^2.3        -      -             918490    6418380    6.86      7.8  9.5e-16
     16x4x4x4x4x16  2^4          -      -             307800    2409644    7.74     34.1  6.7e-16
      53x4x4x4x4x4  53           -      -             542767    7472555   12.68     15.7  1.6e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               16      0      0   3.01   9.10  22.76    8.64
 ALL                             16      0      0   3.01   9.10  22.76    8.64
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   2.72  12.46  27.32   11.03
 prime column                     4      0      0   5.24   9.07  12.68    8.51
 even column                      3      0      0   3.88   6.86  10.45    6.53
 odd column                       1      0      0   3.01   3.01   3.01    3.01
 ALL                             16      0      0   3.01   9.10  22.76    8.64
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   5.24   9.10  15.79    8.41
 > 65536 points                   3      0      0   3.01   3.88   6.86    4.31
 257..1024                        2      0      0  10.91  19.12  27.32   17.27
 1025..4096                       1      0      0  22.76  22.76  22.76   22.76
 ALL                             16      0      0   3.01   9.10  22.76    8.64
```


worst 10: 4x4x4x4x4x53 (- 2.72), 9x9x9x9x9x9 (- 3.01), 10x10x10x10x10x10 (- 3.88), 5x5x5x5x5x5 (- 5.24), 8x8x8x8x8x8 (- 6.86), 12x10x8x6x4x2 (- 6.86), 3x5x7x7x5x3 (- 7.23), 16x4x4x4x4x16 (- 7.74), 6x6x6x6x6x6 (- 10.45), 3x3x3x3x3x3 (- 10.91)
best 5: 2x4x2x4x2x4 (- 27.32), 4x4x4x4x4x4 (- 22.76), 8x8x8x4x4x4 (- 15.79), 4x4x4x8x8x8 (- 13.09), 53x4x4x4x4x4 (- 12.68)
