# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2c interleaved, natural, out of place, K=4 contiguous_k4_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 9.487..14.293

threaded plans that ran serial (engaged = 0 at both flips): 1: 2x2x2x2x2x2x2


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -               3486       9408    2.62      5.1  1.6e-16
  2x2x2x2x2x2x53  2            -      -              11940     170047   12.57     66.6  1.3e-15
   2x3x4x5x6x7x8  2            -      -              41017     888354   15.32    300.8  5.0e-16
   2x4x8x2x4x8x2  2            -      -              18008     180678    8.20    118.3  3.0e-16
   3x3x3x3x3x3x3  3            -      -               8120      62279    6.03     59.8  6.4e-16
   3x5x7x3x5x7x3  3            -      -              52633     765033    8.80    188.7  6.1e-16
   4x4x4x4x4x4x4  2^2          -      -              27427     350450    9.67    167.3  2.0e-16
   4x4x4x4x8x8x8  2^2          -      -             115125    1195681    8.42    387.1  6.4e-16
   5x5x5x5x5x5x5  5            -      -             110313     991912    5.52    230.2  5.0e-16
   6x6x6x6x6x6x6  2.3          -      -             317275    1972369    4.73    319.3  9.1e-16
   8x8x8x8x8x8x4  2^3          -      -            3679925    6613068    1.71    114.0  1.4e-15
   8x8x8x8x8x8x8  2^3          -      -            9449625   13807319    1.39     93.2  1.8e-15
 10x8x6x4x6x8x10  2.5          -      -            3322025    7208718    2.09    109.9  6.3e-16
  53x2x2x2x2x2x2  53           -      -              33003     180345    5.29     24.1  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   1.71   5.78  12.57    5.25
 ALL                             14      0      0   1.71   5.78  12.57    5.25
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   1.39   8.31  15.32    5.47
 prime column                     4      0      0   5.29   5.78   8.80    6.27
 even column                      2      0      0   2.09   3.41   4.73    3.14
 ALL                             14      0      0   1.71   5.78  12.57    5.25
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   1.39   3.41   8.42    3.21
 4097..65536                      4      0      0   8.20   9.23  15.32   10.17
 1025..4096                       3      0      0   5.29   6.03  12.57    7.37
 <= 256 points                    1      0      0   2.62   2.62   2.62    2.62
 ALL                             14      0      0   1.71   5.78  12.57    5.25
```


worst 10: 8x8x8x8x8x8x8 (- 1.39), 8x8x8x8x8x8x4 (- 1.71), 10x8x6x4x6x8x10 (- 2.09), 2x2x2x2x2x2x2 (- 2.62), 6x6x6x6x6x6x6 (- 4.73), 53x2x2x2x2x2x2 (- 5.29), 5x5x5x5x5x5x5 (- 5.52), 3x3x3x3x3x3x3 (- 6.03), 2x4x8x2x4x8x2 (- 8.20), 4x4x4x4x8x8x8 (- 8.42)
best 5: 2x3x4x5x6x7x8 (- 15.32), 2x2x2x2x2x2x53 (- 12.57), 4x4x4x4x4x4x4 (- 9.67), 3x5x7x3x5x7x3 (- 8.80), 4x4x4x4x8x8x8 (- 8.42)
