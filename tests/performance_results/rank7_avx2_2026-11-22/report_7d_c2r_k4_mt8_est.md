# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2r interleaved, natural, out of place, K=4 contiguous_c2r_k4_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 16.952..22.158

threaded plans that ran serial (engaged = 0 at both flips): 1: 2x2x2x2x2x2x2


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -               4889      18214    3.68      1.8  2.2e-16
  2x2x2x2x2x2x53  2            -      -              18578      74750    3.85     21.4  1.3e-15
   2x3x4x5x6x7x8  2            -      -              72733    1011579   13.06     84.8  1.2e-15
   2x4x8x2x4x8x2  2            -      -              37205     617651   15.49     28.6  5.6e-16
   3x3x3x3x3x3x3  3            -      -              15439     139727    7.29     15.7  1.2e-15
   3x5x7x3x5x7x3  3            -      -             157467    1406403    8.68     31.5  1.3e-15
   4x4x4x4x4x4x4  2^2          -      -              28437     667113   22.56     80.7  5.6e-16
   4x4x4x4x8x8x8  2^2          -      -             210112    2453250   10.55    106.0  7.8e-16
   5x5x5x5x5x5x5  5            -      -             324712    2098913    6.22     39.1  1.3e-15
   6x6x6x6x6x6x6  2.3          -      -             368763    4892393   12.53    137.4  1.9e-15
   8x8x8x8x8x8x4  2^3          -      -            2271312   23624769    9.24     92.3  1.0e-15
   8x8x8x8x8x8x8  2^3          -      -            5455925   28477456    5.10     80.7  1.0e-15
 10x8x6x4x6x8x10  2.5          -      -            1694425   11853669    6.71    107.8  1.5e-15
  53x2x2x2x2x2x2  53           -      -              42864     406972    9.08      9.3  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   3.85   8.88  15.49    8.46
 ALL                             14      0      0   3.85   8.88  15.49    8.46
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   3.68   9.89  22.56    8.68
 prime column                     4      0      0   6.22   7.99   9.08    7.73
 even column                      2      0      0   6.71   9.62  12.53    9.17
 ALL                             14      0      0   3.85   8.88  15.49    8.46
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   5.10   7.97  12.53    7.99
 4097..65536                      4      0      0   8.68  14.28  22.56   14.11
 1025..4096                       3      0      0   3.85   7.29   9.08    6.34
 <= 256 points                    1      0      0   3.68   3.68   3.68    3.68
 ALL                             14      0      0   3.85   8.88  15.49    8.46
```


worst 10: 2x2x2x2x2x2x2 (- 3.68), 2x2x2x2x2x2x53 (- 3.85), 8x8x8x8x8x8x8 (- 5.10), 5x5x5x5x5x5x5 (- 6.22), 10x8x6x4x6x8x10 (- 6.71), 3x3x3x3x3x3x3 (- 7.29), 3x5x7x3x5x7x3 (- 8.68), 53x2x2x2x2x2x2 (- 9.08), 8x8x8x8x8x8x4 (- 9.24), 4x4x4x4x8x8x8 (- 10.55)
best 5: 4x4x4x4x4x4x4 (- 22.56), 2x4x8x2x4x8x2 (- 15.49), 2x3x4x5x6x7x8 (- 13.06), 6x6x6x6x6x6x6 (- 12.53), 4x4x4x4x8x8x8 (- 10.55)
