# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2c interleaved, natural, out of place, K=1_fftw_est  cells: 17 listed, 17 benched, comparator: FFTW 6D (out of place, MEASURE)

control cell 8x8x8x8x8x8: 4 readings, 1.028..1.045


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               1256       1022    0.78     18.3  1.4e-16
     2x4x6x8x10x12  2            -      -              68693      68702    1.00     52.0  6.6e-16
       3x3x3x3x3x3  3            -      -               2136       1652    0.73     16.2  5.3e-16
       3x5x7x7x5x3  3            -      -              26902      22884    0.84     27.5  4.9e-16
       4x4x4x4x4x4  2^2          -      -               7432       6824    0.90     33.1  2.2e-16
      4x4x4x4x4x53  2^2          -      -             238628     353651    1.41     17.9  1.0e-15
       4x4x4x8x8x8  2^2          -      -              45100      49499    1.09     54.5  2.4e-16
       5x5x5x5x5x5  5            -      -              32847      29690    0.89     33.1  4.7e-16
       6x6x6x6x6x6  2.3          -      -              79667      74085    0.92     45.4  7.6e-16
       8x8x8x4x4x4  2^3          -      -              66156      65454    0.83     37.1  2.9e-16
       8x8x8x8x8x8  2^3          -      -             442050     454400    0.97     53.4  4.9e-16
       9x9x9x9x9x9  3^2          -      -            1124138    1165737    1.03     45.0  8.2e-16
 10x10x10x10x10x10  2.5          -      -            2134738    2134756    0.97     46.7  5.8e-16
     12x10x8x6x4x2  2^2.3        -      -             147051     104485    0.70     24.3  5.8e-16
     16x4x4x4x4x16  2^4          -      -             110980     132151    1.14     47.2  3.3e-16
 16x16x16x16x16x16  2^4          -      -           80066150   84054387    1.04     25.1  4.5e-15
      53x4x4x4x4x4  53           -      -             172561     398941    2.29     24.7  6.8e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               17      3     11   0.73   0.97   1.41    0.99
 ALL                             17      3     11   0.73   0.97   1.41    0.99
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      9      1      5   0.78   1.00   1.41    1.00
 prime column                     4      1      3   0.73   0.87   2.29    1.06
 even column                      3      1      3   0.70   0.92   0.97    0.85
 odd column                       1      0      0   1.03   1.03   1.03    1.03
 ALL                             17      3     11   0.73   0.97   1.41    0.99
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      1      6   0.83   0.96   2.29    1.05
 > 65536 points                   4      0      2   0.97   1.00   1.04    1.00
 257..1024                        2      2      2   0.73   0.75   0.78    0.75
 1025..4096                       1      0      1   0.90   0.90   0.90    0.90
 ALL                             17      3     11   0.73   0.97   1.41    0.99
```


worst 10: 12x10x8x6x4x2 (- 0.70), 3x3x3x3x3x3 (- 0.73), 2x4x2x4x2x4 (- 0.78), 8x8x8x4x4x4 (- 0.83), 3x5x7x7x5x3 (- 0.84), 5x5x5x5x5x5 (- 0.89), 4x4x4x4x4x4 (- 0.90), 6x6x6x6x6x6 (- 0.92), 8x8x8x8x8x8 (- 0.97), 10x10x10x10x10x10 (- 0.97)
best 5: 53x4x4x4x4x4 (- 2.29), 4x4x4x4x4x53 (- 1.41), 16x4x4x4x4x16 (- 1.14), 4x4x4x8x8x8 (- 1.09), 16x16x16x16x16x16 (- 1.04)
