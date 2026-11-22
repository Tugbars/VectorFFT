# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2r interleaved, natural, out of place, K=1_c2r_fftw_est  cells: 14 listed, 14 benched, comparator: FFTW 7D (out of place, MEASURE)

control cell 4x4x4x4x4x4x4: 4 readings, 0.802..0.818


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            child  -               1220        312    0.25      1.8  3.4e-16
  2x2x2x2x2x2x53  2            child  -              15060      29274    1.61      6.6  1.1e-15
   2x3x4x5x6x7x8  2            child  -              62326      48543    0.78     24.7  1.2e-15
   2x4x8x2x4x8x2  2            child  -              32686      14006    0.42      8.1  5.6e-16
   3x3x3x3x3x3x3  3            child  -              11086       3166    0.28      5.5  1.2e-15
   3x5x7x3x5x7x3  3            child  -             148459      46297    0.31      8.4  1.2e-15
   4x4x4x4x4x4x4  2^2          child  -              26869      22003    0.81     21.3  4.4e-16
   4x4x4x4x8x8x8  2^2          child  -             192277     151153    0.76     29.0  7.8e-16
   5x5x5x5x5x5x5  5            child  -             350120     100965    0.29      9.1  1.0e-15
   6x6x6x6x6x6x6  2.3          child  -             491114     437946    0.86     25.8  2.0e-15
   8x8x8x8x8x8x4  2^3          child  -            2628075    2120175    0.77     19.9  1.2e-15
   8x8x8x8x8x8x8  2^3          child  -            5160875    4695256    0.90     21.3  1.3e-15
 10x8x6x4x6x8x10  2.5          child  -            1615537    1389725    0.85     28.3  1.5e-15
  53x2x2x2x2x2x2  53           child  -              37875      26498    0.68      2.6  9.8e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           14      9     13   0.28   0.77   0.90    0.60
 ALL                             14      9     13   0.28   0.77   0.90    0.60
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      5      7   0.25   0.77   1.61    0.70
 prime column                     4      4      4   0.28   0.30   0.68    0.36
 even column                      2      0      2   0.85   0.86   0.86    0.86
 ALL                             14      9     13   0.28   0.77   0.90    0.60
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      3      6   0.29   0.81   0.90    0.69
 4097..65536                      4      3      4   0.31   0.60   0.81    0.54
 1025..4096                       3      2      2   0.28   0.68   1.61    0.68
 <= 256 points                    1      1      1   0.25   0.25   0.25    0.25
 ALL                             14      9     13   0.28   0.77   0.90    0.60
```


worst 10: 2x2x2x2x2x2x2 (child 0.25), 3x3x3x3x3x3x3 (child 0.28), 5x5x5x5x5x5x5 (child 0.29), 3x5x7x3x5x7x3 (child 0.31), 2x4x8x2x4x8x2 (child 0.42), 53x2x2x2x2x2x2 (child 0.68), 4x4x4x4x8x8x8 (child 0.76), 8x8x8x8x8x8x4 (child 0.77), 2x3x4x5x6x7x8 (child 0.78), 4x4x4x4x4x4x4 (child 0.81)
best 5: 2x2x2x2x2x2x53 (child 1.61), 8x8x8x8x8x8x8 (child 0.90), 6x6x6x6x6x6x6 (child 0.86), 10x8x6x4x6x8x10 (child 0.85), 4x4x4x4x4x4x4 (child 0.81)
