# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2c interleaved, natural, out of place, K=1_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 8.461..8.576


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -                892       8362    9.30      5.0  1.8e-16
  2x2x2x2x2x2x53  2            -      -              14590      64313    4.02     13.6  8.7e-16
   2x3x4x5x6x7x8  2            -      -              69378     395957    5.70     44.5  4.7e-16
   2x4x8x2x4x8x2  2            -      -              18530     161414    8.65     28.7  2.7e-16
   3x3x3x3x3x3x3  3            -      -               7288      58101    7.96     16.6  5.1e-16
   3x5x7x3x5x7x3  3            -      -              74530     455726    6.08     33.3  4.6e-16
   4x4x4x4x4x4x4  2^2          -      -              31845     272384    8.37     36.0  2.6e-16
   4x4x4x4x8x8x8  2^2          -      -             221433    1123316    5.07     50.3  7.4e-16
   5x5x5x5x5x5x5  5            -      -             179624     917472    5.04     35.3  4.8e-16
   6x6x6x6x6x6x6  2.3          -      -             588812    2508219    4.25     43.0  8.3e-16
   8x8x8x8x8x8x4  2^3          -      -            2306838   10229794    4.38     45.5  1.3e-15
   8x8x8x8x8x8x8  2^3          -      -            5731575   17856075    3.10     38.4  2.1e-15
 10x8x6x4x6x8x10  2.5          -      -            1951550    6920356    3.40     46.8  5.6e-16
  53x2x2x2x2x2x2  53           -      -              28764     210765    7.30      6.9  1.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   3.40   5.38   8.65    5.57
 ALL                             14      0      0   3.40   5.38   8.65    5.57
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   3.10   5.38   9.30    5.66
 prime column                     4      0      0   5.04   6.69   7.96    6.50
 even column                      2      0      0   3.40   3.83   4.25    3.80
 ALL                             14      0      0   3.40   5.38   8.65    5.57
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   3.10   4.31   5.07    4.14
 4097..65536                      4      0      0   5.70   7.23   8.65    7.08
 1025..4096                       3      0      0   4.02   7.30   7.96    6.16
 <= 256 points                    1      0      0   9.30   9.30   9.30    9.30
 ALL                             14      0      0   3.40   5.38   8.65    5.57
```


worst 10: 8x8x8x8x8x8x8 (- 3.10), 10x8x6x4x6x8x10 (- 3.40), 2x2x2x2x2x2x53 (- 4.02), 6x6x6x6x6x6x6 (- 4.25), 8x8x8x8x8x8x4 (- 4.38), 5x5x5x5x5x5x5 (- 5.04), 4x4x4x4x8x8x8 (- 5.07), 2x3x4x5x6x7x8 (- 5.70), 3x5x7x3x5x7x3 (- 6.08), 53x2x2x2x2x2x2 (- 7.30)
best 5: 2x2x2x2x2x2x2 (- 9.30), 2x4x8x2x4x8x2 (- 8.65), 4x4x4x4x4x4x4 (- 8.37), 3x3x3x3x3x3x3 (- 7.96), 53x2x2x2x2x2x2 (- 7.30)
