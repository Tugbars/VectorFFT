# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2c interleaved, natural, out of place, K=4 lane-major_k4_lm_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 13.279..13.426


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            lm     -               1429      37754   26.39     12.5  1.8e-16
  2x2x2x2x2x2x53  2            lm     -              51734     295874    5.70     15.4  1.3e-15
   2x3x4x5x6x7x8  2            lm     -             295383    2672875    9.03     41.8  4.4e-16
   2x4x8x2x4x8x2  2            lm     -              74472     963644   12.90     28.6  2.6e-16
   3x3x3x3x3x3x3  3            lm     -              18113     283753   15.62     26.8  5.0e-16
   3x5x7x3x5x7x3  3            lm     -             289453    2806337    9.64     34.3  4.6e-16
   4x4x4x4x4x4x4  2^2          lm     -             118070    1571262   13.17     38.9  3.8e-16
   4x4x4x4x8x8x8  2^2          lm     -             960650    8648419    8.97     46.4  1.7e-15
   5x5x5x5x5x5x5  5            lm     -             649212    6154124    9.45     39.1  5.0e-16
   6x6x6x6x6x6x6  2.3          lm     -            2707250   22529925    8.22     37.4  8.8e-16
   8x8x8x8x8x8x4  2^3          lm     -           14032063  130755231    9.30     29.9  1.8e-15
   8x8x8x8x8x8x8  2^3          lm     -           28273900  251365300    8.87     31.2  2.3e-15
 10x8x6x4x6x8x10  2.5          lm     -           11530063  111647481    9.66     31.7  5.5e-16
  53x2x2x2x2x2x2  53           lm     -              60452     987987   16.30     13.2  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              14      0      0   8.22   9.54  16.30   10.85
 ALL                             14      0      0   8.22   9.54  16.30   10.85
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   5.70   9.17  26.39   10.69
 prime column                     4      0      0   9.45  12.63  16.30   12.34
 even column                      2      0      0   8.22   8.94   9.66    8.91
 ALL                             14      0      0   8.22   9.54  16.30   10.85
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   8.22   9.14   9.66    9.07
 4097..65536                      4      0      0   9.03  11.27  13.17   11.03
 1025..4096                       3      0      0   5.70  15.62  16.30   11.32
 <= 256 points                    1      0      0  26.39  26.39  26.39   26.39
 ALL                             14      0      0   8.22   9.54  16.30   10.85
```


worst 10: 2x2x2x2x2x2x53 (lm 5.70), 6x6x6x6x6x6x6 (lm 8.22), 8x8x8x8x8x8x8 (lm 8.87), 4x4x4x4x8x8x8 (lm 8.97), 2x3x4x5x6x7x8 (lm 9.03), 8x8x8x8x8x8x4 (lm 9.30), 5x5x5x5x5x5x5 (lm 9.45), 3x5x7x3x5x7x3 (lm 9.64), 10x8x6x4x6x8x10 (lm 9.66), 2x4x8x2x4x8x2 (lm 12.90)
best 5: 2x2x2x2x2x2x2 (lm 26.39), 53x2x2x2x2x2x2 (lm 16.30), 3x3x3x3x3x3x3 (lm 15.62), 4x4x4x4x4x4x4 (lm 13.17), 2x4x8x2x4x8x2 (lm 12.90)
