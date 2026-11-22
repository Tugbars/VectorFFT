# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D r2c interleaved, natural, out of place, K=1_r2c_mt8_est  cells: 21 listed, 21 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 4.746..8.309

threaded plans that ran serial (engaged = 0 at both flips): 2: 2x4x8x16x32, 4x4x4x4x4


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            child  -              25789      86216    3.08     47.6  3.1e-16
     3x5x7x9x11  3            child  -              11944      37676    3.06     29.0  6.1e-16
      4x4x4x4x4  2^2          child  -               1045      10317    9.52     24.5  0.0e+00
    4x4x4x4x256  2^2          child  -              28308      77659    2.48     92.6  3.7e-16
      5x5x5x5x5  5            child  -               3825      17142    4.43     23.7  2.9e-16
      6x6x6x6x6  2.3          child  -               5284      21595    3.89     47.6  5.8e-16
      8x8x8x8x8  2^3          child  -               9971      66159    4.95    123.2  2.7e-16
     8x8x8x8x53  2^3          child  -              99167     366636    2.65     97.0  7.1e-16
     8x8x53x8x8  2^3          child  -              76389     489703    4.31    126.0  6.2e-16
      9x9x9x9x9  3^2          child  -              35846     139588    3.30     65.3  6.9e-16
 10x10x10x10x10  2.5          child  -              28985     188104    3.96    143.3  6.4e-16
 12x12x12x12x12  2^2.3        child  -              69175     344172    2.92    161.2  7.2e-16
     15x9x7x5x3  3.5          child  -               9064      54667    4.73     53.9  5.5e-16
 15x15x15x15x15  3.5          child  -             307475     495375    1.31    120.6  8.0e-16
   16x16x16x8x8  2^4          child  -              64500     363480    4.05    182.9  3.6e-16
 16x16x16x16x16  2^4          child  -             215950     871425    3.61    242.8  3.5e-16
   20x16x12x8x4  2^2.5        child  -              38978     306098    6.86    133.2  4.2e-16
  24x12x6x12x24  2^3.3        child  -             129600     433163    2.42    181.7  7.4e-16
    32x16x8x4x2  2^5          child  -              25635     145792    5.44     47.9  3.6e-16
 32x16x16x16x16  2^5          child  -             406025    1202500    2.45    271.2  3.9e-16
     53x8x8x8x8  53           child  -              56250     468861    5.16    171.0  6.7e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           21      0      0   2.45   3.89   5.44    3.70
 ALL                             21      0      0   2.45   3.89   5.44    3.70
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      0      0   2.48   3.83   9.52    3.90
 even column                      5      0      0   2.42   3.89   6.86    3.76
 prime column                     3      0      0   3.06   4.43   5.16    4.12
 odd column                       3      0      0   1.31   3.30   4.73    2.73
 ALL                             21      0      0   2.45   3.89   5.44    3.70
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  11      0      0   2.42   3.61   5.16    3.31
 4097..65536                      8      0      0   2.48   3.59   5.44    3.74
 257..1024                        1      0      0   9.52   9.52   9.52    9.52
 1025..4096                       1      0      0   4.43   4.43   4.43    4.43
 ALL                             21      0      0   2.45   3.89   5.44    3.70
```


worst 10: 15x15x15x15x15 (child 1.31), 24x12x6x12x24 (child 2.42), 32x16x16x16x16 (child 2.45), 4x4x4x4x256 (child 2.48), 8x8x8x8x53 (child 2.65), 12x12x12x12x12 (child 2.92), 3x5x7x9x11 (child 3.06), 2x4x8x16x32 (child 3.08), 9x9x9x9x9 (child 3.30), 16x16x16x16x16 (child 3.61)
best 5: 4x4x4x4x4 (child 9.52), 20x16x12x8x4 (child 6.86), 32x16x8x4x2 (child 5.44), 53x8x8x8x8 (child 5.16), 8x8x8x8x8 (child 4.95)
