# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D r2c interleaved, natural, out of place, K=1_r2c_mt4_est  cells: 17 listed, 17 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 7.737..8.864

threaded plans that ran serial (engaged = 0 at both flips): 3: 2x4x2x4x2x4, 2x4x6x8x10x12, 3x3x3x3x3x3


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            child  -                988      18634   17.62     11.7  0.0e+00
     2x4x6x8x10x12  2            child  -              40685     264471    6.41     43.9  4.4e-16
       3x3x3x3x3x3  3            child  -               2171      17525    7.86      8.0  5.9e-16
       3x5x7x7x5x3  3            child  -              14049     146066   10.20     26.3  5.2e-16
       4x4x4x4x4x4  2^2          child  -               4052      42029    9.90     30.3  0.0e+00
      4x4x4x4x4x53  2^2          child  -              57899     190330    3.08     36.9  7.7e-16
       4x4x4x8x8x8  2^2          child  -              14690     148051    9.94     83.6  2.8e-16
       5x5x5x5x5x5  5            child  -              20685     167733    8.06     26.3  4.1e-16
       6x6x6x6x6x6  2.3          child  -              25280     255844    8.14     71.6  6.6e-16
       8x8x8x4x4x4  2^3          child  -              20357     272045   12.44     60.4  2.7e-16
       8x8x8x8x8x8  2^3          child  -              84500     762843    8.53    139.6  3.5e-16
       9x9x9x9x9x9  3^2          child  -             357300    1747975    4.75     70.7  7.3e-16
 10x10x10x10x10x10  2.5          child  -             423600    2213412    4.98    117.6  6.2e-16
     12x10x8x6x4x2  2^2.3        child  -              63230     516682    8.08     28.2  5.3e-16
     16x4x4x4x4x16  2^4          child  -              18585     182638    9.21    141.1  3.7e-16
 16x16x16x16x16x16  2^4          child  -           23528613   27995375    1.12     42.8  4.2e-16
      53x4x4x4x4x4  53           child  -              41853     567677   12.75     51.0  7.8e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           17      0      0   3.08   8.14  12.75    7.31
 ALL                             17      0      0   3.08   8.14  12.75    7.31
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      9      0      0   1.12   9.21  17.62    6.95
 prime column                     4      0      0   7.86   9.13  12.75    9.53
 even column                      3      0      0   4.98   8.08   8.14    6.89
 odd column                       1      0      0   4.75   4.75   4.75    4.75
 ALL                             17      0      0   3.08   8.14  12.75    7.31
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   6.41   8.68  12.75    8.30
 > 65536 points                   4      0      0   1.12   4.86   8.53    3.88
 257..1024                        2      0      0   7.86  12.74  17.62   11.77
 1025..4096                       1      0      0   9.90   9.90   9.90    9.91
 ALL                             17      0      0   3.08   8.14  12.75    7.31
```


worst 10: 16x16x16x16x16x16 (child 1.12), 4x4x4x4x4x53 (child 3.08), 9x9x9x9x9x9 (child 4.75), 10x10x10x10x10x10 (child 4.98), 2x4x6x8x10x12 (child 6.41), 3x3x3x3x3x3 (child 7.86), 5x5x5x5x5x5 (child 8.06), 12x10x8x6x4x2 (child 8.08), 6x6x6x6x6x6 (child 8.14), 8x8x8x8x8x8 (child 8.53)
best 5: 2x4x2x4x2x4 (child 17.62), 53x4x4x4x4x4 (child 12.75), 8x8x8x4x4x4 (child 12.44), 3x5x7x7x5x3 (child 10.20), 4x4x4x8x8x8 (child 9.94)
