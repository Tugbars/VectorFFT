# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2r interleaved, natural, out of place, K=8 lane-major_c2r_k8_lm_est  cells: 10 listed, 10 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 6.100..6.643


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            lm     -               6435     228323   35.04     28.6  3.3e-16
     2x4x6x8x10x12  2            lm     -             534600    5689937   10.46     53.4  1.3e-15
       4x4x4x4x4x4  2^2          lm     -              45944    1126974   23.99     42.8  4.4e-16
       4x4x4x8x8x8  2^2          lm     -             364838    4459887   12.20     53.9  7.8e-16
       6x6x6x6x6x6  2.3          lm     -             571875    6112481   10.54     50.6  1.9e-15
       8x8x8x4x4x4  2^3          lm     -             451800    6996512   13.95     43.5  6.7e-16
       8x8x8x8x8x8  2^3          lm     -            4590950   29991950    6.50     41.1  8.9e-16
 10x10x10x10x10x10  2.5          lm     -           20015600  112865012    5.57     39.8  1.6e-15
     16x4x4x4x4x16  2^4          lm     -             747375    5986337    7.99     56.1  6.7e-16
      53x4x4x4x4x4  53           lm     -            1179000   15677500   11.49     29.0  1.7e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              10      0      0   6.50  11.01  35.04   11.80
 ALL                             10      0      0   6.50  11.01  35.04   11.80
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      7      0      0   6.50  12.20  35.04   13.40
 even column                      2      0      0   5.57   8.05  10.54    7.66
 prime column                     1      0      0  11.49  11.49  11.49   11.49
 ALL                             10      0      0   6.50  11.01  35.04   11.80
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      6      0      0   7.99  11.01  13.95   10.95
 > 65536 points                   2      0      0   5.57   6.03   6.50    6.02
 257..1024                        1      0      0  35.04  35.04  35.04   35.04
 1025..4096                       1      0      0  23.99  23.99  23.99   23.99
 ALL                             10      0      0   6.50  11.01  35.04   11.80
```


worst 10: 10x10x10x10x10x10 (lm 5.57), 8x8x8x8x8x8 (lm 6.50), 16x4x4x4x4x16 (lm 7.99), 2x4x6x8x10x12 (lm 10.46), 6x6x6x6x6x6 (lm 10.54), 53x4x4x4x4x4 (lm 11.49), 4x4x4x8x8x8 (lm 12.20), 8x8x8x4x4x4 (lm 13.95), 4x4x4x4x4x4 (lm 23.99), 2x4x2x4x2x4 (lm 35.04)
best 5: 2x4x2x4x2x4 (lm 35.04), 4x4x4x4x4x4 (lm 23.99), 8x8x8x4x4x4 (lm 13.95), 4x4x4x8x8x8 (lm 12.20), 53x4x4x4x4x4 (lm 11.49)
