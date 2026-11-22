# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D c2r interleaved, natural, out of place, K=8 lane-major_c2r_k8_lm_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 8.209..9.226


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            lm     -             325450    2102006    6.02     60.4  7.8e-16
      4x4x4x4x4  2^2          lm     -               9685     234940   24.00     42.3  4.4e-16
    4x4x4x4x256  2^2          lm     -             677562    3002769    4.33     61.9  1.0e-15
      6x6x6x6x6  2.3          lm     -              75753     835864   10.56     53.1  1.5e-15
      8x8x8x8x8  2^3          lm     -             327875    2737531    7.01     60.0  6.7e-16
     8x8x53x8x8  2^3          lm     -            4597312   27027431    5.87     33.5  1.9e-15
 10x10x10x10x10  2.5          lm     -             999050    6818625    6.58     66.5  1.5e-15
 12x12x12x12x12  2^2.3        lm     -            3512200   17399300    4.91     50.8  1.9e-15
   16x16x16x8x8  2^4          lm     -            5403500   22543168    4.14     34.9  8.9e-16
 16x16x16x16x16  2^4          lm     -           22848700   84779181    3.71     36.7  1.0e-15
   20x16x12x8x4  2^2.5        lm     -            2084350   14662963    6.94     39.9  1.2e-15
  24x12x6x12x24  2^3.3        lm     -            8011538   35852200    4.44     47.0  2.0e-15
 32x16x16x16x16  2^5          lm     -           46847750  180522962    3.80     37.6  1.1e-15
     53x8x8x8x8  53           lm     -            5218212   28865512    5.47     29.5  1.8e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              14      0      0   3.80   5.67  10.56    6.04
 ALL                             14      0      0   3.80   5.67  10.56    6.04
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   3.71   5.10  24.00    5.92
 even column                      5      0      0   4.44   6.58  10.56    6.38
 prime column                     1      0      0   5.47   5.47   5.47    5.47
 ALL                             14      0      0   3.80   5.67  10.56    6.04
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   9      0      0   3.71   4.91   6.94    4.98
 4097..65536                      4      0      0   4.33   6.51  10.56    6.63
 257..1024                        1      0      0  24.00  24.00  24.00   24.00
 ALL                             14      0      0   3.80   5.67  10.56    6.04
```


worst 10: 16x16x16x16x16 (lm 3.71), 32x16x16x16x16 (lm 3.80), 16x16x16x8x8 (lm 4.14), 4x4x4x4x256 (lm 4.33), 24x12x6x12x24 (lm 4.44), 12x12x12x12x12 (lm 4.91), 53x8x8x8x8 (lm 5.47), 8x8x53x8x8 (lm 5.87), 2x4x8x16x32 (lm 6.02), 10x10x10x10x10 (lm 6.58)
best 5: 4x4x4x4x4 (lm 24.00), 6x6x6x6x6 (lm 10.56), 8x8x8x8x8 (lm 7.01), 20x16x12x8x4 (lm 6.94), 10x10x10x10x10 (lm 6.58)
