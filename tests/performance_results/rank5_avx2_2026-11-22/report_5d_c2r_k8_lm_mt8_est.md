# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D c2r interleaved, natural, out of place, K=8 lane-major_c2r_k8_lm_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 7.598..13.592

threaded plans that ran serial (engaged = 0 at both flips): 1: 2x4x8x16x32


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            lm     -             327850     393737    0.98     60.0  7.8e-16
      4x4x4x4x4  2^2          lm     -               4489      50105    9.18     91.2  4.4e-16
    4x4x4x4x256  2^2          lm     -             173338     611850    3.24    242.0  1.0e-15
      6x6x6x6x6  2.3          lm     -              20281     188453    7.68    198.2  1.5e-15
      8x8x8x8x8  2^3          lm     -              53037     582025    7.46    370.7  6.7e-16
     8x8x53x8x8  2^3          lm     -            1117550    3853919    3.02    137.7  1.9e-15
 10x10x10x10x10  2.5          lm     -             216500    1150137    4.25    306.9  1.5e-15
 12x12x12x12x12  2^2.3        lm     -             966313    2837900    2.56    184.6  1.9e-15
   16x16x16x8x8  2^4          lm     -            1489250    3747919    2.29    126.7  8.9e-16
 16x16x16x16x16  2^4          lm     -           10743812   13330450    1.04     78.1  1.0e-15
   20x16x12x8x4  2^2.5        lm     -             310562    2229062    6.63    267.6  1.2e-15
  24x12x6x12x24  2^3.3        lm     -            2345412    5589319    2.38    160.6  2.0e-15
 32x16x16x16x16  2^5          lm     -           21816262   28889093    1.15     80.7  1.1e-15
     53x8x8x8x8  53           lm     -             792625    4470474    4.69    194.2  1.8e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              14      0      1   1.04   3.13   7.68    3.19
 ALL                             14      0      1   1.04   3.13   7.68    3.19
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      1   0.98   2.65   9.18    2.55
 even column                      5      0      0   2.38   4.25   7.68    4.21
 prime column                     1      0      0   4.69   4.69   4.69    4.69
 ALL                             14      0      1   1.04   3.13   7.68    3.19
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   9      0      0   1.04   2.56   6.63    2.66
 4097..65536                      4      0      1   0.98   5.35   7.68    3.68
 257..1024                        1      0      0   9.18   9.18   9.18    9.18
 ALL                             14      0      1   1.04   3.13   7.68    3.19
```


worst 10: 2x4x8x16x32 (lm 0.98), 16x16x16x16x16 (lm 1.04), 32x16x16x16x16 (lm 1.15), 16x16x16x8x8 (lm 2.29), 24x12x6x12x24 (lm 2.38), 12x12x12x12x12 (lm 2.56), 8x8x53x8x8 (lm 3.02), 4x4x4x4x256 (lm 3.24), 10x10x10x10x10 (lm 4.25), 53x8x8x8x8 (lm 4.69)
best 5: 4x4x4x4x4 (lm 9.18), 6x6x6x6x6 (lm 7.68), 8x8x8x8x8 (lm 7.46), 20x16x12x8x4 (lm 6.63), 53x8x8x8x8 (lm 4.69)
