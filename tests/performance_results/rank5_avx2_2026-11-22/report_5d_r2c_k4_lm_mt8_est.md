# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D r2c interleaved, natural, out of place, K=4 lane-major_r2c_k4_lm_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 16.600..20.984

threaded plans that ran serial (engaged = 0 at both flips): 1: 2x4x8x16x32


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            lm     -             104120     449867    3.85     94.4  3.3e-16
      4x4x4x4x4  2^2          lm     -               3981      66860   16.58     51.4  0.0e+00
    4x4x4x4x256  2^2          lm     -              82825     699156    7.67    253.2  3.9e-16
      6x6x6x6x6  2.3          lm     -              15128     249998   16.15    132.9  6.4e-16
      8x8x8x8x8  2^3          lm     -              31220     562206   16.30    314.9  3.6e-16
     8x8x53x8x8  2^3          lm     -             277362    4102312   12.44    277.5  6.7e-16
 10x10x10x10x10  2.5          lm     -             116213    1261813   10.64    285.8  5.3e-16
 12x12x12x12x12  2^2.3        lm     -             262575    2593662    8.92    339.7  6.6e-16
   16x16x16x8x8  2^4          lm     -             247075    3583169   13.54    382.0  3.4e-16
 16x16x16x16x16  2^4          lm     -            2664813   11649074    4.35    157.4  4.2e-16
   20x16x12x8x4  2^2.5        lm     -             153000    2419787   15.41    271.6  4.5e-16
  24x12x6x12x24  2^3.3        lm     -             355988    4667269   12.37    529.1  8.5e-16
 32x16x16x16x16  2^5          lm     -            7347688   28640393    3.77    119.9  4.5e-16
     53x8x8x8x8  53           lm     -             235362    4201606   15.30    327.0  6.8e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              14      0      0   3.85  12.40  16.30    9.99
 ALL                             14      0      0   3.85  12.40  16.30    9.99
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   3.77  10.06  16.58    8.28
 even column                      5      0      0   8.92  12.37  16.15   12.39
 prime column                     1      0      0  15.30  15.30  15.30   15.30
 ALL                             14      0      0   3.85  12.40  16.30    9.99
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   9      0      0   3.77  12.37  15.41    9.71
 4097..65536                      4      0      0   3.85  11.91  16.30    9.39
 257..1024                        1      0      0  16.58  16.58  16.58   16.58
 ALL                             14      0      0   3.85  12.40  16.30    9.99
```


worst 10: 32x16x16x16x16 (lm 3.77), 2x4x8x16x32 (lm 3.85), 16x16x16x16x16 (lm 4.35), 4x4x4x4x256 (lm 7.67), 12x12x12x12x12 (lm 8.92), 10x10x10x10x10 (lm 10.64), 24x12x6x12x24 (lm 12.37), 8x8x53x8x8 (lm 12.44), 16x16x16x8x8 (lm 13.54), 53x8x8x8x8 (lm 15.30)
best 5: 4x4x4x4x4 (lm 16.58), 8x8x8x8x8 (lm 16.30), 6x6x6x6x6 (lm 16.15), 20x16x12x8x4 (lm 15.41), 53x8x8x8x8 (lm 15.30)
