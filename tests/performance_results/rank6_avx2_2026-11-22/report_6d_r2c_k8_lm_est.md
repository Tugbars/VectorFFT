# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D r2c interleaved, natural, out of place, K=8 lane-major_r2c_k8_lm_est  cells: 10 listed, 10 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 11.221..12.767


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            lm     -               4420     233588   52.04     41.7  0.0e+00
     2x4x6x8x10x12  2            lm     -             368600    6622706   17.79     77.5  5.0e-16
       4x4x4x4x4x4  2^2          lm     -              31533    1238445   39.24     62.4  0.0e+00
       4x4x4x8x8x8  2^2          lm     -             250900    5089806   19.98     78.4  2.7e-16
       6x6x6x6x6x6  2.3          lm     -             420188    7391300   17.23     68.9  6.4e-16
       8x8x8x4x4x4  2^3          lm     -             289750    7875056   26.75     67.9  2.7e-16
       8x8x8x8x8x8  2^3          lm     -            2956087   37350125   12.02     63.8  3.0e-16
 10x10x10x10x10x10  2.5          lm     -           17014012  211688856   12.40     46.9  6.1e-16
     16x4x4x4x4x16  2^4          lm     -             584825    8057631   13.76     71.7  3.3e-16
      53x4x4x4x4x4  53           lm     -             957563   18031062   18.67     35.7  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              10      0      0  12.40  18.23  52.04   20.46
 ALL                             10      0      0  12.40  18.23  52.04   20.46
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      7      0      0  12.02  19.98  52.04   22.81
 even column                      2      0      0  12.40  14.82  17.23   14.62
 prime column                     1      0      0  18.67  18.67  18.67   18.67
 ALL                             10      0      0  12.40  18.23  52.04   20.46
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      6      0      0  13.76  18.23  26.75   18.65
 > 65536 points                   2      0      0  12.02  12.21  12.40   12.21
 257..1024                        1      0      0  52.04  52.04  52.04   52.04
 1025..4096                       1      0      0  39.24  39.24  39.24   39.24
 ALL                             10      0      0  12.40  18.23  52.04   20.46
```


worst 10: 8x8x8x8x8x8 (lm 12.02), 10x10x10x10x10x10 (lm 12.40), 16x4x4x4x4x16 (lm 13.76), 6x6x6x6x6x6 (lm 17.23), 2x4x6x8x10x12 (lm 17.79), 53x4x4x4x4x4 (lm 18.67), 4x4x4x8x8x8 (lm 19.98), 8x8x8x4x4x4 (lm 26.75), 4x4x4x4x4x4 (lm 39.24), 2x4x2x4x2x4 (lm 52.04)
best 5: 2x4x2x4x2x4 (lm 52.04), 4x4x4x4x4x4 (lm 39.24), 8x8x8x4x4x4 (lm 26.75), 4x4x4x8x8x8 (lm 19.98), 53x4x4x4x4x4 (lm 18.67)
