# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D r2c interleaved, natural, out of place, K=8 lane-major_r2c_k8_lm_mt8_est  cells: 10 listed, 10 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 15.564..17.804

threaded plans that ran serial (engaged = 0 at both flips): 2: 2x4x2x4x2x4, 2x4x6x8x10x12


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            lm     -               4182      54042   12.02     44.1  0.0e+00
     2x4x6x8x10x12  2            lm     -             344325    1490300    4.06     82.9  5.0e-16
       4x4x4x4x4x4  2^2          lm     -              22385     336179   14.62     87.8  0.0e+00
       4x4x4x8x8x8  2^2          lm     -              90500    1260637   13.53    217.2  2.7e-16
       6x6x6x6x6x6  2.3          lm     -             104875    1834568   15.94    276.0  6.4e-16
       8x8x8x4x4x4  2^3          lm     -              71650    1865469   25.20    274.4  2.7e-16
       8x8x8x8x8x8  2^3          lm     -             443163    7624356   16.51    425.9  3.0e-16
 10x10x10x10x10x10  2.5          lm     -            5345050   39621463    7.34    149.2  6.1e-16
     16x4x4x4x4x16  2^4          lm     -             100600    1883068   17.58    416.9  3.3e-16
      53x4x4x4x4x4  53           lm     -             149863    3598243   18.66    227.8  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              10      0      0   7.34  15.28  25.20   13.14
 ALL                             10      0      0   7.34  15.28  25.20   13.14
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      7      0      0   4.06  14.62  25.20   13.22
 even column                      2      0      0   7.34  11.64  15.94   10.81
 prime column                     1      0      0  18.66  18.66  18.66   18.66
 ALL                             10      0      0   7.34  15.28  25.20   13.14
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      6      0      0   4.06  16.76  25.20   13.91
 > 65536 points                   2      0      0   7.34  11.92  16.51   11.01
 257..1024                        1      0      0  12.02  12.02  12.02   12.02
 1025..4096                       1      0      0  14.62  14.62  14.62   14.62
 ALL                             10      0      0   7.34  15.28  25.20   13.14
```


worst 10: 2x4x6x8x10x12 (lm 4.06), 10x10x10x10x10x10 (lm 7.34), 2x4x2x4x2x4 (lm 12.02), 4x4x4x8x8x8 (lm 13.53), 4x4x4x4x4x4 (lm 14.62), 6x6x6x6x6x6 (lm 15.94), 8x8x8x8x8x8 (lm 16.51), 16x4x4x4x4x16 (lm 17.58), 53x4x4x4x4x4 (lm 18.66), 8x8x8x4x4x4 (lm 25.20)
best 5: 8x8x8x4x4x4 (lm 25.20), 53x4x4x4x4x4 (lm 18.66), 16x4x4x4x4x16 (lm 17.58), 8x8x8x8x8x8 (lm 16.51), 6x6x6x6x6x6 (lm 15.94)
