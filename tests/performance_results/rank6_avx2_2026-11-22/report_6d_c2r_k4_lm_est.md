# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2r interleaved, natural, out of place, K=4 lane-major_c2r_k4_lm_est  cells: 10 listed, 10 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 6.846..7.390


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            lm     -               4155     114075   27.38     22.2  4.4e-16
     2x4x6x8x10x12  2            lm     -             262540    2688960    9.73     54.4  1.1e-15
       4x4x4x4x4x4  2^2          lm     -              24799     562712   22.48     39.6  4.4e-16
       4x4x4x8x8x8  2^2          lm     -             175920    2150246   11.95     55.9  6.7e-16
       6x6x6x6x6x6  2.3          lm     -             266500    2980515   11.13     54.3  1.7e-15
       8x8x8x4x4x4  2^3          lm     -             231633    3436293   13.53     42.4  6.7e-16
       8x8x8x8x8x8  2^3          lm     -            1674425   12372087    7.08     56.4  8.3e-16
 10x10x10x10x10x10  2.5          lm     -            9592650   43783525    4.52     41.6  1.6e-15
     16x4x4x4x4x16  2^4          lm     -             349975    2772274    7.90     59.9  7.8e-16
      53x4x4x4x4x4  53           lm     -             589367    7724172   13.08     29.0  1.7e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              10      0      0   7.08  11.54  27.38   11.36
 ALL                             10      0      0   7.08  11.54  27.38   11.36
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      7      0      0   7.08  11.95  27.38   12.73
 even column                      2      0      0   4.52   7.83  11.13    7.09
 prime column                     1      0      0  13.08  13.08  13.08   13.08
 ALL                             10      0      0   7.08  11.54  27.38   11.36
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      6      0      0   7.90  11.54  13.53   11.04
 > 65536 points                   2      0      0   4.52   5.80   7.08    5.66
 257..1024                        1      0      0  27.38  27.38  27.38   27.38
 1025..4096                       1      0      0  22.48  22.48  22.48   22.48
 ALL                             10      0      0   7.08  11.54  27.38   11.36
```


worst 10: 10x10x10x10x10x10 (lm 4.52), 8x8x8x8x8x8 (lm 7.08), 16x4x4x4x4x16 (lm 7.90), 2x4x6x8x10x12 (lm 9.73), 6x6x6x6x6x6 (lm 11.13), 4x4x4x8x8x8 (lm 11.95), 53x4x4x4x4x4 (lm 13.08), 8x8x8x4x4x4 (lm 13.53), 4x4x4x4x4x4 (lm 22.48), 2x4x2x4x2x4 (lm 27.38)
best 5: 2x4x2x4x2x4 (lm 27.38), 4x4x4x4x4x4 (lm 22.48), 8x8x8x4x4x4 (lm 13.53), 53x4x4x4x4x4 (lm 13.08), 4x4x4x8x8x8 (lm 11.95)
