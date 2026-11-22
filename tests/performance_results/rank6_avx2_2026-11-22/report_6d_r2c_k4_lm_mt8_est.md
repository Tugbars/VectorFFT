# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D r2c interleaved, natural, out of place, K=4 lane-major_r2c_k4_lm_mt8_est  cells: 10 listed, 10 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 17.753..19.815

threaded plans that ran serial (engaged = 0 at both flips): 2: 2x4x2x4x2x4, 2x4x6x8x10x12


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            lm     -               3026      48835   15.28     30.5  0.0e+00
     2x4x6x8x10x12  2            lm     -             164710     902540    5.23     86.7  4.8e-16
       4x4x4x4x4x4  2^2          lm     -              12681     273951   21.10     77.5  0.0e+00
       4x4x4x8x8x8  2^2          lm     -              48073     761906   15.33    204.5  2.5e-16
       6x6x6x6x6x6  2.3          lm     -              54590    1115485   19.64    265.1  7.9e-16
       8x8x8x4x4x4  2^3          lm     -              40280    1095993   26.84    244.1  2.7e-16
       8x8x8x8x8x8  2^3          lm     -             223363    4371300   18.30    422.5  3.2e-16
 10x10x10x10x10x10  2.5          lm     -            1561287   17278550   10.93    255.3  6.1e-16
     16x4x4x4x4x16  2^4          lm     -              51588    1170900   20.56    406.5  3.6e-16
      53x4x4x4x4x4  53           lm     -              83000    2479261   27.54    205.7  8.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              10      0      0  10.93  18.97  27.54   16.55
 ALL                             10      0      0  10.93  18.97  27.54   16.55
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      7      0      0   5.23  18.30  26.84   15.94
 even column                      2      0      0  10.93  15.28  19.64   14.65
 prime column                     1      0      0  27.54  27.54  27.54   27.54
 ALL                             10      0      0  10.93  18.97  27.54   16.55
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      6      0      0   5.23  20.10  27.54   16.97
 > 65536 points                   2      0      0  10.93  14.62  18.30   14.14
 257..1024                        1      0      0  15.28  15.28  15.28   15.28
 1025..4096                       1      0      0  21.10  21.10  21.10   21.10
 ALL                             10      0      0  10.93  18.97  27.54   16.55
```


worst 10: 2x4x6x8x10x12 (lm 5.23), 10x10x10x10x10x10 (lm 10.93), 2x4x2x4x2x4 (lm 15.28), 4x4x4x8x8x8 (lm 15.33), 8x8x8x8x8x8 (lm 18.30), 6x6x6x6x6x6 (lm 19.64), 16x4x4x4x4x16 (lm 20.56), 4x4x4x4x4x4 (lm 21.10), 8x8x8x4x4x4 (lm 26.84), 53x4x4x4x4x4 (lm 27.54)
best 5: 53x4x4x4x4x4 (lm 27.54), 8x8x8x4x4x4 (lm 26.84), 4x4x4x4x4x4 (lm 21.10), 16x4x4x4x4x16 (lm 20.56), 6x6x6x6x6x6 (lm 19.64)
