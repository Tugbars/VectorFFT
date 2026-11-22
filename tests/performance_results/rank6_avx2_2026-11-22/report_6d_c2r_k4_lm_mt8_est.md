# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2r interleaved, natural, out of place, K=4 lane-major_c2r_k4_lm_mt8_est  cells: 10 listed, 10 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 11.700..13.730

threaded plans that ran serial (engaged = 0 at both flips): 2: 2x4x2x4x2x4, 2x4x6x8x10x12


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            lm     -               3975      34906    8.48     23.2  4.4e-16
     2x4x6x8x10x12  2            lm     -             260660     785680    2.64     54.8  1.1e-15
       4x4x4x4x4x4  2^2          lm     -               9926     165966   16.25     99.0  4.4e-16
       4x4x4x8x8x8  2^2          lm     -              48093     639470   11.40    204.4  6.7e-16
       6x6x6x6x6x6  2.3          lm     -              53390     800845   12.81    271.1  1.7e-15
       8x8x8x4x4x4  2^3          lm     -              38360     890740   19.84    256.3  6.7e-16
       8x8x8x8x8x8  2^3          lm     -             284413    3427631   11.98    331.8  8.3e-16
 10x10x10x10x10x10  2.5          lm     -            3189737   11463550    3.56    125.0  1.6e-15
     16x4x4x4x4x16  2^4          lm     -              51575     881175   14.31    406.6  7.8e-16
      53x4x4x4x4x4  53           lm     -              82822    2006800   20.33    206.1  1.7e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              10      0      0   3.56  12.40  20.33   10.27
 ALL                             10      0      0   3.56  12.40  20.33   10.27
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      7      0      0   2.64  11.98  19.84   10.51
 even column                      2      0      0   3.56   8.19  12.81    6.76
 prime column                     1      0      0  20.33  20.33  20.33   20.33
 ALL                             10      0      0   3.56  12.40  20.33   10.27
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      6      0      0   2.64  13.56  20.33   11.43
 > 65536 points                   2      0      0   3.56   7.77  11.98    6.54
 257..1024                        1      0      0   8.48   8.48   8.48    8.48
 1025..4096                       1      0      0  16.25  16.25  16.25   16.25
 ALL                             10      0      0   3.56  12.40  20.33   10.27
```


worst 10: 2x4x6x8x10x12 (lm 2.64), 10x10x10x10x10x10 (lm 3.56), 2x4x2x4x2x4 (lm 8.48), 4x4x4x8x8x8 (lm 11.40), 8x8x8x8x8x8 (lm 11.98), 6x6x6x6x6x6 (lm 12.81), 16x4x4x4x4x16 (lm 14.31), 4x4x4x4x4x4 (lm 16.25), 8x8x8x4x4x4 (lm 19.84), 53x4x4x4x4x4 (lm 20.33)
best 5: 53x4x4x4x4x4 (lm 20.33), 8x8x8x4x4x4 (lm 19.84), 4x4x4x4x4x4 (lm 16.25), 16x4x4x4x4x16 (lm 14.31), 6x6x6x6x6x6 (lm 12.81)
