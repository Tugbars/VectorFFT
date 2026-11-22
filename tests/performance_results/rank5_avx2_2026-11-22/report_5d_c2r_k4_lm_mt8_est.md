# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D c2r interleaved, natural, out of place, K=4 lane-major_c2r_k4_lm_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 12.088..16.478

threaded plans that ran serial (engaged = 0 at both flips): 1: 2x4x8x16x32


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            lm     -             145020     285130    1.72     67.8  7.8e-16
      4x4x4x4x4  2^2          lm     -               2762      39432   13.22     74.1  4.4e-16
    4x4x4x4x256  2^2          lm     -              80462     419450    3.95    260.6  8.9e-16
      6x6x6x6x6  2.3          lm     -              10261     161565   11.65    195.9  1.3e-15
      8x8x8x8x8  2^3          lm     -              28300     409480   12.05    347.4  7.8e-16
     8x8x53x8x8  2^3          lm     -             299137    3239719   10.23    257.3  1.9e-15
 10x10x10x10x10  2.5          lm     -             107738     989731    8.40    308.3  1.5e-15
 12x12x12x12x12  2^2.3        lm     -             249912    1883437    6.59    356.9  1.9e-15
   16x16x16x8x8  2^4          lm     -             266500    2448131    8.92    354.1  7.8e-16
 16x16x16x16x16  2^4          lm     -            3998338    8682262    2.16    104.9  8.9e-16
   20x16x12x8x4  2^2.5        lm     -             157625    1803281   10.97    263.6  1.0e-15
  24x12x6x12x24  2^3.3        lm     -             535512    3588568    6.02    351.7  2.0e-15
 32x16x16x16x16  2^5          lm     -            8085925   17795750    2.18    108.9  1.0e-15
     53x8x8x8x8  53           lm     -             257050    3274143   10.66    299.4  2.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              14      0      0   2.16   8.66  12.05    6.44
 ALL                             14      0      0   2.16   8.66  12.05    6.44
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   1.72   6.43  13.22    5.11
 even column                      5      0      0   6.02   8.40  11.65    8.43
 prime column                     1      0      0  10.66  10.66  10.66   10.66
 ALL                             14      0      0   2.16   8.66  12.05    6.44
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   9      0      0   2.16   8.40  10.97    6.35
 4097..65536                      4      0      0   1.72   7.80  12.05    5.56
 257..1024                        1      0      0  13.22  13.22  13.22   13.22
 ALL                             14      0      0   2.16   8.66  12.05    6.44
```


worst 10: 2x4x8x16x32 (lm 1.72), 16x16x16x16x16 (lm 2.16), 32x16x16x16x16 (lm 2.18), 4x4x4x4x256 (lm 3.95), 24x12x6x12x24 (lm 6.02), 12x12x12x12x12 (lm 6.59), 10x10x10x10x10 (lm 8.40), 16x16x16x8x8 (lm 8.92), 8x8x53x8x8 (lm 10.23), 53x8x8x8x8 (lm 10.66)
best 5: 4x4x4x4x4 (lm 13.22), 8x8x8x8x8 (lm 12.05), 6x6x6x6x6 (lm 11.65), 20x16x12x8x4 (lm 10.97), 53x8x8x8x8 (lm 10.66)
