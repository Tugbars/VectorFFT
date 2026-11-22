# gauntlet report (4D)

run: `rank4_avx2_2026-11-22`  contract: 4D c2r interleaved, natural, out of place, K=1_c2r_mt4_est  cells: 36 listed, 36 benched, comparator: MKL DFTI 4D (out of place)

control cell 16x16x16x16: 4 readings, 2.868..2.983

threaded plans that ran serial (engaged = 0 at both flips): 1: 4x4x4x4


## every shape

```
        shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x64x64x64  2            child  -             334900     749612    1.96     74.4  1.1e-15
      4x4x4x4  2^2          child  -                252       3312   12.96     20.3  3.3e-16
   4x32x32x32  2^2          child  -              38337     111865    2.83    145.3  1.1e-15
     5x7x9x11  5            child  -               4546      11518    2.44     22.4  1.0e-15
   6x12x24x96  2.3          child  -              66692     154462    2.30    107.8  1.3e-15
      8x8x8x8  2^3          child  -               2242      10947    4.80     54.8  6.7e-16
    8x8x8x256  2^3          child  -              38987      77133    1.97    142.9  1.0e-15
  8x8x128x128  2^3          child  -             394837     658818    1.61    132.8  1.3e-15
      9x9x9x9  3^2          child  -               7168      17388    2.27     29.0  1.6e-15
  10x12x14x16  2.5          child  -              12416      44213    3.37     79.6  1.2e-15
  12x12x12x12  2^2.3        child  -               8927      32829    3.36     83.3  1.2e-15
  15x15x15x15  3.5          child  -              48742      42011    0.82     40.6  1.6e-15
  16x16x16x16  2^4          child  -              18982      52849    2.75    138.1  1.0e-15
  16x16x16x32  2^4          child  -              35493      87585    2.24    156.9  9.4e-16
  16x16x16x53  2^4          child  -             240767     509541    2.09     40.0  1.6e-15
  16x16x16x64  2^4          child  -              74647     150146    1.88    158.0  1.1e-15
  16x53x16x16  2^4          child  -             104272     412689    3.61     92.3  1.9e-15
  20x20x20x20  2^2.5        child  -              88062     151933    1.72     78.5  1.1e-15
  24x24x24x24  2^3.3        child  -             136325     311029    1.74    111.6  1.4e-15
  25x25x25x25  5^2          child  -             267330     412930    1.29     67.9  1.6e-15
     27x9x9x9  3^3          child  -              18609      45202    1.62     37.7  1.7e-15
  32x32x32x32  2^5          child  -             380175     712006    1.84    137.9  1.2e-15
  32x32x32x64  2^5          child  -             827675    1368281    1.57    133.0  1.3e-15
    35x15x9x5  5.7          child  -              30741      55771    1.53     27.9  1.4e-15
  40x30x20x10  2^3.5        child  -              95106     250237    2.40    112.8  1.3e-15
  45x45x15x15  3^2.5        child  -             345062     386775    1.09     62.1  1.9e-15
  48x48x48x48  2^4.3        child  -            3801800    5862831    1.48     78.0  1.8e-15
  53x16x16x16  53           child  -              94606     413117    4.02    101.7  1.9e-15
  64x16x16x16  2^6          child  -              73027     202470    2.74    161.5  1.1e-15
   64x64x64x4  2^6          child  -             548413    1730356    3.09     95.6  1.1e-15
  64x64x64x64  2^6          child  -           13998862   31496962    2.22     71.9  1.7e-15
   96x24x12x6  2^5.3        child  -              60742     235235    3.43    118.4  1.3e-15
    101x8x8x8  101          child  -              28325     124623    4.13     71.5  1.2e-15
 127x12x12x12  127          child  -             111872     367433    3.03     87.0  2.0e-15
  128x128x8x8  2^7          child  -             439625    1310200    2.73    119.3  1.0e-15
    256x8x8x8  2^8          child  -              52507     200643    3.70    106.1  8.9e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           36      0      1   1.48   2.29   4.02    2.39
 ALL                             36      0      1   1.48   2.29   4.02    2.39
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     18      0      0   1.61   2.49   4.80    2.68
 even column                      8      0      0   1.48   2.35   3.43    2.36
 odd column                       6      0      1   0.82   1.41   2.27    1.37
 prime column                     4      0      0   2.44   3.53   4.13    3.33
 ALL                             36      0      1   1.48   2.29   4.02    2.39
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  25      0      0   1.48   2.22   3.61    2.22
 4097..65536                      8      0      1   0.82   2.51   4.13    2.22
 1025..4096                       2      0      0   2.44   3.62   4.80    3.42
 <= 256 points                    1      0      0  12.96  12.96  12.96   12.95
 ALL                             36      0      1   1.48   2.29   4.02    2.39
```


worst 10: 15x15x15x15 (child 0.82), 45x45x15x15 (child 1.09), 25x25x25x25 (child 1.29), 48x48x48x48 (child 1.48), 35x15x9x5 (child 1.53), 32x32x32x64 (child 1.57), 8x8x128x128 (child 1.61), 27x9x9x9 (child 1.62), 20x20x20x20 (child 1.72), 24x24x24x24 (child 1.74)
best 5: 4x4x4x4 (child 12.96), 8x8x8x8 (child 4.80), 101x8x8x8 (child 4.13), 53x16x16x16 (child 4.02), 256x8x8x8 (child 3.70)
