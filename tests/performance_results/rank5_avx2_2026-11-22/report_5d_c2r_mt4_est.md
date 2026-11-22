# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D c2r interleaved, natural, out of place, K=1_c2r_mt4_est  cells: 21 listed, 21 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 6.269..6.789

threaded plans that ran serial (engaged = 0 at both flips): 2: 2x4x8x16x32, 4x4x4x4x4


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            child  -              34825      98272    2.76     35.3  7.8e-16
     3x5x7x9x11  3            child  -              11507      47713    3.70     30.1  1.2e-15
      4x4x4x4x4  2^2          child  -               1097      10499    9.25     23.3  3.9e-16
    4x4x4x4x256  2^2          child  -              24649      91743    3.65    106.3  8.9e-16
      5x5x5x5x5  5            child  -               6213      26704    4.16     14.6  8.5e-16
      6x6x6x6x6  2.3          child  -               5231      37045    6.70     48.0  1.2e-15
      8x8x8x8x8  2^3          child  -              13169      83314    5.84     93.3  7.8e-16
     8x8x8x8x53  2^3          child  -             248928     575839    2.13     38.7  1.7e-15
     8x8x53x8x8  2^3          child  -             122850     672572    5.34     78.3  1.7e-15
      9x9x9x9x9  3^2          child  -              51381     151267    2.77     45.5  1.9e-15
 10x10x10x10x10  2.5          child  -              36223     183122    5.00    114.6  1.4e-15
 12x12x12x12x12  2^2.3        child  -              88431     342722    3.52    126.1  1.7e-15
     15x9x7x5x3  3.5          child  -              20032      93282    4.58     24.4  1.3e-15
 15x15x15x15x15  3.5          child  -             746650     769931    0.95     49.7  2.1e-15
   16x16x16x8x8  2^4          child  -              95747     431076    4.04    123.2  1.0e-15
 16x16x16x16x16  2^4          child  -             457163    1108943    2.41    114.7  1.1e-15
   20x16x12x8x4  2^2.5        child  -              58459     388328    6.51     88.8  1.0e-15
  24x12x6x12x24  2^3.3        child  -             206537     602381    2.87    114.0  1.7e-15
    32x16x8x4x2  2^5          child  -              41910     219723    4.97     29.3  6.7e-16
 32x16x16x16x16  2^5          child  -            1355100    2372000    1.66     81.2  1.2e-15
     53x8x8x8x8  53           child  -             121517     697569    5.48     79.2  1.8e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           21      0      1   2.13   4.04   6.51    3.75
 ALL                             21      0      1   2.13   4.04   6.51    3.75
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      0      0   2.13   3.85   9.25    3.71
 even column                      5      0      0   2.87   5.00   6.70    4.66
 prime column                     3      0      0   3.70   4.16   5.48    4.38
 odd column                       3      0      1   0.95   2.77   4.58    2.29
 ALL                             21      0      1   2.13   4.04   6.51    3.75
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  11      0      1   1.66   3.52   5.48    3.16
 4097..65536                      8      0      0   2.76   4.14   6.70    4.17
 257..1024                        1      0      0   9.25   9.25   9.25    9.25
 1025..4096                       1      0      0   4.16   4.16   4.16    4.16
 ALL                             21      0      1   2.13   4.04   6.51    3.75
```


worst 10: 15x15x15x15x15 (child 0.95), 32x16x16x16x16 (child 1.66), 8x8x8x8x53 (child 2.13), 16x16x16x16x16 (child 2.41), 2x4x8x16x32 (child 2.76), 9x9x9x9x9 (child 2.77), 24x12x6x12x24 (child 2.87), 12x12x12x12x12 (child 3.52), 4x4x4x4x256 (child 3.65), 3x5x7x9x11 (child 3.70)
best 5: 4x4x4x4x4 (child 9.25), 6x6x6x6x6 (child 6.70), 20x16x12x8x4 (child 6.51), 8x8x8x8x8 (child 5.84), 53x8x8x8x8 (child 5.48)
