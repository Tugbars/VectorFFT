# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D c2r interleaved, natural, out of place, K=1_c2r_est  cells: 21 listed, 21 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 8.278..8.473


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            child  -              31640     196744    6.20     38.8  7.8e-16
     3x5x7x9x11  3            child  -              27868      97328    3.46     12.4  1.2e-15
      4x4x4x4x4  2^2          child  -               1116      27364   24.34     22.9  3.9e-16
    4x4x4x4x256  2^2          child  -              64970     219165    3.33     40.3  8.9e-16
      5x5x5x5x5  5            child  -              11857      54394    4.54      7.7  8.5e-16
      6x6x6x6x6  2.3          child  -               8358      91286   10.88     30.1  1.2e-15
      8x8x8x8x8  2^3          child  -              29859     250016    8.27     41.2  7.8e-16
     8x8x8x8x53  2^3          child  -             929211    2267200    2.40     10.4  1.7e-15
     8x8x53x8x8  2^3          child  -             425578    2567097    5.98     22.6  1.7e-15
      9x9x9x9x9  3^2          child  -             144112     414134    2.82     16.2  1.9e-15
 10x10x10x10x10  2.5          child  -              98949     567002    5.52     42.0  1.4e-15
 12x12x12x12x12  2^2.3        child  -             255019    1224056    4.54     43.7  1.7e-15
     15x9x7x5x3  3.5          child  -              57061     257960    4.52      8.6  1.3e-15
 15x15x15x15x15  3.5          child  -            2156100    2744156    1.27     17.2  2.1e-15
   16x16x16x8x8  2^4          child  -             359047    1609656    4.45     32.9  1.0e-15
 16x16x16x16x16  2^4          child  -            1618825    4296706    2.54     32.4  1.1e-15
   20x16x12x8x4  2^2.5        child  -             214556    1453334    6.77     24.2  1.0e-15
  24x12x6x12x24  2^3.3        child  -             678337    2279612    3.33     34.7  1.7e-15
    32x16x8x4x2  2^5          child  -             145151     786297    5.41      8.5  6.7e-16
 32x16x16x16x16  2^5          child  -            4392750    9462643    2.09     25.1  1.2e-15
     53x8x8x8x8  53           child  -             412183    2641436    6.39     23.3  1.8e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           21      0      0   2.40   4.54   8.27    4.58
 ALL                             21      0      0   2.40   4.54   8.27    4.58
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      0      0   2.40   4.93  24.34    4.88
 even column                      5      0      0   3.33   5.52  10.88    5.72
 prime column                     3      0      0   3.46   4.54   6.39    4.65
 odd column                       3      0      0   1.27   2.82   4.52    2.53
 ALL                             21      0      0   2.40   4.54   8.27    4.58
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  11      0      0   2.09   4.45   6.39    3.65
 4097..65536                      8      0      0   2.82   4.97  10.88    5.08
 257..1024                        1      0      0  24.34  24.34  24.34   24.34
 1025..4096                       1      0      0   4.54   4.54   4.54    4.54
 ALL                             21      0      0   2.40   4.54   8.27    4.58
```


worst 10: 15x15x15x15x15 (child 1.27), 32x16x16x16x16 (child 2.09), 8x8x8x8x53 (child 2.40), 16x16x16x16x16 (child 2.54), 9x9x9x9x9 (child 2.82), 24x12x6x12x24 (child 3.33), 4x4x4x4x256 (child 3.33), 3x5x7x9x11 (child 3.46), 16x16x16x8x8 (child 4.45), 15x9x7x5x3 (child 4.52)
best 5: 4x4x4x4x4 (child 24.34), 6x6x6x6x6 (child 10.88), 8x8x8x8x8 (child 8.27), 20x16x12x8x4 (child 6.77), 53x8x8x8x8 (child 6.39)
