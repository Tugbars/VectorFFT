# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D c2c interleaved, natural, out of place, K=4 lane-major_k4_lm_est  cells: 21 listed, 21 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 7.720..7.997


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            lm     -             197840    1380437    6.93     49.7  7.2e-16
     3x5x7x9x11  3            lm     -              61913     424413    6.85     44.8  5.2e-16
      4x4x4x4x4  2^2          lm     -               5901      84134   14.08     34.7  1.9e-16
    4x4x4x4x256  2^2          lm     -             440313    2864018    6.42     47.6  8.4e-16
      5x5x5x5x5  5            lm     -              18742     185186    9.76     38.7  4.0e-16
      6x6x6x6x6  2.3          lm     -              44684     389870    8.64     45.0  8.6e-16
      8x8x8x8x8  2^3          lm     -             188573    1512357    7.88     52.1  3.1e-16
     8x8x8x8x53  2^3          lm     -            3519062   11069925    3.08     21.9  1.2e-15
     8x8x53x8x8  2^3          lm     -            2797488   12765119    4.50     27.5  1.3e-15
      9x9x9x9x9  3^2          lm     -             383987    2794575    7.26     48.7  6.9e-16
 10x10x10x10x10  2.5          lm     -             643700    4595737    7.10     51.6  5.1e-16
 12x12x12x12x12  2^2.3        lm     -            2052450   12404631    5.99     43.5  9.3e-16
     15x9x7x5x3  3.5          lm     -             114634    1302713   11.13     34.1  7.1e-16
 15x15x15x15x15  3.5          lm     -            9673062   61553399    6.34     30.7  6.5e-16
   16x16x16x8x8  2^4          lm     -            4272450   15327087    3.49     22.1  8.4e-16
 16x16x16x16x16  2^4          lm     -           22800188   77318225    3.38     18.4  1.7e-15
   20x16x12x8x4  2^2.5        lm     -            1364887    8919681    6.48     30.4  4.7e-16
  24x12x6x12x24  2^3.3        lm     -            7508337   35428994    4.71     25.1  8.1e-16
    32x16x8x4x2  2^5          lm     -             432873    4668000   10.66     22.7  3.3e-16
 32x16x16x16x16  2^5          lm     -           46898638  173506874    3.67     18.8  1.2e-15
     53x8x8x8x8  53           lm     -            3011600   14703219    4.86     25.6  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              21      0      0   3.49   6.48  10.66    6.29
 ALL                             21      0      0   3.49   6.48  10.66    6.29
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      0      0   3.38   5.46  14.08    5.62
 even column                      5      0      0   4.71   6.48   8.64    6.45
 prime column                     3      0      0   4.86   6.85   9.76    6.88
 odd column                       3      0      0   6.34   7.26  11.13    8.00
 ALL                             21      0      0   3.49   6.48  10.66    6.29
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  11      0      0   3.38   4.71   6.48    4.69
 4097..65536                      8      0      0   6.42   7.57  11.13    8.07
 257..1024                        1      0      0  14.08  14.08  14.08   14.08
 1025..4096                       1      0      0   9.76   9.76   9.76    9.76
 ALL                             21      0      0   3.49   6.48  10.66    6.29
```


worst 10: 8x8x8x8x53 (lm 3.08), 16x16x16x16x16 (lm 3.38), 16x16x16x8x8 (lm 3.49), 32x16x16x16x16 (lm 3.67), 8x8x53x8x8 (lm 4.50), 24x12x6x12x24 (lm 4.71), 53x8x8x8x8 (lm 4.86), 12x12x12x12x12 (lm 5.99), 15x15x15x15x15 (lm 6.34), 4x4x4x4x256 (lm 6.42)
best 5: 4x4x4x4x4 (lm 14.08), 15x9x7x5x3 (lm 11.13), 32x16x8x4x2 (lm 10.66), 5x5x5x5x5 (lm 9.76), 6x6x6x6x6 (lm 8.64)
