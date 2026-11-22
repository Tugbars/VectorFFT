# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D c2c interleaved, natural, out of place, K=4 lane-major_k4_lm_mt8_est  cells: 21 listed, 21 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 5.625..8.869

threaded plans that ran serial (engaged = 0 at both flips): 1: 2x4x8x16x32


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            lm     -             203393     804906    3.84     48.3  7.2e-16
     3x5x7x9x11  3            lm     -              28135     221755    7.40     98.6  5.2e-16
      4x4x4x4x4  2^2          lm     -               4239      88605   17.97     48.3  1.9e-16
    4x4x4x4x256  2^2          lm     -             125850     959881    6.27    166.6  8.4e-16
      5x5x5x5x5  5            lm     -               8281     190748   19.37     87.6  4.0e-16
      6x6x6x6x6  2.3          lm     -              16459     412272   22.55    122.1  8.6e-16
      8x8x8x8x8  2^3          lm     -              43847     273773    5.55    224.2  3.1e-16
     8x8x8x8x53  2^3          lm     -             479400    1469450    2.39    160.6  1.2e-15
     8x8x53x8x8  2^3          lm     -             389637    1780212    3.93    197.5  1.3e-15
      9x9x9x9x9  3^2          lm     -             103875     766619    6.85    180.2  6.9e-16
 10x10x10x10x10  2.5          lm     -             153662    1083250    7.03    216.2  5.1e-16
 12x12x12x12x12  2^2.3        lm     -             365838    2223056    4.95    243.8  9.3e-16
     15x9x7x5x3  3.5          lm     -              26843     329912   10.85    145.7  7.1e-16
 15x15x15x15x15  3.5          lm     -            2434738   10903681    4.42    121.9  6.5e-16
   16x16x16x8x8  2^4          lm     -             425888    2169475    4.28    221.6  8.4e-16
 16x16x16x16x16  2^4          lm     -            7688350   20932550    2.71     54.6  1.7e-15
   20x16x12x8x4  2^2.5        lm     -             192563    1456512    5.72    215.8  4.7e-16
  24x12x6x12x24  2^3.3        lm     -             842900    4704475    4.81    223.5  8.1e-16
    32x16x8x4x2  2^5          lm     -              58413     630913    8.64    168.3  3.3e-16
 32x16x16x16x16  2^5          lm     -           16347837   60200594    3.34     53.9  1.2e-15
     53x8x8x8x8  53           lm     -             406075    2148318    4.56    189.5  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              21      0      0   3.34   5.55  17.97    6.13
 ALL                             21      0      0   3.34   5.55  17.97    6.13
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      0      0   2.71   4.10  17.97    4.87
 even column                      5      0      0   4.81   5.72  22.55    7.36
 prime column                     3      0      0   4.56   7.40  19.37    8.68
 odd column                       3      0      0   4.42   6.85  10.85    6.90
 ALL                             21      0      0   3.34   5.55  17.97    6.13
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  11      0      0   2.71   4.42   5.72    4.19
 4097..65536                      8      0      0   3.84   7.13  22.55    7.84
 257..1024                        1      0      0  17.97  17.97  17.97   17.97
 1025..4096                       1      0      0  19.37  19.37  19.37   19.37
 ALL                             21      0      0   3.34   5.55  17.97    6.13
```


worst 10: 8x8x8x8x53 (lm 2.39), 16x16x16x16x16 (lm 2.71), 32x16x16x16x16 (lm 3.34), 2x4x8x16x32 (lm 3.84), 8x8x53x8x8 (lm 3.93), 16x16x16x8x8 (lm 4.28), 15x15x15x15x15 (lm 4.42), 53x8x8x8x8 (lm 4.56), 24x12x6x12x24 (lm 4.81), 12x12x12x12x12 (lm 4.95)
best 5: 6x6x6x6x6 (lm 22.55), 5x5x5x5x5 (lm 19.37), 4x4x4x4x4 (lm 17.97), 15x9x7x5x3 (lm 10.85), 32x16x8x4x2 (lm 8.64)
