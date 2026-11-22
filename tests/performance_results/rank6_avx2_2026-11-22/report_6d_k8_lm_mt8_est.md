# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2c interleaved, natural, out of place, K=8 lane-major_k8_lm_mt8_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 7.779..9.638

threaded plans that ran serial (engaged = 0 at both flips): 2: 2x4x2x4x2x4, 2x4x6x8x10x12


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            lm     -               5358      79241   14.32     34.4  2.1e-16
     2x4x6x8x10x12  2            lm     -             660725    3313306    4.74     43.2  6.1e-16
       3x3x3x3x3x3  3            lm     -               6630      96070   14.39     41.8  4.6e-16
       3x5x7x7x5x3  3            lm     -              56305     803720   13.98    105.2  5.4e-16
       4x4x4x4x4x4  2^2          lm     -              18087     567060   28.16    108.7  2.6e-16
      4x4x4x4x4x53  2^2          lm     -             381125    2180987    5.57     89.6  1.2e-15
       4x4x4x8x8x8  2^2          lm     -             110350    1227444   10.02    178.2  9.5e-16
       5x5x5x5x5x5  5            lm     -              51533     587163    9.78    169.0  4.4e-16
       6x6x6x6x6x6  2.3          lm     -             131888    1440112    8.43    219.5  8.3e-16
       8x8x8x4x4x4  2^3          lm     -              83537    1117606   12.87    235.4  7.5e-16
       8x8x8x8x8x8  2^3          lm     -             996325    8743137    7.83    189.4  8.1e-16
       9x9x9x9x9x9  3^2          lm     -            4248638   48947800   11.42     95.2  8.2e-16
 10x10x10x10x10x10  2.5          lm     -            9731700  103086481   10.29     81.9  5.9e-16
     12x10x8x6x4x2  2^2.3        lm     -             154988    2662899   14.71    184.2  5.5e-16
     16x4x4x4x4x16  2^4          lm     -             140600    1754100   11.84    298.3  8.1e-16
      53x4x4x4x4x4  53           lm     -             189200    2384431    9.77    180.5  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              16      0      0   5.57  10.86  14.71   10.83
 ALL                             16      0      0   5.57  10.86  14.71   10.83
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   4.74  10.93  28.16   10.31
 prime column                     4      0      0   9.77  11.88  14.39   11.78
 even column                      3      0      0   8.43  10.29  14.71   10.85
 odd column                       1      0      0  11.42  11.42  11.42   11.42
 ALL                             16      0      0   5.57  10.86  14.71   10.83
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   5.57   9.90  14.71    9.60
 > 65536 points                   3      0      0   7.83  10.29  11.42    9.73
 257..1024                        2      0      0  14.32  14.36  14.39   14.36
 1025..4096                       1      0      0  28.16  28.16  28.16   28.16
 ALL                             16      0      0   5.57  10.86  14.71   10.83
```


worst 10: 2x4x6x8x10x12 (lm 4.74), 4x4x4x4x4x53 (lm 5.57), 8x8x8x8x8x8 (lm 7.83), 6x6x6x6x6x6 (lm 8.43), 53x4x4x4x4x4 (lm 9.77), 5x5x5x5x5x5 (lm 9.78), 4x4x4x8x8x8 (lm 10.02), 10x10x10x10x10x10 (lm 10.29), 9x9x9x9x9x9 (lm 11.42), 16x4x4x4x4x16 (lm 11.84)
best 5: 4x4x4x4x4x4 (lm 28.16), 12x10x8x6x4x2 (lm 14.71), 3x3x3x3x3x3 (lm 14.39), 2x4x2x4x2x4 (lm 14.32), 3x5x7x7x5x3 (lm 13.98)
