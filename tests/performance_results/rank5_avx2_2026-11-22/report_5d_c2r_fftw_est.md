# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D c2r interleaved, natural, out of place, K=1_c2r_fftw_est  cells: 21 listed, 21 benched, comparator: FFTW 5D (out of place, MEASURE)

control cell 8x8x8x8x8: 4 readings, 0.980..1.013


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            child  -              31671      30772    0.95     38.8  7.8e-16
     3x5x7x9x11  3            child  -              27933      11271    0.40     12.4  1.1e-15
      4x4x4x4x4  2^2          child  -               1116       1006    0.90     22.9  3.9e-16
    4x4x4x4x256  2^2          child  -              65334      60623    0.92     40.1  7.8e-16
      5x5x5x5x5  5            child  -              11894       3133    0.26      7.6  7.8e-16
      6x6x6x6x6  2.3          child  -               8497       7921    0.91     29.6  1.1e-15
      8x8x8x8x8  2^3          child  -              29441      29520    0.99     41.7  8.9e-16
     8x8x8x8x53  2^3          child  -             911233    1839680    2.01     10.6  1.6e-15
     8x8x53x8x8  2^3          child  -             447689     914994    2.04     21.5  1.6e-15
      9x9x9x9x9  3^2          child  -             139704      65803    0.47     16.7  1.7e-15
 10x10x10x10x10  2.5          child  -             103138      93770    0.91     40.3  1.2e-15
 12x12x12x12x12  2^2.3        child  -             274731     271965    0.99     40.6  1.8e-15
     15x9x7x5x3  3.5          child  -              57106      17553    0.30      8.6  1.2e-15
 15x15x15x15x15  3.5          child  -            2348225     979100    0.39     15.8  1.8e-15
   16x16x16x8x8  2^4          child  -             370187     386140    1.03     31.9  1.0e-15
 16x16x16x16x16  2^4          child  -            1906538    1854462    0.96     27.5  1.1e-15
   20x16x12x8x4  2^2.5        child  -             217891     192576    0.87     23.8  1.1e-15
  24x12x6x12x24  2^3.3        child  -             713650     958556    1.33     33.0  1.9e-15
    32x16x8x4x2  2^5          child  -             151681      66007    0.43      8.1  8.9e-16
 32x16x16x16x16  2^5          child  -            5040275    4716787    0.92     21.8  1.2e-15
     53x8x8x8x8  53           child  -             422250     988875    2.31     22.8  1.8e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           21      6     16   0.39   0.92   2.01    0.82
 ALL                             21      6     16   0.39   0.92   2.01    0.82
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      1      7   0.90   0.96   2.04    1.02
 even column                      5      0      4   0.87   0.91   1.33    0.99
 prime column                     3      2      2   0.26   0.40   2.31    0.62
 odd column                       3      3      3   0.30   0.39   0.47    0.38
 ALL                             21      6     16   0.39   0.92   2.01    0.82
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  11      1      6   0.87   0.99   2.04    1.12
 4097..65536                      8      4      8   0.30   0.69   0.99    0.61
 257..1024                        1      0      1   0.90   0.90   0.90    0.90
 1025..4096                       1      1      1   0.26   0.26   0.26    0.26
 ALL                             21      6     16   0.39   0.92   2.01    0.82
```


worst 10: 5x5x5x5x5 (child 0.26), 15x9x7x5x3 (child 0.30), 15x15x15x15x15 (child 0.39), 3x5x7x9x11 (child 0.40), 32x16x8x4x2 (child 0.43), 9x9x9x9x9 (child 0.47), 20x16x12x8x4 (child 0.87), 4x4x4x4x4 (child 0.90), 6x6x6x6x6 (child 0.91), 10x10x10x10x10 (child 0.91)
best 5: 53x8x8x8x8 (child 2.31), 8x8x53x8x8 (child 2.04), 8x8x8x8x53 (child 2.01), 24x12x6x12x24 (child 1.33), 16x16x16x8x8 (child 1.03)
