# gauntlet report (4D)

run: `rank4_avx2_2026-11-22`  contract: 4D r2c interleaved, natural, out of place, K=1_r2c_mt4_est  cells: 36 listed, 36 benched, comparator: MKL DFTI 4D (out of place)

control cell 16x16x16x16: 4 readings, 2.750..3.308

threaded plans that ran serial (engaged = 0 at both flips): 1: 4x4x4x4


## every shape

```
        shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x64x64x64  2            child  -             261375     666037    2.53     95.3  4.6e-16
      4x4x4x4  2^2          child  -                246       3440   13.76     20.8  0.0e+00
   4x32x32x32  2^2          child  -              43730     101183    2.26    127.4  3.3e-16
     5x7x9x11  5            child  -               5055      11899    2.34     20.2  4.6e-16
   6x12x24x96  2.3          child  -              70996     147400    2.04    101.3  5.1e-16
      8x8x8x8  2^3          child  -               3360      10616    2.87     36.6  2.1e-16
    8x8x8x256  2^3          child  -              38363      71466    1.70    145.2  4.2e-16
  8x8x128x128  2^3          child  -             369275     576275    1.40    142.0  4.2e-16
      9x9x9x9  3^2          child  -               7894      19065    2.35     26.3  5.5e-16
  10x12x14x16  2.5          child  -              15109      42427    2.67     65.4  4.9e-16
  12x12x12x12  2^2.3        child  -               8652      34014    3.52     85.9  8.5e-16
  15x15x15x15  3.5          child  -              37827      56466    1.31     52.3  5.5e-16
  16x16x16x16  2^4          child  -              18951      72246    3.15    138.3  3.2e-16
  16x16x16x32  2^4          child  -              40580      86595    2.05    137.3  3.1e-16
  16x16x16x53  2^4          child  -             162044     514275    2.68     59.4  6.8e-16
  16x16x16x64  2^4          child  -              78193     162203    1.66    150.9  3.6e-16
  16x53x16x16  2^4          child  -             110411     429053    3.82     87.1  7.1e-16
  20x20x20x20  2^2.5        child  -              85421     169869    1.81     81.0  4.5e-16
  24x24x24x24  2^3.3        child  -             142992     317070    2.10    106.4  6.3e-16
  25x25x25x25  5^2          child  -             251360     409825    1.33     72.2  5.5e-16
     27x9x9x9  3^3          child  -              18667      40763    1.91     37.6  6.9e-16
  32x32x32x32  2^5          child  -             368562     629418    1.61    142.3  4.4e-16
  32x32x32x64  2^5          child  -             738012    1301181    1.67    149.2  4.9e-16
    35x15x9x5  5.7          child  -              22718      63831    2.52     37.8  5.6e-16
  40x30x20x10  2^3.5        child  -              91512     249519    2.56    117.2  4.4e-16
  45x45x15x15  3^2.5        child  -             252813     409206    1.60     84.7  6.8e-16
  48x48x48x48  2^4.3        child  -            3236938    5083481    1.50     91.6  6.7e-16
  53x16x16x16  53           child  -              99583     427089    4.20     96.6  6.5e-16
  64x16x16x16  2^6          child  -              73580     214093    2.70    160.3  3.8e-16
   64x64x64x4  2^6          child  -             762025    1858737    2.24     68.8  4.3e-16
  64x64x64x64  2^6          child  -           15837725   18006993    1.06     63.6  4.4e-16
   96x24x12x6  2^5.3        child  -              72096     254800    3.30     99.7  6.0e-16
    101x8x8x8  101          child  -              32381     137333    4.19     62.5  1.4e-15
 127x12x12x12  127          child  -             106961     392983    3.51     91.0  7.5e-16
  128x128x8x8  2^7          child  -             513975    1615550    3.08    102.0  4.7e-16
    256x8x8x8  2^8          child  -              61213     242153    3.91     91.0  3.6e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           36      0      0   1.40   2.35   3.91    2.40
 ALL                             36      0      0   1.40   2.35   3.91    2.40
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     18      0      0   1.40   2.40   3.91    2.48
 even column                      8      0      0   1.50   2.33   3.52    2.35
 odd column                       6      0      0   1.31   1.75   2.52    1.78
 prime column                     4      0      0   2.34   3.85   4.20    3.47
 ALL                             36      0      0   1.40   2.35   3.91    2.40
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  25      0      0   1.40   2.10   3.82    2.18
 4097..65536                      8      0      0   1.31   2.60   4.19    2.56
 1025..4096                       2      0      0   2.34   2.60   2.87    2.59
 <= 256 points                    1      0      0  13.76  13.76  13.76   13.76
 ALL                             36      0      0   1.40   2.35   3.91    2.40
```


worst 10: 64x64x64x64 (child 1.06), 15x15x15x15 (child 1.31), 25x25x25x25 (child 1.33), 8x8x128x128 (child 1.40), 48x48x48x48 (child 1.50), 45x45x15x15 (child 1.60), 32x32x32x32 (child 1.61), 16x16x16x64 (child 1.66), 32x32x32x64 (child 1.67), 8x8x8x256 (child 1.70)
best 5: 4x4x4x4 (child 13.76), 53x16x16x16 (child 4.20), 101x8x8x8 (child 4.19), 256x8x8x8 (child 3.91), 16x53x16x16 (child 3.82)
