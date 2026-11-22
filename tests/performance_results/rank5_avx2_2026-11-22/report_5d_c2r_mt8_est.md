# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D c2r interleaved, natural, out of place, K=1_c2r_mt8_est  cells: 21 listed, 21 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 5.586..9.517

threaded plans that ran serial (engaged = 0 at both flips): 2: 2x4x8x16x32, 4x4x4x4x4


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            child  -              30131     101078    3.19     40.8  7.8e-16
     3x5x7x9x11  3            child  -              10553      38607    3.44     32.9  1.2e-15
      4x4x4x4x4  2^2          child  -               1122      10552    9.19     22.8  3.9e-16
    4x4x4x4x256  2^2          child  -              23133      80061    2.81    113.3  8.9e-16
      5x5x5x5x5  5            child  -               4054      16581    4.05     22.4  8.5e-16
      6x6x6x6x6  2.3          child  -               3457      22697    6.26     72.7  1.2e-15
      8x8x8x8x8  2^3          child  -               7412      69836    6.94    165.8  7.8e-16
     8x8x8x8x53  2^3          child  -             132606     390141    2.05     72.6  1.7e-15
     8x8x53x8x8  2^3          child  -              71494     431205    4.40    134.6  1.7e-15
      9x9x9x9x9  3^2          child  -              36316     138116    3.56     64.4  1.9e-15
 10x10x10x10x10  2.5          child  -              25538     154565    4.29    162.6  1.4e-15
 12x12x12x12x12  2^2.3        child  -              55644     277569    3.72    200.4  1.7e-15
     15x9x7x5x3  3.5          child  -              12413      61981    4.15     39.4  1.3e-15
 15x15x15x15x15  3.5          child  -             424613     588869    1.00     87.3  2.1e-15
   16x16x16x8x8  2^4          child  -              54633     339590    3.86    215.9  1.0e-15
 16x16x16x16x16  2^4          child  -             230450     974581    3.26    227.5  1.1e-15
   20x16x12x8x4  2^2.5        child  -              37091     341225    8.43    140.0  1.0e-15
  24x12x6x12x24  2^3.3        child  -             109912     426756    2.86    214.2  1.7e-15
    32x16x8x4x2  2^5          child  -              28827     122262    3.63     42.6  6.7e-16
 32x16x16x16x16  2^5          child  -             772487    1487956    1.61    142.5  1.2e-15
     53x8x8x8x8  53           child  -              59217     510394    5.34    162.5  1.8e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           21      0      0   2.05   3.72   6.94    3.73
 ALL                             21      0      0   2.05   3.72   6.94    3.73
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      0      0   2.05   3.44   9.19    3.61
 even column                      5      0      0   2.86   4.29   8.43    4.75
 prime column                     3      0      0   3.44   4.05   5.34    4.20
 odd column                       3      0      0   1.00   3.56   4.15    2.45
 ALL                             21      0      0   2.05   3.72   6.94    3.73
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  11      0      0   1.61   3.72   5.34    3.21
 4097..65536                      8      0      0   2.81   3.60   6.94    4.04
 257..1024                        1      0      0   9.19   9.19   9.19    9.19
 1025..4096                       1      0      0   4.05   4.05   4.05    4.05
 ALL                             21      0      0   2.05   3.72   6.94    3.73
```


worst 10: 15x15x15x15x15 (child 1.00), 32x16x16x16x16 (child 1.61), 8x8x8x8x53 (child 2.05), 4x4x4x4x256 (child 2.81), 24x12x6x12x24 (child 2.86), 2x4x8x16x32 (child 3.19), 16x16x16x16x16 (child 3.26), 3x5x7x9x11 (child 3.44), 9x9x9x9x9 (child 3.56), 32x16x8x4x2 (child 3.63)
best 5: 4x4x4x4x4 (child 9.19), 20x16x12x8x4 (child 8.43), 8x8x8x8x8 (child 6.94), 6x6x6x6x6 (child 6.26), 53x8x8x8x8 (child 5.34)
