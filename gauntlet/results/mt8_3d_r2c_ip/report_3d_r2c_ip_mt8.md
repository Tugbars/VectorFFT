# gauntlet report (3D)

run: `mt8_3d_r2c_ip`  contract: 3D r2c interleaved, natural, out of place, K=1_r2c_ip_mt8  cells: 11 listed, 11 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 0.677..0.717

threaded plans that ran serial (engaged = 0 at both flips): 0


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  8x128x2048  2^3          plane+child raced         319288     552594    1.69    344.8  5.1e-16
     9x16x30  3^2          plane+child+strips raced           2784       2940    0.88     46.8  5.3e-16
    16x16x16  2^4          plane+payonce+strips raced           3228       2456    0.69     38.1  2.6e-16
  16x64x1024  2^4          plane+child raced         166287     212350    1.19    315.3  5.7e-16
  16x256x256  2^4          plane+payonce+strips raced         250287     232969    0.87    209.5  4.8e-16
   32x8x1000  2^5          plane+child raced          43700      67370    1.28    263.1  4.1e-16
    32x32x32  2^5          plane+child raced          11583      11301    0.95    106.1  3.9e-16
  32x256x256  2^5          plane+child+strips raced         452075     474000    0.99    243.5  6.1e-16
    64x64x64  2^6          plane+child+strips raced          69207      50323    0.69    170.5  3.4e-16
  64x128x128  2^6          plane+payonce raced         230025     181368    0.70    227.9  4.5e-16
 128x128x128  2^7          plane+child+strips raced         411888     384031    0.89    267.3  4.2e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 plane+child                      4      0      1   0.95   1.24   1.69    1.25
 plane+child+strips               4      1      4   0.69   0.89   0.99    0.86
 plane+payonce+strips             2      1      2   0.69   0.78   0.87    0.77
 plane+payonce                    1      1      1   0.70   0.70   0.70    0.70
 ALL                             11      3      8   0.69   0.89   1.28    0.95
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      3      7   0.69   0.92   1.69    0.95
 odd column                       1      0      1   0.88   0.88   0.88    0.88
 ALL                             11      3      8   0.69   0.89   1.28    0.95
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   8      2      5   0.69   0.94   1.69    0.99
 4097..65536                      2      0      2   0.88   0.91   0.95    0.91
 1025..4096                       1      1      1   0.69   0.69   0.69    0.69
 ALL                             11      3      8   0.69   0.89   1.28    0.95
```


worst 10: 16x16x16 (plane+payonce+strips 0.69), 64x64x64 (plane+child+strips 0.69), 64x128x128 (plane+payonce 0.70), 16x256x256 (plane+payonce+strips 0.87), 9x16x30 (plane+child+strips 0.88), 128x128x128 (plane+child+strips 0.89), 32x32x32 (plane+child 0.95), 32x256x256 (plane+child+strips 0.99), 16x64x1024 (plane+child 1.19), 32x8x1000 (plane+child 1.28)
best 5: 8x128x2048 (plane+child 1.69), 32x8x1000 (plane+child 1.28), 16x64x1024 (plane+child 1.19), 32x256x256 (plane+child+strips 0.99), 32x32x32 (plane+child 0.95)
