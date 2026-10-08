# gauntlet report (3D)

run: `mt8_3d_r2c_oop`  contract: 3D r2c interleaved, natural, out of place, K=1_r2c_mt8  cells: 11 listed, 11 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 0.696..0.822

threaded plans that ran serial (engaged = 0 at both flips): 0


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  8x128x2048  2^3          plane+child raced         332375     597644    1.46    331.3  5.1e-16
     9x16x30  3^2          plane+child+strips raced           3498       3111    0.85     37.3  5.3e-16
    16x16x16  2^4          plane+child raced           2586       2826    1.07     47.5  2.6e-16
  16x64x1024  2^4          plane+child raced         167000     221025    1.26    313.9  5.7e-16
  16x256x256  2^4          plane+payonce+strips raced         214987     231337    1.00    243.9  4.8e-16
   32x8x1000  2^5          plane+payonce raced          56180      64256    1.14    204.7  4.1e-16
    32x32x32  2^5          plane+child raced          15130       9544    0.59     81.2  3.4e-16
  32x256x256  2^5          plane+child+strips raced         493862     465043    0.86    222.9  4.6e-16
    64x64x64  2^6          plane+payonce raced          60633      50817    0.78    194.6  4.0e-16
  64x128x128  2^6          plane+payonce+strips raced         244875     206412    0.78    214.1  4.0e-16
 128x128x128  2^7          plane+child+strips raced         443800     417669    0.91    248.1  4.4e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 plane+child                      4      1      1   0.59   1.16   1.46    1.04
 plane+child+strips               3      0      3   0.85   0.86   0.91    0.87
 plane+payonce+strips             2      1      1   0.78   0.89   1.00    0.89
 plane+payonce                    2      1      1   0.78   0.96   1.14    0.94
 ALL                             11      3      6   0.78   0.91   1.26    0.95
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      3      5   0.78   0.96   1.46    0.96
 odd column                       1      0      1   0.85   0.85   0.85    0.85
 ALL                             11      3      6   0.78   0.91   1.26    0.95
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   8      2      4   0.78   0.96   1.46    1.00
 4097..65536                      2      1      2   0.59   0.72   0.85    0.71
 1025..4096                       1      0      0   1.07   1.07   1.07    1.07
 ALL                             11      3      6   0.78   0.91   1.26    0.95
```


worst 10: 32x32x32 (plane+child 0.59), 64x64x64 (plane+payonce 0.78), 64x128x128 (plane+payonce+strips 0.78), 9x16x30 (plane+child+strips 0.85), 32x256x256 (plane+child+strips 0.86), 128x128x128 (plane+child+strips 0.91), 16x256x256 (plane+payonce+strips 1.00), 16x16x16 (plane+child 1.07), 32x8x1000 (plane+payonce 1.14), 16x64x1024 (plane+child 1.26)
best 5: 8x128x2048 (plane+child 1.46), 16x64x1024 (plane+child 1.26), 32x8x1000 (plane+payonce 1.14), 16x16x16 (plane+child 1.07), 16x256x256 (plane+payonce+strips 1.00)
