# gauntlet report (3D)

run: `ip3d_c2r_oop_fftw`  contract: 3D c2r interleaved, natural, out of place, K=1_c2r_fftw  cells: 11 listed, 11 benched, comparator: FFTW 3D (out of place, MEASURE)

control cell 64x64x64: 4 readings, 1.269..1.280


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  8x128x2048  2^3          payonce raced        3096312    2560050    0.82     35.6  1.3e-15
     9x16x30  3^2          payonce raced           5503       4489    0.81     23.7  9.6e-16
    16x16x16  2^4          child  raced           2470       3141    1.22     49.7  6.7e-16
  16x64x1024  2^4          child  raced        1073237    1065188    0.99     48.9  1.2e-15
  16x256x256  2^4          payonce raced        1298737    1691737    1.30     40.4  1.2e-15
   32x8x1000  2^5          payonce raced         303333     259186    0.85     37.9  1.4e-15
    32x32x32  2^5          child  raced          24329      31972    1.31     50.5  8.9e-16
  32x256x256  2^5          payonce raced        3498600    3582793    1.01     31.5  1.3e-15
    64x64x64  2^6          payonce raced         270360     342977    1.14     43.6  1.0e-15
  64x128x128  2^6          band   raced        1239000    1418168    1.14     42.3  1.3e-15
 128x128x128  2^7          band   raced        3193275    3326281    1.03     34.5  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 payonce                          6      0      3   0.81   0.93   1.30    0.97
 child                            3      0      1   0.99   1.22   1.31    1.17
 band                             2      0      0   1.03   1.08   1.14    1.08
 ALL                             11      0      4   0.82   1.03   1.30    1.04
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      0      3   0.85   1.08   1.31    1.07
 odd column                       1      0      1   0.81   0.81   0.81    0.81
 ALL                             11      0      4   0.82   1.03   1.30    1.04
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   8      0      3   0.82   1.02   1.30    1.02
 4097..65536                      2      0      1   0.81   1.06   1.31    1.03
 1025..4096                       1      0      0   1.22   1.22   1.22    1.22
 ALL                             11      0      4   0.82   1.03   1.30    1.04
```


worst 10: 9x16x30 (payonce 0.81), 8x128x2048 (payonce 0.82), 32x8x1000 (payonce 0.85), 16x64x1024 (child 0.99), 32x256x256 (payonce 1.01), 128x128x128 (band 1.03), 64x128x128 (band 1.14), 64x64x64 (payonce 1.14), 16x16x16 (child 1.22), 16x256x256 (payonce 1.30)
best 5: 32x32x32 (child 1.31), 16x256x256 (payonce 1.30), 16x16x16 (child 1.22), 64x64x64 (payonce 1.14), 64x128x128 (band 1.14)
