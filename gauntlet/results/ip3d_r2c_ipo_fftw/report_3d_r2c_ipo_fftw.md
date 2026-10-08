# gauntlet report (3D)

run: `ip3d_r2c_ipo_fftw`  contract: 3D r2c interleaved, natural, out of place, K=1_r2c_ipo_fftw  cells: 11 listed, 11 benched, comparator: FFTW 3D (out of place, MEASURE)

control cell 64x64x64: 4 readings, 1.106..1.156


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  8x128x2048  2^3          child  raced        2077525    2396494    1.14     53.0  4.4e-16
     9x16x30  3^2          child  raced           6097       4295    0.70     21.4  5.3e-16
    16x16x16  2^4          child  raced           4090       3115    0.73     30.0  2.6e-16
  16x64x1024  2^4          child  raced         947475    1063519    1.11     55.3  4.6e-16
  16x256x256  2^4          payonce raced        1491925    1667688    1.11     35.1  4.2e-16
   32x8x1000  2^5          child  raced         238267     275933    1.11     48.3  4.1e-16
    32x32x32  2^5          child  raced          30162      32451    1.06     40.7  3.9e-16
  32x256x256  2^5          child  raced        3085762    3474250    1.09     35.7  4.2e-16
    64x64x64  2^6          child  raced         290533     338453    1.16     40.6  3.9e-16
  64x128x128  2^6          child  raced        1328350    1371412    1.02     39.5  4.8e-16
 128x128x128  2^7          child+strips raced        2785525    2990125    1.04     39.5  4.2e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                            9      2      2   0.70   1.09   1.16    1.00
 payonce                          1      0      0   1.11   1.11   1.11    1.11
 child+strips                     1      0      0   1.04   1.04   1.04    1.04
 ALL                             11      2      2   0.73   1.09   1.14    1.01
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      1      1   1.02   1.10   1.16    1.05
 odd column                       1      1      1   0.70   0.70   0.70    0.70
 ALL                             11      2      2   0.73   1.09   1.14    1.01
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   8      0      0   1.02   1.11   1.16    1.10
 4097..65536                      2      1      1   0.70   0.88   1.06    0.86
 1025..4096                       1      1      1   0.73   0.73   0.73    0.73
 ALL                             11      2      2   0.73   1.09   1.14    1.01
```


worst 10: 9x16x30 (child 0.70), 16x16x16 (child 0.73), 64x128x128 (child 1.02), 128x128x128 (child+strips 1.04), 32x32x32 (child 1.06), 32x256x256 (child 1.09), 32x8x1000 (child 1.11), 16x64x1024 (child 1.11), 16x256x256 (payonce 1.11), 8x128x2048 (child 1.14)
best 5: 64x64x64 (child 1.16), 8x128x2048 (child 1.14), 16x256x256 (payonce 1.11), 16x64x1024 (child 1.11), 32x8x1000 (child 1.11)
