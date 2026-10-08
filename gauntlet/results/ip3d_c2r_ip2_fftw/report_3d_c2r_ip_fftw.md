# gauntlet report (3D)

run: `ip3d_c2r_ip2_fftw`  contract: 3D c2r interleaved, natural, out of place, K=1_c2r_ip_fftw  cells: 11 listed, 11 benched, comparator: FFTW 3D (out of place, MEASURE)

control cell 64x64x64: 4 readings, 1.312..1.441


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  8x128x2048  2^3          payonce raced        2899162    2356837    0.80     38.0  1.3e-15
     9x16x30  3^2          payonce raced           5386       4225    0.78     24.2  9.8e-16
    16x16x16  2^4          payonce raced           2668       3101    1.15     46.1  6.7e-16
  16x64x1024  2^4          payonce raced        1091288    1048675    0.96     48.0  1.2e-15
  16x256x256  2^4          payonce raced        1165100    1739581    1.49     45.0  1.2e-15
   32x8x1000  2^5          payonce raced         276000     267170    0.96     41.7  1.5e-15
    32x32x32  2^5          payonce raced          24993      32046    1.21     49.2  8.9e-16
  32x256x256  2^5          payonce raced        2878225    3516931    1.20     38.3  1.4e-15
    64x64x64  2^6          payonce raced         228007     328777    1.40     51.7  1.1e-15
  64x128x128  2^6          payonce raced        1089287    1416112    1.28     48.1  1.3e-15
 128x128x128  2^7          payonce raced        2312812    3069756    1.25     47.6  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 payonce                         11      1      4   0.80   1.20   1.40    1.11
 ALL                             11      1      4   0.80   1.20   1.40    1.11
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      0      3   0.96   1.20   1.49    1.15
 odd column                       1      1      1   0.78   0.78   0.78    0.78
 ALL                             11      1      4   0.80   1.20   1.40    1.11
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   8      0      3   0.80   1.22   1.49    1.14
 4097..65536                      2      1      1   0.78   1.00   1.21    0.97
 1025..4096                       1      0      0   1.15   1.15   1.15    1.15
 ALL                             11      1      4   0.80   1.20   1.40    1.11
```


worst 10: 9x16x30 (payonce 0.78), 8x128x2048 (payonce 0.80), 16x64x1024 (payonce 0.96), 32x8x1000 (payonce 0.96), 16x16x16 (payonce 1.15), 32x256x256 (payonce 1.20), 32x32x32 (payonce 1.21), 128x128x128 (payonce 1.25), 64x128x128 (payonce 1.28), 64x64x64 (payonce 1.40)
best 5: 16x256x256 (payonce 1.49), 64x64x64 (payonce 1.40), 64x128x128 (payonce 1.28), 128x128x128 (payonce 1.25), 32x32x32 (payonce 1.21)
