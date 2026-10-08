# gauntlet report (3D)

run: `ip3d_c2r_ipo_fftw`  contract: 3D c2r interleaved, natural, out of place, K=1_c2r_ipo_fftw  cells: 11 listed, 11 benched, comparator: FFTW 3D (out of place, MEASURE)

control cell 64x64x64: 4 readings, 1.333..1.434


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  8x128x2048  2^3          payonce raced        2724000    2365912    0.83     40.4  1.3e-15
     9x16x30  3^2          payonce raced           5071       4208    0.82     25.7  9.6e-16
    16x16x16  2^4          payonce raced           2652       3122    1.17     46.3  6.7e-16
  16x64x1024  2^4          payonce raced        1132112    1052787    0.91     46.3  1.2e-15
  16x256x256  2^4          payonce raced        1176662    1725362    1.47     44.6  1.3e-15
   32x8x1000  2^5          payonce raced         278047     271733    0.97     41.4  1.5e-15
    32x32x32  2^5          payonce raced          26583      31924    1.20     46.2  8.3e-16
  32x256x256  2^5          payonce raced        2924225    3545775    1.20     37.7  1.4e-15
    64x64x64  2^6          payonce raced         229753     332193    1.43     51.3  1.1e-15
  64x128x128  2^6          payonce raced        1087562    1423344    1.29     48.2  1.3e-15
 128x128x128  2^7          payonce raced        2211475    3085850    1.39     49.8  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 payonce                         11      0      4   0.83   1.20   1.43    1.13
 ALL                             11      0      4   0.83   1.20   1.43    1.13
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      0      3   0.91   1.20   1.47    1.17
 odd column                       1      0      1   0.82   0.82   0.82    0.82
 ALL                             11      0      4   0.83   1.20   1.43    1.13
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   8      0      3   0.83   1.25   1.47    1.16
 4097..65536                      2      0      1   0.82   1.01   1.20    0.99
 1025..4096                       1      0      0   1.17   1.17   1.17    1.17
 ALL                             11      0      4   0.83   1.20   1.43    1.13
```


worst 10: 9x16x30 (payonce 0.82), 8x128x2048 (payonce 0.83), 16x64x1024 (payonce 0.91), 32x8x1000 (payonce 0.97), 16x16x16 (payonce 1.17), 32x32x32 (payonce 1.20), 32x256x256 (payonce 1.20), 64x128x128 (payonce 1.29), 128x128x128 (payonce 1.39), 64x64x64 (payonce 1.43)
best 5: 16x256x256 (payonce 1.47), 64x64x64 (payonce 1.43), 128x128x128 (payonce 1.39), 64x128x128 (payonce 1.29), 32x256x256 (payonce 1.20)
