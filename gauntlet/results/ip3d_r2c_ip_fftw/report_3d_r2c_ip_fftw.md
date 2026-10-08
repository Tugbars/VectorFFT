# gauntlet report (3D)

run: `ip3d_r2c_ip_fftw`  contract: 3D r2c interleaved, natural, out of place, K=1_r2c_ip_fftw  cells: 11 listed, 11 benched, comparator: FFTW 3D (out of place, MEASURE)

control cell 64x64x64: 4 readings, 1.141..1.155


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  8x128x2048  2^3          child  raced        2159675    2576556    1.17     51.0  4.9e-16
     9x16x30  3^2          child+strips raced           6274       4294    0.68     20.8  5.3e-16
    16x16x16  2^4          child  raced           4489       3132    0.69     27.4  2.6e-16
  16x64x1024  2^4          child  raced         956538    1086087    1.01     54.8  4.6e-16
  16x256x256  2^4          payonce raced        1606775    1683949    1.04     32.6  4.2e-16
   32x8x1000  2^5          child  raced         233340     276000    1.15     49.3  4.1e-16
    32x32x32  2^5          child  raced          30184      32560    1.07     40.7  3.9e-16
  32x256x256  2^5          child  raced        3444175    3854269    1.09     32.0  4.6e-16
    64x64x64  2^6          child+strips raced         292940     341973    1.15     40.3  3.9e-16
  64x128x128  2^6          child  raced        1417338    1378468    0.97     37.0  4.8e-16
 128x128x128  2^7          child+strips raced        2888125    3043399    1.01     38.1  4.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                            7      1      2   0.69   1.07   1.17    1.01
 child+strips                     3      1      1   0.68   1.01   1.15    0.93
 payonce                          1      0      0   1.04   1.04   1.04    1.04
 ALL                             11      2      3   0.69   1.04   1.15    0.99
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      1      2   0.97   1.06   1.17    1.03
 odd column                       1      1      1   0.68   0.68   0.68    0.68
 ALL                             11      2      3   0.69   1.04   1.15    0.99
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   8      0      1   0.97   1.06   1.17    1.07
 4097..65536                      2      1      1   0.68   0.88   1.07    0.86
 1025..4096                       1      1      1   0.69   0.69   0.69    0.69
 ALL                             11      2      3   0.69   1.04   1.15    0.99
```


worst 10: 9x16x30 (child+strips 0.68), 16x16x16 (child 0.69), 64x128x128 (child 0.97), 128x128x128 (child+strips 1.01), 16x64x1024 (child 1.01), 16x256x256 (payonce 1.04), 32x32x32 (child 1.07), 32x256x256 (child 1.09), 32x8x1000 (child 1.15), 64x64x64 (child+strips 1.15)
best 5: 8x128x2048 (child 1.17), 64x64x64 (child+strips 1.15), 32x8x1000 (child 1.15), 32x256x256 (child 1.09), 32x32x32 (child 1.07)
