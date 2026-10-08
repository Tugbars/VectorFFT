# gauntlet report (3D)

run: `ip3d_r2c_oop_fftw`  contract: 3D r2c interleaved, natural, out of place, K=1_r2c_fftw  cells: 11 listed, 11 benched, comparator: FFTW 3D (out of place, MEASURE)

control cell 64x64x64: 4 readings, 1.042..1.095


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  8x128x2048  2^3          child  raced        2907213    3401925    1.08     37.9  5.1e-16
     9x16x30  3^2          payonce raced           6185       4447    0.71     21.1  4.5e-16
    16x16x16  2^4          child  raced           2302       3255    1.40     53.4  2.6e-16
  16x64x1024  2^4          child  raced        1096275    1174843    1.04     47.8  4.6e-16
  16x256x256  2^4          payonce raced        1683062    1798769    1.02     31.2  4.2e-16
   32x8x1000  2^5          child  raced         272760     280817    1.02     42.2  4.1e-16
    32x32x32  2^5          child  raced          29111      32494    1.10     42.2  3.9e-16
  32x256x256  2^5          child  raced        4303588    4780699    1.09     25.6  4.2e-16
    64x64x64  2^6          child  raced         306080     343530    1.04     38.5  3.7e-16
  64x128x128  2^6          child+strips raced        1448625    1406268    0.88     36.2  5.7e-16
 128x128x128  2^7          child+strips raced        3828300    3182418    0.81     28.8  4.2e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                            7      0      0   1.02   1.08   1.40    1.10
 payonce                          2      1      1   0.71   0.87   1.02    0.85
 child+strips                     2      0      2   0.81   0.84   0.88    0.84
 ALL                             11      1      3   0.81   1.04   1.10    1.00
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      0      2   0.88   1.04   1.40    1.04
 odd column                       1      1      1   0.71   0.71   0.71    0.71
 ALL                             11      1      3   0.81   1.04   1.10    1.00
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   8      0      2   0.81   1.03   1.09    0.99
 4097..65536                      2      1      1   0.71   0.91   1.10    0.89
 1025..4096                       1      0      0   1.40   1.40   1.40    1.40
 ALL                             11      1      3   0.81   1.04   1.10    1.00
```


worst 10: 9x16x30 (payonce 0.71), 128x128x128 (child+strips 0.81), 64x128x128 (child+strips 0.88), 32x8x1000 (child 1.02), 16x256x256 (payonce 1.02), 64x64x64 (child 1.04), 16x64x1024 (child 1.04), 8x128x2048 (child 1.08), 32x256x256 (child 1.09), 32x32x32 (child 1.10)
best 5: 16x16x16 (child 1.40), 32x32x32 (child 1.10), 32x256x256 (child 1.09), 8x128x2048 (child 1.08), 16x64x1024 (child 1.04)
