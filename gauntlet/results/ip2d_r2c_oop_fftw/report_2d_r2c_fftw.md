# gauntlet report (2D)

run: `ip2d_r2c_oop_fftw`  contract: 2D r2c interleaved, natural, out of place, K=1_r2c_fftw  cells: 16 listed, 16 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 1.370..1.447


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            141        151    1.07     33.7  2.8e-16
     16x1000  2^4          chain  raced          12826      13008    1.01     43.6  4.1e-16
     16x1024  2^4          chain  raced          13308      11006    0.82     43.1  3.9e-16
       17x64  17           chain  raced            772       2006    2.56     35.5  4.8e-16
     32x1024  2^5          chain  raced          23722      22848    0.96     51.8  4.5e-16
       64x15  2^6          chain  raced           1018        646    0.62     23.3  3.8e-16
       64x30  2^6          chain  raced           2172       1745    0.80     24.1  4.4e-16
       64x64  2^6          chain  raced           2627       3591    1.35     46.8  3.0e-16
      64x256  2^6          chain  raced          11040      11545    1.01     51.9  3.9e-16
     128x128  2^7          chain  raced          11587      12158    1.04     49.5  3.6e-16
     128x512  2^7          chain  raced          51780      52650    1.01     50.6  4.7e-16
     256x256  2^8          chain  raced          61643      69414    1.12     42.5  4.1e-16
    256x1024  2^8          chain  raced         314740     330163    1.03     37.5  4.3e-16
     512x512  2^9          chain  raced         366033     352373    0.96     32.2  5.1e-16
   1024x1024  2^10         chain  raced        1445537    1667881    1.02     36.3  4.5e-16
   2048x2048  2^11         chain  raced        8009463   12220550    1.49     28.8  5.3e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      2      5   0.80   1.01   1.49    1.06
 ALL                             16      2      5   0.80   1.01   1.49    1.06
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      2      5   0.80   1.01   1.35    1.00
 odd column                       1      0      0   1.07   1.07   1.07    1.07
 prime column                     1      0      0   2.56   2.56   2.56    2.56
 ALL                             16      2      5   0.80   1.01   1.49    1.06
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      0      2   0.82   1.01   1.12    0.99
 > 65536 points                   4      0      1   0.96   1.02   1.49    1.10
 1025..4096                       3      1      1   0.80   1.35   2.56    1.40
 <= 256 points                    1      0      0   1.07   1.07   1.07    1.07
 257..1024                        1      1      1   0.62   0.62   0.62    0.62
 ALL                             16      2      5   0.80   1.01   1.49    1.06
```


worst 10: 64x15 (chain 0.62), 64x30 (chain 0.80), 16x1024 (chain 0.82), 32x1024 (chain 0.96), 512x512 (chain 0.96), 16x1000 (chain 1.01), 128x512 (chain 1.01), 64x256 (chain 1.01), 1024x1024 (chain 1.02), 256x1024 (chain 1.03)
best 5: 17x64 (chain 2.56), 2048x2048 (chain 1.49), 64x64 (chain 1.35), 256x256 (chain 1.12), 15x16 (chain 1.07)
