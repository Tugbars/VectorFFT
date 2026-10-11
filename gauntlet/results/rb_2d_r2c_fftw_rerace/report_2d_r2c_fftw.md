# gauntlet report (2D)

run: `rb_2d_r2c_fftw_rerace`  contract: 2D r2c interleaved, natural, out of place, K=1_r2c_fftw  cells: 36 listed, 36 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 1.363..1.419


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
      2x1024  2            chain  raced           1093       1097    0.96     51.6  3.7e-16
      2x2048  2            chain  raced           2588       2665    0.98     47.5  2.8e-16
      2x4096  2            chain  raced           5739       5796    1.01     46.4  3.0e-16
         4x4  2^2          chain  raced             10         13    1.23     15.7  0.0e+00
         4x8  2^2          chain  raced             13         19    1.49     31.2  2.5e-16
      4x1024  2^2          chain  raced           2555       2634    0.78     48.1  3.1e-16
      4x2048  2^2          chain  raced           5482       5338    0.97     48.6  3.9e-16
      4x4096  2^2          chain  raced          11911      11744    0.98     48.1  3.3e-16
         8x4  2^3          chain  raced             17         21    1.22     23.8  7.2e-17
       8x512  2^3          chain  raced           2403       2368    0.97     51.1  2.6e-16
      8x1024  2^3          chain  raced           5167       5115    0.98     51.5  3.3e-16
      8x4096  2^3          chain  raced          24134      24110    0.99     50.9  3.5e-16
      8x8192  2^3          chain  raced          51351      54283    1.05     51.0  4.4e-16
      16x256  2^4          chain  raced           2229       2385    1.06     55.1  2.6e-16
      16x512  2^4          chain  raced           6116       5363    0.87     43.5  3.2e-16
     16x1024  2^4          chain  raced          12861      11412    0.85     44.6  2.8e-16
     16x2048  2^4          chain  raced          27286      23714    0.86     45.0  3.7e-16
     16x4096  2^4          chain  raced          57770      52742    0.89     45.4  3.8e-16
     16x8192  2^4          chain  raced         124130     119603    0.94     44.9  3.5e-16
      32x512  2^5          chain  raced          11453      11118    0.93     50.1  3.4e-16
     32x1024  2^5          chain  raced          23739      23304    0.96     51.8  3.8e-16
     32x2048  2^5          chain  raced          51480      50251    0.96     50.9  5.1e-16
       64x16  2^6          chain  raced            522        695    1.31     49.0  3.6e-16
      64x256  2^6          chain  raced          11291      11683    1.01     50.8  3.9e-16
     64x1024  2^6          chain  raced          49154      49030    0.95     53.3  4.1e-16
     64x2048  2^6          chain  raced         109940     110420    0.99     50.7  4.6e-16
     64x4096  2^6          chain  raced         260080     265997    1.01     45.4  4.0e-16
       128x4  2^7          chain  raced            391        399    1.00     29.5  2.5e-16
    128x1024  2^7          chain  raced         108273     108400    0.99     51.4  5.0e-16
       256x4  2^8          chain  raced            919        864    0.93     27.9  3.3e-16
       256x8  2^8          chain  raced           1729       1633    0.94     32.6  2.6e-16
       512x4  2^9          chain  raced           2419       1892    0.77     23.3  3.2e-16
      1024x4  2^10         chain  raced           5717       5456    0.92     21.5  4.0e-16
      4096x4  2^12         chain  raced          33129      31273    0.88     17.3  3.8e-16
      8192x4  2^13         chain  raced          78531      65443    0.81     15.6  3.5e-16
      8192x8  2^13         chain  raced         103959     121134    1.07     25.2  3.8e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           36      2     25   0.85   0.97   1.22    0.98
 ALL                             36      2     25   0.85   0.97   1.22    0.98
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     36      2     25   0.85   0.97   1.22    0.98
 ALL                             36      2     25   0.85   0.97   1.22    0.98
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     18      0     14   0.85   0.96   1.05    0.94
 1025..4096                       8      2      7   0.77   0.95   1.06    0.92
 > 65536 points                   4      0      3   0.94   0.99   1.01    0.98
 <= 256 points                    3      0      0   1.22   1.23   1.49    1.31
 257..1024                        3      0      1   0.93   1.00   1.31    1.07
 ALL                             36      2     25   0.85   0.97   1.22    0.98
```


worst 10: 512x4 (chain 0.77), 4x1024 (chain 0.78), 8192x4 (chain 0.81), 16x1024 (chain 0.85), 16x2048 (chain 0.86), 16x512 (chain 0.87), 4096x4 (chain 0.88), 16x4096 (chain 0.89), 1024x4 (chain 0.92), 32x512 (chain 0.93)
best 5: 4x8 (chain 1.49), 64x16 (chain 1.31), 4x4 (chain 1.23), 8x4 (chain 1.22), 8192x8 (chain 1.07)
