# gauntlet report (2D)

run: `cc_2d_r2c_oop`  contract: 2D r2c interleaved, natural, out of place, K=1_r2c_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.749..0.860


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            236        379    1.60     20.1  2.8e-16
     16x1000  2^4          chain  raced          11820       6437    0.49     47.3  4.6e-16
     16x1024  2^4          chain  raced           3883       5450    1.33    147.7  5.2e-16
       17x64  17           chain  raced           1787       1842    1.03     15.4  6.4e-16
     32x1024  2^5          chain  raced           8104      16398    1.47    151.6  4.7e-16
       64x15  2^6          chain  raced           1397       4031    2.68     17.0  4.2e-16
       64x30  2^6          chain  raced           2860       1701    0.59     18.3  3.3e-16
       64x64  2^6          chain  raced           4268       3380    0.75     28.8  2.7e-16
      64x256  2^6          chain  raced          12690       5776    0.41     45.2  3.9e-16
     128x128  2^7          chain  raced          11603       7381    0.52     49.4  3.3e-16
     128x512  2^7          chain  raced          19156      17823    0.85    136.8  5.0e-16
     256x256  2^8          chain  raced          20241      20323    0.91    129.5  5.4e-16
    256x1024  2^8          chain  raced          70040      64470    0.88    168.4  4.9e-16
     512x512  2^9          chain  raced          60960      70733    1.07    193.5  5.1e-16
   1024x1024  2^10         chain  raced         222512     271125    1.13    235.6  5.6e-16
   2048x2048  2^11         chain  raced        2910500    1590687    0.46     79.3  4.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      6      9   0.46   0.89   1.60    0.89
 ALL                             16      6      9   0.46   0.89   1.60    0.89
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      6      9   0.46   0.86   1.47    0.84
 odd column                       1      0      0   1.60   1.60   1.60    1.60
 prime column                     1      0      0   1.03   1.03   1.03    1.03
 ALL                             16      6      9   0.46   0.89   1.60    0.89
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      3      5   0.41   0.85   1.47    0.77
 > 65536 points                   4      1      2   0.46   0.97   1.13    0.84
 1025..4096                       3      2      2   0.59   0.75   1.03    0.77
 <= 256 points                    1      0      0   1.60   1.60   1.60    1.60
 257..1024                        1      0      0   2.68   2.68   2.68    2.68
 ALL                             16      6      9   0.46   0.89   1.60    0.89
```


worst 10: 64x256 (chain 0.41), 2048x2048 (chain 0.46), 16x1000 (chain 0.49), 128x128 (chain 0.52), 64x30 (chain 0.59), 64x64 (chain 0.75), 128x512 (chain 0.85), 256x1024 (chain 0.88), 256x256 (chain 0.91), 17x64 (chain 1.03)
best 5: 64x15 (chain 2.68), 15x16 (chain 1.60), 32x1024 (chain 1.47), 16x1024 (chain 1.33), 1024x1024 (chain 1.13)
