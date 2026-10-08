# gauntlet report (2D)

run: `cr_2d_r2c_ip`  contract: 2D r2c interleaved, natural, out of place, K=1_r2c_ip_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.745..0.791


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            737        377    0.50      6.4  3.1e-16
     16x1000  2^4          chain  raced           9089       5154    0.52     61.5  4.6e-16
     16x1024  2^4          chain  raced           3726       4643    1.15    153.9  5.2e-16
       17x64  17           chain  raced           1689       1839    1.09     16.2  6.4e-16
     32x1024  2^5          chain  raced           7559      12806    1.54    162.6  5.7e-16
       64x15  2^6          chain  raced           1620       3729    2.29     14.7  4.6e-16
       64x30  2^6          chain  raced           2412       1830    0.65     21.7  3.3e-16
       64x64  2^6          chain  raced           4192       3245    0.74     29.3  3.0e-16
      64x256  2^6          chain  raced           4130       4345    1.04    138.8  3.9e-16
     128x128  2^7          chain  raced           5476       6334    1.10    104.7  3.6e-16
     128x512  2^7          chain  raced          19615      16548    0.84    133.6  4.1e-16
     256x256  2^8          chain  raced          19810      23259    1.05    132.3  4.1e-16
    256x1024  2^8          chain  raced          65480      58580    0.85    180.2  4.9e-16
     512x512  2^9          chain  raced          66553      69727    1.05    177.2  5.1e-16
   1024x1024  2^10         chain  raced         246800     264800    1.06    212.4  5.6e-16
   2048x2048  2^11         chain  raced        2898775    1045762    0.36     79.6  5.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      5      7   0.50   1.05   1.54    0.90
 ALL                             16      5      7   0.50   1.05   1.54    0.90
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      4      6   0.52   1.05   1.54    0.93
 odd column                       1      1      1   0.50   0.50   0.50    0.50
 prime column                     1      0      0   1.09   1.09   1.09    1.09
 ALL                             16      5      7   0.50   1.05   1.54    0.90
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      1      2   0.52   1.05   1.54    0.99
 > 65536 points                   4      1      2   0.36   0.95   1.06    0.76
 1025..4096                       3      2      2   0.65   0.74   1.09    0.81
 <= 256 points                    1      1      1   0.50   0.50   0.50    0.50
 257..1024                        1      0      0   2.29   2.29   2.29    2.29
 ALL                             16      5      7   0.50   1.05   1.54    0.90
```


worst 10: 2048x2048 (chain 0.36), 15x16 (chain 0.50), 16x1000 (chain 0.52), 64x30 (chain 0.65), 64x64 (chain 0.74), 128x512 (chain 0.84), 256x1024 (chain 0.85), 64x256 (chain 1.04), 512x512 (chain 1.05), 256x256 (chain 1.05)
best 5: 64x15 (chain 2.29), 32x1024 (chain 1.54), 16x1024 (chain 1.15), 128x128 (chain 1.10), 17x64 (chain 1.09)
