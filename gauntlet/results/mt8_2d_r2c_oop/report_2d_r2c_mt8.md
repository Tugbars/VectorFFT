# gauntlet report (2D)

run: `mt8_2d_r2c_oop`  contract: 2D r2c interleaved, natural, out of place, K=1_r2c_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.760..0.781


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            221        377    1.71     21.5  2.7e-16
     16x1000  2^4          chain  raced           4703       6988    1.49    118.8  4.6e-16
     16x1024  2^4          chain  raced          10452       9249    0.70     54.9  3.9e-16
       17x64  17           chain  raced           1985       1844    0.91     13.8  6.4e-16
     32x1024  2^5          chain  raced          29020      13676    0.33     42.3  4.7e-16
       64x15  2^6          chain  raced           1423       3742    2.35     16.7  3.7e-16
       64x30  2^6          chain  raced           2474       1711    0.66     21.2  4.4e-16
       64x64  2^6          chain  raced           4101       3164    0.76     30.0  2.7e-16
      64x256  2^6          chain  raced           4554       5716    1.14    125.9  3.9e-16
     128x128  2^7          chain  raced           4945       7537    1.10    116.0  3.3e-16
     128x512  2^7          chain  raced          17010      16338    0.88    154.1  5.0e-16
     256x256  2^8          chain  raced          21026      22100    1.05    124.7  5.4e-16
    256x1024  2^8          chain  raced          68407      62016    0.87    172.4  6.1e-16
     512x512  2^9          chain  raced         118427      71840    0.61     99.6  5.1e-16
   1024x1024  2^10         chain  raced         204900     280118    1.23    255.9  5.6e-16
   2048x2048  2^11         chain  raced        2775625    1495693    0.53     83.1  5.6e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      6      9   0.53   0.89   1.71    0.92
 ALL                             16      6      9   0.53   0.89   1.71    0.92
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      6      8   0.53   0.87   1.49    0.88
 odd column                       1      0      0   1.71   1.71   1.71    1.71
 prime column                     1      0      1   0.91   0.91   0.91    0.91
 ALL                             16      6      9   0.53   0.89   1.71    0.92
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      2      3   0.33   1.05   1.49    0.87
 > 65536 points                   4      2      3   0.53   0.74   1.23    0.77
 1025..4096                       3      2      3   0.66   0.76   0.91    0.77
 <= 256 points                    1      0      0   1.71   1.71   1.71    1.71
 257..1024                        1      0      0   2.35   2.35   2.35    2.35
 ALL                             16      6      9   0.53   0.89   1.71    0.92
```


worst 10: 32x1024 (chain 0.33), 2048x2048 (chain 0.53), 512x512 (chain 0.61), 64x30 (chain 0.66), 16x1024 (chain 0.70), 64x64 (chain 0.76), 256x1024 (chain 0.87), 128x512 (chain 0.88), 17x64 (chain 0.91), 256x256 (chain 1.05)
best 5: 64x15 (chain 2.35), 15x16 (chain 1.71), 16x1000 (chain 1.49), 1024x1024 (chain 1.23), 64x256 (chain 1.14)
