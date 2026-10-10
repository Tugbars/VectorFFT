# gauntlet report (2D)

run: `st2_2d_c2r_oop`  contract: 2D c2r interleaved, natural, out of place, K=1_c2r_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.815..0.876


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            243        313    1.29     19.6  8.5e-16
     16x1000  2^4          chain  raced           7776       5587    0.69     71.8  9.9e-16
     16x1024  2^4          chain  raced           7042       5866    0.83     81.4  1.1e-15
       17x64  17           chain  raced           1569       2379    1.43     17.5  1.0e-15
     32x1024  2^5          chain  raced          11725       9774    0.80    104.8  8.9e-16
       64x15  2^6          chain  raced           1924       4238    2.20     12.4  5.9e-16
       64x30  2^6          chain  raced           2008       3956    1.76     26.1  8.0e-16
       64x64  2^6          chain  raced           2500       2069    0.80     49.1  7.8e-16
      64x256  2^6          chain  raced           7046       5441    0.76     81.4  8.9e-16
     128x128  2^7          chain  raced           5508       5809    0.77    104.1  1.0e-15
     128x512  2^7          chain  raced          13484      16119    1.05    194.4  1.0e-15
     256x256  2^8          chain  raced          16721      17091    0.96    156.8  8.9e-16
    256x1024  2^8          chain  raced          42693      60843    1.27    276.3  1.1e-15
     512x512  2^9          chain  raced          54307      67313    1.24    217.2  1.2e-15
   1024x1024  2^10         chain  raced         217850     288688    1.32    240.7  1.3e-15
   2048x2048  2^11         chain  raced        2165112    2049750    0.95    106.5  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      4      8   0.76   1.01   1.76    1.07
 ALL                             16      4      8   0.76   1.01   1.76    1.07
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      4      8   0.76   0.96   1.76    1.04
 odd column                       1      0      0   1.29   1.29   1.29    1.29
 prime column                     1      0      0   1.43   1.43   1.43    1.43
 ALL                             16      4      8   0.76   1.01   1.76    1.07
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      4      6   0.69   0.80   1.05    0.83
 > 65536 points                   4      0      1   0.95   1.26   1.32    1.19
 1025..4096                       3      0      1   0.80   1.43   1.76    1.27
 <= 256 points                    1      0      0   1.29   1.29   1.29    1.29
 257..1024                        1      0      0   2.20   2.20   2.20    2.20
 ALL                             16      4      8   0.76   1.01   1.76    1.07
```


worst 10: 16x1000 (chain 0.69), 64x256 (chain 0.76), 128x128 (chain 0.77), 32x1024 (chain 0.80), 64x64 (chain 0.80), 16x1024 (chain 0.83), 2048x2048 (chain 0.95), 256x256 (chain 0.96), 128x512 (chain 1.05), 512x512 (chain 1.24)
best 5: 64x15 (chain 2.20), 64x30 (chain 1.76), 17x64 (chain 1.43), 1024x1024 (chain 1.32), 15x16 (chain 1.29)
