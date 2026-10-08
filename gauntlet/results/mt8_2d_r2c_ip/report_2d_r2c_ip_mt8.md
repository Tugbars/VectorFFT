# gauntlet report (2D)

run: `mt8_2d_r2c_ip`  contract: 2D r2c interleaved, natural, out of place, K=1_r2c_ip_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 3 readings, 0.730..0.802


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            711        379    0.53      6.7  2.7e-16
     16x1000  2^4          chain  raced          10559       5226    0.49     52.9  4.6e-16
     16x1024  2^4          chain  raced           4346       5522    1.27    131.9  5.2e-16
       17x64  17           chain  raced           1739       1873    1.04     15.8  6.4e-16
     32x1024  2^5          chain  raced          21542      13619    0.56     57.0  4.7e-16
       64x15  2^6          chain  raced           1523       3901    2.25     15.6  3.7e-16
       64x30  2^6          chain  raced           2628       1802    0.68     19.9  4.4e-16
       64x64  2^6          chain  raced           4207       3450    0.80     29.2  3.0e-16
      64x256  2^6          chain  raced          12201       4456    0.36     47.0  3.9e-16
     128x128  2^7          chain  raced           4325       6785    1.32    132.6  3.6e-16
     128x512  2^7          chain  raced          21297      18169    0.76    123.1  5.0e-16
     256x256  2^8          chain  raced          27744      22940    0.65     94.5  4.1e-16
    256x1024  2^8          chain  raced          62560      61137    0.92    188.6  4.9e-16
     512x512  2^9          chain  raced         117500      70110    0.52    100.4  5.1e-16
   1024x1024  2^10         chain  raced         535750     258194    0.44     97.9  5.1e-16
   2048x2048  2^11         chain  raced        2283962    1060356    0.40    101.0  5.6e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16     10     12   0.40   0.67   1.32    0.72
 ALL                             16     10     12   0.40   0.67   1.32    0.72
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      9     11   0.40   0.67   1.32    0.71
 odd column                       1      1      1   0.53   0.53   0.53    0.53
 prime column                     1      0      0   1.04   1.04   1.04    1.04
 ALL                             16     10     12   0.40   0.67   1.32    0.72
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      5      5   0.36   0.65   1.32    0.70
 > 65536 points                   4      3      4   0.40   0.48   0.92    0.54
 1025..4096                       3      1      2   0.68   0.80   1.04    0.83
 <= 256 points                    1      1      1   0.53   0.53   0.53    0.53
 257..1024                        1      0      0   2.25   2.25   2.25    2.25
 ALL                             16     10     12   0.40   0.67   1.32    0.72
```


worst 10: 64x256 (chain 0.36), 2048x2048 (chain 0.40), 1024x1024 (chain 0.44), 16x1000 (chain 0.49), 512x512 (chain 0.52), 15x16 (chain 0.53), 32x1024 (chain 0.56), 256x256 (chain 0.65), 64x30 (chain 0.68), 128x512 (chain 0.76)
best 5: 64x15 (chain 2.25), 128x128 (chain 1.32), 16x1024 (chain 1.27), 17x64 (chain 1.04), 256x1024 (chain 0.92)
