# gauntlet report (2D)

run: `gauntlet_2d-pow2`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 18 listed, 18 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.270..1.297


## every shape

```
       N1xN2  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         8x8  2^3          chain  raced             94         36    0.39     20.5  1.5e-16
       16x16  2^4          chain  raced            259        146    0.56     39.5  2.5e-16
     16x4096  2^4          chain  raced         130317     133365    1.01     40.2  3.1e-16
       32x32  2^5          chain  raced            917        720    0.79     55.8  2.4e-16
     32x1024  2^5          chain  raced          55948      68030    1.19     43.9  3.5e-16
       64x64  2^6          chain  raced           5548       7105    1.21     44.3  3.8e-16
      64x256  2^6          chain  raced          27561      32464    1.16     41.6  3.8e-16
     128x128  2^7          chain  raced          25930      30116    1.14     44.2  4.3e-16
      256x64  2^8          chain  raced          23036      31116    1.34     49.8  4.8e-16
     256x256  2^8          chain  raced         114863     127723    1.05     45.6  5.8e-16
     512x512  2^9          chain  raced         577337     852568    1.44     40.9  5.4e-16
     1024x32  2^10         chain  raced          74920      76397    0.86     32.8  4.7e-16
   1024x1024  2^10         chain  replayed     2896987    4986912    1.65     36.2  1.3e-15
     4096x16  2^12         chain  raced         201413     144725    0.72     26.0  4.4e-16
     4096x64  2^12         chain  raced         828362     826743    0.97     28.5  6.3e-16
     8192x64  2^13         chain  raced        1866475    1929281    0.91     26.7  1.3e-15
    16384x64  2^14         chain  raced        4203537    5010469    1.18     24.9  1.3e-15
    32768x64  2^15         chain  raced        9471212   15537718    1.56     23.2  1.6e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           18      4      7   0.56   1.09   1.56    1.00
 ALL                             18      4      7   0.56   1.09   1.56    1.00
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     18      4      7   0.56   1.09   1.56    1.00
 ALL                             18      4      7   0.56   1.09   1.56    1.00
```


## by plane size (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      8      1      2   0.72   1.09   1.34    1.04
 > 65536 points                   6      0      2   0.91   1.31   1.65    1.25
 <= 256 points                    2      2      2   0.39   0.47   0.56    0.46
 257..1024                        1      1      1   0.79   0.79   0.79    0.79
 1025..4096                       1      0      0   1.21   1.21   1.21    1.21
 ALL                             18      4      7   0.56   1.09   1.56    1.00
```


worst 10: 8x8 (chain 0.39), 16x16 (chain 0.56), 4096x16 (chain 0.72), 32x32 (chain 0.79), 1024x32 (chain 0.86), 8192x64 (chain 0.91), 4096x64 (chain 0.97), 16x4096 (chain 1.01), 256x256 (chain 1.05), 128x128 (chain 1.14)
best 5: 1024x1024 (chain 1.65), 32768x64 (chain 1.56), 512x512 (chain 1.44), 256x64 (chain 1.34), 64x64 (chain 1.21)
