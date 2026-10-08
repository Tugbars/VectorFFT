# gauntlet report (2D)

run: `cr_2d_c2r_oop`  contract: 2D c2r interleaved, natural, out of place, K=1_c2r_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.217..0.304


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            244        311    1.28     19.5  4.9e-16
     16x1000  2^4          chain  raced           7067       5521    0.74     79.0  1.0e-15
     16x1024  2^4          chain  raced           6786       5396    0.80     84.5  1.1e-15
       17x64  17           chain  raced           1595       2357    0.60     17.2  1.0e-15
     32x1024  2^5          chain  raced           7659      10004    0.85    160.4  8.9e-16
       64x15  2^6          chain  raced           1955       4093    2.09     12.2  6.4e-16
       64x30  2^6          chain  raced           2936       3529    1.20     17.8  8.1e-16
       64x64  2^6          chain  raced           9000       2005    0.22     13.7  7.8e-16
      64x256  2^6          chain  raced           6732       5115    0.76     85.2  8.9e-16
     128x128  2^7          chain  raced          25467       5487    0.15     22.5  9.4e-16
     128x512  2^7          chain  raced          13164      15579    1.18    199.1  1.0e-15
     256x256  2^8          chain  raced          16666      16643    0.88    157.3  1.0e-15
    256x1024  2^8          chain  raced          60240      59166    0.96    195.8  1.1e-15
     512x512  2^9          chain  raced          60520      69313    1.13    194.9  1.2e-15
   1024x1024  2^10         chain  raced         222687     284887    1.23    235.4  1.2e-15
   2048x2048  2^11         chain  raced        3291750    2016268    0.61     70.1  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      7     10   0.22   0.87   1.28    0.78
 ALL                             16      7     10   0.22   0.87   1.28    0.78
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      6      9   0.22   0.87   1.23    0.77
 odd column                       1      0      0   1.28   1.28   1.28    1.28
 prime column                     1      1      1   0.60   0.60   0.60    0.60
 ALL                             16      7     10   0.22   0.87   1.28    0.78
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      4      6   0.15   0.80   1.18    0.67
 > 65536 points                   4      1      2   0.61   1.04   1.23    0.95
 1025..4096                       3      2      2   0.22   0.60   1.20    0.54
 <= 256 points                    1      0      0   1.28   1.28   1.28    1.28
 257..1024                        1      0      0   2.09   2.09   2.09    2.09
 ALL                             16      7     10   0.22   0.87   1.28    0.78
```


worst 10: 128x128 (chain 0.15), 64x64 (chain 0.22), 17x64 (chain 0.60), 2048x2048 (chain 0.61), 16x1000 (chain 0.74), 64x256 (chain 0.76), 16x1024 (chain 0.80), 32x1024 (chain 0.85), 256x256 (chain 0.88), 256x1024 (chain 0.96)
best 5: 64x15 (chain 2.09), 15x16 (chain 1.28), 1024x1024 (chain 1.23), 64x30 (chain 1.20), 128x512 (chain 1.18)
