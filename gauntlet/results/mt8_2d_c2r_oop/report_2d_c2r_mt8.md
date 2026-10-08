# gauntlet report (2D)

run: `mt8_2d_c2r_oop`  contract: 2D c2r interleaved, natural, out of place, K=1_c2r_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.234..0.379


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            237        311    1.31     20.0  4.9e-16
     16x1000  2^4          chain  raced           5031       5614    1.08    111.0  9.9e-16
     16x1024  2^4          chain  raced           6561       5937    0.91     87.4  1.1e-15
       17x64  17           chain  raced           1640       2354    1.44     16.7  1.0e-15
     32x1024  2^5          chain  raced          12134       9306    0.69    101.3  9.2e-16
       64x15  2^6          chain  raced           1940       4077    2.10     12.3  5.9e-16
       64x30  2^6          chain  raced           2883       4286    1.36     18.2  8.1e-16
       64x64  2^6          chain  raced           5183       1997    0.39     23.7  7.8e-16
      64x256  2^6          chain  raced           6625       4756    0.72     86.6  8.9e-16
     128x128  2^7          chain  raced          20316       5797    0.17     28.2  1.0e-15
     128x512  2^7          chain  raced          14580      15832    1.05    179.8  1.0e-15
     256x256  2^8          chain  raced          19744      16990    0.84    132.8  1.0e-15
    256x1024  2^8          chain  raced          60700      59410    0.94    194.3  1.1e-15
     512x512  2^9          chain  raced         123640      68713    0.53     95.4  1.1e-15
   1024x1024  2^10         chain  raced         466438     282962    0.52    112.4  1.4e-15
   2048x2048  2^11         chain  raced        3378487    1982506    0.58     68.3  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      7     10   0.39   0.87   1.44    0.79
 ALL                             16      7     10   0.39   0.87   1.44    0.79
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      7     10   0.39   0.78   1.36    0.73
 odd column                       1      0      0   1.31   1.31   1.31    1.31
 prime column                     1      0      0   1.44   1.44   1.44    1.44
 ALL                             16      7     10   0.39   0.87   1.44    0.79
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      3      5   0.17   0.84   1.08    0.69
 > 65536 points                   4      3      4   0.52   0.55   0.94    0.62
 1025..4096                       3      1      1   0.39   1.36   1.44    0.91
 <= 256 points                    1      0      0   1.31   1.31   1.31    1.31
 257..1024                        1      0      0   2.10   2.10   2.10    2.10
 ALL                             16      7     10   0.39   0.87   1.44    0.79
```


worst 10: 128x128 (chain 0.17), 64x64 (chain 0.39), 1024x1024 (chain 0.52), 512x512 (chain 0.53), 2048x2048 (chain 0.58), 32x1024 (chain 0.69), 64x256 (chain 0.72), 256x256 (chain 0.84), 16x1024 (chain 0.91), 256x1024 (chain 0.94)
best 5: 64x15 (chain 2.10), 17x64 (chain 1.44), 64x30 (chain 1.36), 15x16 (chain 1.31), 16x1000 (chain 1.08)
