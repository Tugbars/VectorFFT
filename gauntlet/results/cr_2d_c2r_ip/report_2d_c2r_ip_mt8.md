# gauntlet report (2D)

run: `cr_2d_c2r_ip`  contract: 2D c2r interleaved, natural, out of place, K=1_c2r_ip_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.320..0.524


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            895        409    0.36      5.3  4.9e-16
     16x1000  2^4          chain  raced          10055       5350    0.43     55.6  9.9e-16
     16x1024  2^4          chain  raced           7451       4290    0.54     77.0  8.9e-16
       17x64  17           chain  raced           2135       1863    0.42     12.9  1.0e-15
     32x1024  2^5          chain  raced           8856       8295    0.86    138.8  8.9e-16
       64x15  2^6          chain  raced           1946       4290    2.21     12.2  6.4e-16
       64x30  2^6          chain  raced           4049       1926    0.47     12.9  7.7e-16
       64x64  2^6          chain  raced           6301       3247    0.51     19.5  7.8e-16
      64x256  2^6          chain  raced           7745       4826    0.61     74.0  7.8e-16
     128x128  2^7          chain  raced           9268       6060    0.60     61.9  9.4e-16
     128x512  2^7          chain  raced          19028      16464    0.78    137.8  1.0e-15
     256x256  2^8          chain  raced          20500      23939    1.06    127.9  1.0e-15
    256x1024  2^8          chain  raced          72393      52746    0.71    162.9  1.1e-15
     512x512  2^9          chain  raced          68273      71406    1.00    172.8  1.2e-15
   1024x1024  2^10         chain  raced         206350     255643    1.00    254.1  1.3e-15
   2048x2048  2^11         chain  raced        3462588     982288    0.28     66.6  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16     11     13   0.36   0.61   1.06    0.65
 ALL                             16     11     13   0.36   0.61   1.06    0.65
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      9     11   0.43   0.66   1.06    0.70
 odd column                       1      1      1   0.36   0.36   0.36    0.36
 prime column                     1      1      1   0.42   0.42   0.42    0.42
 ALL                             16     11     13   0.36   0.61   1.06    0.65
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      5      6   0.43   0.61   1.06    0.67
 > 65536 points                   4      2      3   0.28   0.85   1.00    0.67
 1025..4096                       3      3      3   0.42   0.47   0.51    0.47
 <= 256 points                    1      1      1   0.36   0.36   0.36    0.36
 257..1024                        1      0      0   2.21   2.21   2.21    2.21
 ALL                             16     11     13   0.36   0.61   1.06    0.65
```


worst 10: 2048x2048 (chain 0.28), 15x16 (chain 0.36), 17x64 (chain 0.42), 16x1000 (chain 0.43), 64x30 (chain 0.47), 64x64 (chain 0.51), 16x1024 (chain 0.54), 128x128 (chain 0.60), 64x256 (chain 0.61), 256x1024 (chain 0.71)
best 5: 64x15 (chain 2.21), 256x256 (chain 1.06), 1024x1024 (chain 1.00), 512x512 (chain 1.00), 32x1024 (chain 0.86)
