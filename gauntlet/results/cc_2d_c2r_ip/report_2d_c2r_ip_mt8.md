# gauntlet report (2D)

run: `cc_2d_c2r_ip`  contract: 2D c2r interleaved, natural, out of place, K=1_c2r_ip_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 2 readings, 0.479..0.560


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced           1018        417    0.40      4.7  5.4e-16
     16x1000  2^4          chain  raced          19851       5186    0.24     28.1  9.9e-16
     16x1024  2^4          chain  raced           7256       4334    0.56     79.0  8.9e-16
       17x64  17           chain  raced           2416       1903    0.75     11.4  1.0e-15
     32x1024  2^5          chain  raced           9158       9755    1.03    134.2  8.9e-16
       64x15  2^6          chain  raced           1987       4662    2.35     12.0  5.9e-16
       64x30  2^6          chain  raced           4313       1928    0.45     12.1  7.7e-16
       64x64  2^6          chain  raced           7212       3520    0.49     17.0  7.8e-16
      64x256  2^6          chain  raced           8105       4833    0.51     70.8  8.9e-16
     128x128  2^7          chain  raced          25910       6036    0.22     22.1  1.0e-15
     128x512  2^7          chain  raced          23169      17274    0.71    113.1  1.0e-15
     256x256  2^8          chain  raced          21143      23370    0.95    124.0  9.4e-16
    256x1024  2^8          chain  raced          75667      54043    0.70    155.9  1.1e-15
     512x512  2^9          chain  raced          68453      69946    0.95    172.3  1.2e-15
   1024x1024  2^10         chain  raced         228788     255693    0.99    229.2  1.3e-15
   2048x2048  2^11         chain  raced        3761612    1032512    0.27     61.3  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16     11     14   0.24   0.63   1.03    0.60
 ALL                             16     11     14   0.24   0.63   1.03    0.60
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      9     12   0.24   0.63   1.03    0.61
 odd column                       1      1      1   0.40   0.40   0.40    0.40
 prime column                     1      1      1   0.75   0.75   0.75    0.75
 ALL                             16     11     14   0.24   0.63   1.03    0.60
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      5      6   0.22   0.56   1.03    0.52
 > 65536 points                   4      2      4   0.27   0.83   0.99    0.65
 1025..4096                       3      3      3   0.45   0.49   0.75    0.55
 <= 256 points                    1      1      1   0.40   0.40   0.40    0.40
 257..1024                        1      0      0   2.35   2.35   2.35    2.35
 ALL                             16     11     14   0.24   0.63   1.03    0.60
```


worst 10: 128x128 (chain 0.22), 16x1000 (chain 0.24), 2048x2048 (chain 0.27), 15x16 (chain 0.40), 64x30 (chain 0.45), 64x64 (chain 0.49), 64x256 (chain 0.51), 16x1024 (chain 0.56), 256x1024 (chain 0.70), 128x512 (chain 0.71)
best 5: 64x15 (chain 2.35), 32x1024 (chain 1.03), 1024x1024 (chain 0.99), 512x512 (chain 0.95), 256x256 (chain 0.95)
