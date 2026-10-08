# gauntlet report (2D)

run: `st2_2d_r2c_oop`  contract: 2D r2c interleaved, natural, out of place, K=1_r2c_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.067..1.289


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            259        411    1.59     18.3  2.7e-16
     16x1000  2^4          chain  raced           5035       6582    1.30    111.0  4.6e-16
     16x1024  2^4          chain  raced           4577       6199    1.30    125.3  5.2e-16
       17x64  17           chain  raced           1815       2071    1.14     15.1  6.4e-16
     32x1024  2^5          chain  raced           8681      12202    1.41    141.5  4.7e-16
       64x15  2^6          chain  raced           1809       5558    3.07     13.1  3.7e-16
       64x30  2^6          chain  raced           2775       2342    0.70     18.9  3.3e-16
       64x64  2^6          chain  raced           2984       3695    1.19     41.2  2.7e-16
      64x256  2^6          chain  raced           5584       6543    1.17    102.7  3.9e-16
     128x128  2^7          chain  raced           6261       8475    1.35     91.6  3.6e-16
     128x512  2^7          chain  raced          16656      16840    0.90    157.4  3.8e-16
     256x256  2^8          chain  raced          19490      23065    1.05    134.5  4.1e-16
    256x1024  2^8          chain  raced          58780      61013    0.98    200.7  6.1e-16
     512x512  2^9          chain  raced          68933      75896    1.09    171.1  5.1e-16
   1024x1024  2^10         chain  raced         261825     291881    1.01    200.2  5.6e-16
   2048x2048  2^11         chain  raced        2150625    2596750    1.21    107.3  5.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      1      3   0.90   1.18   1.59    1.21
 ALL                             16      1      3   0.90   1.18   1.59    1.21
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      1      3   0.90   1.18   1.41    1.19
 odd column                       1      0      0   1.59   1.59   1.59    1.59
 prime column                     1      0      0   1.14   1.14   1.14    1.14
 ALL                             16      1      3   0.90   1.18   1.59    1.21
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      0      1   0.90   1.30   1.41    1.20
 > 65536 points                   4      0      1   0.98   1.05   1.21    1.07
 1025..4096                       3      1      1   0.70   1.14   1.19    0.98
 <= 256 points                    1      0      0   1.59   1.59   1.59    1.59
 257..1024                        1      0      0   3.07   3.07   3.07    3.07
 ALL                             16      1      3   0.90   1.18   1.59    1.21
```


worst 10: 64x30 (chain 0.70), 128x512 (chain 0.90), 256x1024 (chain 0.98), 1024x1024 (chain 1.01), 256x256 (chain 1.05), 512x512 (chain 1.09), 17x64 (chain 1.14), 64x256 (chain 1.17), 64x64 (chain 1.19), 2048x2048 (chain 1.21)
best 5: 64x15 (chain 3.07), 15x16 (chain 1.59), 32x1024 (chain 1.41), 128x128 (chain 1.35), 16x1000 (chain 1.30)
