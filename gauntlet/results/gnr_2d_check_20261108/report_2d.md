# gauntlet report (2D)

run: `gnr_2d_check_20261108`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 4 listed, 4 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.934..1.323


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     256x256  2^8          chain+rb2 replayed      224917     223969    0.95     23.3  4.4e-16
     512x512  2^9          chain  replayed     1274797    2113506    1.32     18.5  4.1e-16
    1024x256  2^10         chain  replayed     1953456    2240159    1.06     12.1  4.3e-16
   1024x1024  2^10         chain  replayed     5438290    9013786    0.96     19.3  4.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            3      0      1   0.96   1.06   1.32    1.10
 chain+rb2                        1      0      1   0.95   0.95   0.95    0.95
 ALL                              4      0      2   0.95   1.01   1.32    1.06
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      4      0      2   0.95   1.01   1.32    1.06
 ALL                              4      0      2   0.95   1.01   1.32    1.06
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   3      0      1   0.96   1.06   1.32    1.10
 4097..65536                      1      0      1   0.95   0.95   0.95    0.95
 ALL                              4      0      2   0.95   1.01   1.32    1.06
```


worst 10: 256x256 (chain+rb2 0.95), 1024x1024 (chain 0.96), 1024x256 (chain 1.06), 512x512 (chain 1.32)
best 5: 512x512 (chain 1.32), 1024x256 (chain 1.06), 1024x1024 (chain 0.96), 256x256 (chain+rb2 0.95)
