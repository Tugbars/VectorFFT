# gauntlet report (2D)

run: `gnr_2d_ip256_20261108`  contract: 2D c2c interleaved, natural, out of place, K=1_ip  cells: 1 listed, 1 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.949..1.030


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     256x256  2^8          chain  replayed      221268     149216    0.65     23.7  3.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            1      1      1   0.65   0.65   0.65    0.65
 ALL                              1      1      1   0.65   0.65   0.65    0.65
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      1      1      1   0.65   0.65   0.65    0.65
 ALL                              1      1      1   0.65   0.65   0.65    0.65
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      1      1      1   0.65   0.65   0.65    0.65
 ALL                              1      1      1   0.65   0.65   0.65    0.65
```


worst 10: 256x256 (chain 0.65)
best 5: 256x256 (chain 0.65)
