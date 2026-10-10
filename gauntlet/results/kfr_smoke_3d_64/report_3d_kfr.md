# gauntlet report (3D)

run: `kfr_smoke_3d_64`  contract: 3D c2c interleaved, natural, out of place, K=1_kfr  cells: 1 listed, 1 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.656..1.698


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    64x64x64  2^6          2p     raced         577375     969494    1.67     40.9  4.9e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 2p                               1      0      0   1.67   1.67   1.67    1.67
 ALL                              1      0      0   1.67   1.67   1.67    1.67
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      1      0      0   1.67   1.67   1.67    1.67
 ALL                              1      0      0   1.67   1.67   1.67    1.67
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   1      0      0   1.67   1.67   1.67    1.67
 ALL                              1      0      0   1.67   1.67   1.67    1.67
```


worst 10: 64x64x64 (2p 1.67)
best 5: 64x64x64 (2p 1.67)
