# gauntlet report (2D)

run: `gnr_2d_1mib_20261108`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 2 listed, 2 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.919..1.141


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     128x256  2^7          chain+rb2 raced          66256      59556    0.84     37.1  3.4e-16
     256x128  2^8          chain+rb2 raced          71259      56795    0.78     34.5  3.4e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain+rb2                        2      1      2   0.78   0.81   0.84    0.81
 ALL                              2      1      2   0.78   0.81   0.84    0.81
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      2      1      2   0.78   0.81   0.84    0.81
 ALL                              2      1      2   0.78   0.81   0.84    0.81
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      2      1      2   0.78   0.81   0.84    0.81
 ALL                              2      1      2   0.78   0.81   0.84    0.81
```


worst 10: 256x128 (chain+rb2 0.78), 128x256 (chain+rb2 0.84)
best 5: 128x256 (chain+rb2 0.84), 256x128 (chain+rb2 0.78)
