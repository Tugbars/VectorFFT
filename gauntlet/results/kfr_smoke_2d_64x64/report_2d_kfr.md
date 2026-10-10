# gauntlet report (2D)

run: `kfr_smoke_2d_64x64`  contract: 2D c2c interleaved, natural, out of place, K=1_kfr  cells: 1 listed, 1 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.177..2.124


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       64x64  2^6          csk    replayed        5756       6334    0.93     42.7  3.8e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 csk                              1      0      1   0.93   0.93   0.93    0.93
 ALL                              1      0      1   0.93   0.93   0.93    0.93
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      1      0      1   0.93   0.93   0.93    0.93
 ALL                              1      0      1   0.93   0.93   0.93    0.93
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 1025..4096                       1      0      1   0.93   0.93   0.93    0.93
 ALL                              1      0      1   0.93   0.93   0.93    0.93
```


worst 10: 64x64 (csk 0.93)
best 5: 64x64 (csk 0.93)
