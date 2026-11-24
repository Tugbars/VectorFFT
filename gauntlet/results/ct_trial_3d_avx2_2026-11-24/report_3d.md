# gauntlet report (3D)

run: `ct_trial_3d_avx2_2026-11-24`  contract: 3D c2c interleaved, natural, out of place, K=1  cells: 1 listed, 1 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.393..1.421


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   75x24x315  3.5^2        2p     raced        1384625    1897493    1.36     39.1  7.6e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 2p                               1      0      0   1.36   1.36   1.36    1.36
 ALL                              1      0      0   1.36   1.36   1.36    1.36
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 odd column                       1      0      0   1.36   1.36   1.36    1.36
 ALL                              1      0      0   1.36   1.36   1.36    1.36
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   1      0      0   1.36   1.36   1.36    1.36
 ALL                              1      0      0   1.36   1.36   1.36    1.36
```


worst 10: 75x24x315 (2p 1.36)
best 5: 75x24x315 (2p 1.36)
