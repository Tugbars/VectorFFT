# gauntlet report (3D)

run: `ct_default_3d_avx2_2026-11-24`  contract: 3D c2c interleaved, natural, out of place, K=1  cells: 1 listed, 1 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.370..1.413


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   75x24x315  3.5^2        2p     raced        1515375    2210293    1.44     35.8  7.6e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 2p                               1      0      0   1.44   1.44   1.44    1.44
 ALL                              1      0      0   1.44   1.44   1.44    1.44
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 odd column                       1      0      0   1.44   1.44   1.44    1.44
 ALL                              1      0      0   1.44   1.44   1.44    1.44
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   1      0      0   1.44   1.44   1.44    1.44
 ALL                              1      0      0   1.44   1.44   1.44    1.44
```


worst 10: 75x24x315 (2p 1.44)
best 5: 75x24x315 (2p 1.44)
