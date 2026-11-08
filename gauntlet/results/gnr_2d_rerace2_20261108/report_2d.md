# gauntlet report (2D)

run: `gnr_2d_rerace2_20261108`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 2 listed, 2 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.932..1.197


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     256x256  2^8          chain  raced         188085     178454    0.58     27.9  4.0e-16
    1024x256  2^10         chain  raced        1298844    2430345    1.71     18.2  4.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            2      1      1   0.58   1.15   1.71    1.00
 ALL                              2      1      1   0.58   1.15   1.71    1.00
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      2      1      1   0.58   1.15   1.71    1.00
 ALL                              2      1      1   0.58   1.15   1.71    1.00
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      1      1      1   0.58   0.58   0.58    0.58
 > 65536 points                   1      0      0   1.71   1.71   1.71    1.71
 ALL                              2      1      1   0.58   1.15   1.71    1.00
```


worst 10: 256x256 (chain 0.58), 1024x256 (chain 1.71)
best 5: 1024x256 (chain 1.71), 256x256 (chain 0.58)
