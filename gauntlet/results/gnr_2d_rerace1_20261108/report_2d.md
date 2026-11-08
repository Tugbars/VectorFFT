# gauntlet report (2D)

run: `gnr_2d_rerace1_20261108`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 2 listed, 2 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.941..1.416


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     256x256  2^8          chain+rb2 raced         249953     181965    0.68     21.0  3.5e-16
    1024x256  2^10         chain+rb2 raced        1238436    1838680    0.87     19.1  4.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain+rb2                        2      1      2   0.68   0.78   0.87    0.77
 ALL                              2      1      2   0.68   0.78   0.87    0.77
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      2      1      2   0.68   0.78   0.87    0.77
 ALL                              2      1      2   0.68   0.78   0.87    0.77
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      1      1      1   0.68   0.68   0.68    0.68
 > 65536 points                   1      0      1   0.87   0.87   0.87    0.87
 ALL                              2      1      2   0.68   0.78   0.87    0.77
```


worst 10: 256x256 (chain+rb2 0.68), 1024x256 (chain+rb2 0.87)
best 5: 1024x256 (chain+rb2 0.87), 256x256 (chain+rb2 0.68)
