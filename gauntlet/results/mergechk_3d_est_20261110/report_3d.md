# gauntlet report (3D)

run: `mergechk_3d_est_20261110`  contract: 3D c2c interleaved, natural, out of place, K=1  cells: 5 listed, 5 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.337..1.370


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  16x256x256  2^4          2p     replayed     4121500    4690250    1.13     25.4  1.1e-15
    32x32x32  2^5          2p     replayed       41741      62801    1.49     58.9  3.8e-16
    64x64x64  2^6          2p     replayed      610463     821994    1.34     38.6  4.9e-16
 128x128x128  2^7          2p     replayed     9413487   11958219    1.20     23.4  1.3e-15
   256x16x16  2^8          ztt    replayed       88787     136223    1.45     59.0  3.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 2p                               4      0      0   1.13   1.27   1.49    1.28
 ztt                              1      0      0   1.45   1.45   1.45    1.45
 ALL                              5      0      0   1.13   1.34   1.49    1.31
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0   1.13   1.34   1.49    1.31
 ALL                              5      0      0   1.13   1.34   1.49    1.31
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   3      0      0   1.13   1.20   1.34    1.22
 4097..65536                      2      0      0   1.45   1.47   1.49    1.47
 ALL                              5      0      0   1.13   1.34   1.49    1.31
```


worst 10: 16x256x256 (2p 1.13), 128x128x128 (2p 1.20), 64x64x64 (2p 1.34), 256x16x16 (ztt 1.45), 32x32x32 (2p 1.49)
best 5: 32x32x32 (2p 1.49), 256x16x16 (ztt 1.45), 64x64x64 (2p 1.34), 128x128x128 (2p 1.20), 16x256x256 (2p 1.13)
