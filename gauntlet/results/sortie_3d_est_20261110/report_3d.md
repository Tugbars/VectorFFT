# gauntlet report (3D)

run: `sortie_3d_est_20261110`  contract: 3D c2c interleaved, natural, out of place, K=1  cells: 5 listed, 5 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.359..1.412


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  64x256x256  2^6          2p     replayed    14926162   20840650    1.37     30.9  2.8e-15
 128x128x128  2^7          2p     replayed     7929050   10648606    1.30     27.8  1.3e-15
 256x128x128  2^8          2p     replayed    16268075   22385444    1.35     28.4  2.8e-15
  256x256x64  2^8          2p     replayed    14976250   21328744    1.35     30.8  2.6e-15
 256x256x256  2^8          2p     replayed    66175625  101080712    1.50     30.4  1.5e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 2p                               5      0      0   1.30   1.35   1.50    1.37
 ALL                              5      0      0   1.30   1.35   1.50    1.37
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0   1.30   1.35   1.50    1.37
 ALL                              5      0      0   1.30   1.35   1.50    1.37
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   5      0      0   1.30   1.35   1.50    1.37
 ALL                              5      0      0   1.30   1.35   1.50    1.37
```


worst 10: 128x128x128 (2p 1.30), 256x128x128 (2p 1.35), 256x256x64 (2p 1.35), 64x256x256 (2p 1.37), 256x256x256 (2p 1.50)
best 5: 256x256x256 (2p 1.50), 64x256x256 (2p 1.37), 256x256x64 (2p 1.35), 256x128x128 (2p 1.35), 128x128x128 (2p 1.30)
