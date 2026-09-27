# gauntlet report (3D)

run: `avx512_3d_scratch`  contract: 3D c2c interleaved, natural, out of place, K=1  cells: 7 listed, 7 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 0.961..1.219


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    64x64x64  2^6          chain  raced         937663    1074819    0.89     25.2  5.1e-16
     256x8x2  2^8          chain  raced          13356      45415    2.99     18.4  3.0e-16
     512x4x2  2^9          chain  raced          14671      75633    4.37     16.8  4.0e-16
   512x16x16  2^9          chain  raced         501838     638507    1.16     22.2  3.4e-16
  512x32x128  2^9          chain  raced        7774641   13634192    1.28     28.3  4.4e-16
    1024x2x2  2^10         chain  raced          27568     106391    3.56      8.9  3.5e-16
  1024x16x16  2^10         chain  raced        1228858    1523374    1.11     19.2  4.8e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            7      0      1   0.89   1.28   4.37    1.83
 ALL                              7      0      1   0.89   1.28   4.37    1.83
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      7      0      1   0.89   1.28   4.37    1.83
 ALL                              7      0      1   0.89   1.28   4.37    1.83
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   4      0      1   0.89   1.14   1.28    1.10
 1025..4096                       3      0      0   2.99   3.56   4.37    3.59
 ALL                              7      0      1   0.89   1.28   4.37    1.83
```


worst 10: 64x64x64 (chain 0.89), 1024x16x16 (chain 1.11), 512x16x16 (chain 1.16), 512x32x128 (chain 1.28), 256x8x2 (chain 2.99), 1024x2x2 (chain 3.56), 512x4x2 (chain 4.37)
best 5: 512x4x2 (chain 4.37), 1024x2x2 (chain 3.56), 256x8x2 (chain 2.99), 512x32x128 (chain 1.28), 512x16x16 (chain 1.16)
