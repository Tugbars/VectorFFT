# gauntlet report (3D)

run: `avx2_3d_scratch`  contract: 3D c2c interleaved, natural, out of place, K=1  cells: 7 listed, 7 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 0.691..0.891


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    64x64x64  2^6          chain  raced        1275308    1076478    0.77     18.5  5.1e-16
     256x8x2  2^8          chain  raced          12584      48918    3.01     19.5  3.0e-16
     512x4x2  2^9          chain  raced          17932      76950    3.81     13.7  3.0e-16
   512x16x16  2^9          chain  raced         508375     637892    0.93     21.9  3.4e-16
  512x32x128  2^9          chain  raced       11334986   11860678    0.99     19.4  4.2e-16
    1024x2x2  2^10         chain  raced          23795     130944    4.18     10.3  3.5e-16
  1024x16x16  2^10         chain  raced        1463001    1800448    1.17     16.1  4.0e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            7      1      3   0.77   1.17   4.18    1.69
 ALL                              7      1      3   0.77   1.17   4.18    1.69
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      7      1      3   0.77   1.17   4.18    1.69
 ALL                              7      1      3   0.77   1.17   4.18    1.69
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   4      1      3   0.77   0.96   1.17    0.95
 1025..4096                       3      0      0   3.01   3.81   4.18    3.63
 ALL                              7      1      3   0.77   1.17   4.18    1.69
```


worst 10: 64x64x64 (chain 0.77), 512x16x16 (chain 0.93), 512x32x128 (chain 0.99), 1024x16x16 (chain 1.17), 256x8x2 (chain 3.01), 512x4x2 (chain 3.81), 1024x2x2 (chain 4.18)
best 5: 1024x2x2 (chain 4.18), 512x4x2 (chain 3.81), 256x8x2 (chain 3.01), 1024x16x16 (chain 1.17), 512x32x128 (chain 0.99)
