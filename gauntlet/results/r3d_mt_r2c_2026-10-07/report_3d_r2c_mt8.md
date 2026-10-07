# gauntlet report (3D)

run: `r3d_mt_r2c_2026-10-07`  contract: 3D c2c interleaved, natural, out of place, K=1_r2c_mt8  cells: 4 listed, 4 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 0.736..0.790

threaded plans that ran serial (engaged = 0 at both flips): 0


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  8x128x2048  2^3          child  raced         375800     637987    1.59    293.0  5.1e-16
  16x256x256  2^4          payonce raced         227450     234200    1.02    230.5  5.3e-16
    64x64x64  2^6          child  raced          65433      50223    0.74    180.3  4.0e-16
 128x128x128  2^7          child+strips raced         436875     434819    0.97    252.0  4.2e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                            2      1      1   0.74   1.17   1.59    1.08
 payonce                          1      0      0   1.02   1.02   1.02    1.02
 child+strips                     1      0      1   0.97   0.97   0.97    0.97
 ALL                              4      1      2   0.74   0.99   1.59    1.04
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      4      1      2   0.74   0.99   1.59    1.04
 ALL                              4      1      2   0.74   0.99   1.59    1.04
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   4      1      2   0.74   0.99   1.59    1.04
 ALL                              4      1      2   0.74   0.99   1.59    1.04
```


worst 10: 64x64x64 (child 0.74), 128x128x128 (child+strips 0.97), 16x256x256 (payonce 1.02), 8x128x2048 (child 1.59)
best 5: 8x128x2048 (child 1.59), 16x256x256 (payonce 1.02), 128x128x128 (child+strips 0.97), 64x64x64 (child 0.74)
