# gauntlet report (3D)

run: `3d_smoke_2026-09-24`  contract: 3D c2c interleaved, natural, out of place, K=1  cells: 5 listed, 5 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.355..1.392


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x2x4096  2            chain  raced          24279      84878    3.48     47.2  3.6e-16
       8x8x8  2^3          chain  raced            390        366    0.93     59.0  2.4e-16
    16x16x16  2^4          chain  raced           4960       6975    1.40     49.5  3.5e-16
    64x64x64  2^6          chain  raced         583650     797987    1.36     40.4  4.8e-16
  256x256x16  2^8          chain  raced        3084012    4124431    1.30     34.0  9.9e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                                5      0      1   0.93   1.36   3.48    1.52
 ALL                              5      0      1   0.93   1.36   3.48    1.52
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      1   0.93   1.36   3.48    1.52
 ALL                              5      0      1   0.93   1.36   3.48    1.52
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   2      0      0   1.30   1.33   1.36    1.33
 4097..65536                      1      0      0   3.48   3.48   3.48    3.48
 257..1024                        1      0      1   0.93   0.93   0.93    0.93
 1025..4096                       1      0      0   1.40   1.40   1.40    1.40
 ALL                              5      0      1   0.93   1.36   3.48    1.52
```


worst 10: 8x8x8 (- 0.93), 256x256x16 (- 1.30), 64x64x64 (- 1.36), 16x16x16 (- 1.40), 2x2x4096 (- 3.48)
best 5: 2x2x4096 (- 3.48), 16x16x16 (- 1.40), 64x64x64 (- 1.36), 256x256x16 (- 1.30), 8x8x8 (- 0.93)
