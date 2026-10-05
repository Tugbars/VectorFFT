# gauntlet report (2D)

run: `fftw_2d_c2r_plans_2026-10-05`  contract: 2D c2c interleaved, natural, out of place, K=1_c2r_fftw  cells: 5 listed, 4 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 1.298..1.368


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       32x63  2^5          2p     raced              -          -       -        - not benched
       32x64  2^5          chain  raced           1168       1505    1.26     48.2  5.6e-16
       64x64  2^6          chain  raced           2712       3698    1.31     45.3  8.9e-16
      64x256  2^6          chain  raced          11317      11662    1.01     50.7  8.9e-16
     128x128  2^7          chain  raced          12488      12228    0.95     45.9  8.9e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            4      0      1   0.95   1.13   1.31    1.12
 ALL                              4      0      1   0.95   1.13   1.31    1.12
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      4      0      1   0.95   1.13   1.31    1.12
 ALL                              4      0      1   0.95   1.13   1.31    1.12
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 1025..4096                       2      0      0   1.26   1.29   1.31    1.29
 4097..65536                      2      0      1   0.95   0.98   1.01    0.98
 ALL                              4      0      1   0.95   1.13   1.31    1.12
```


worst 10: 128x128 (chain 0.95), 64x256 (chain 1.01), 32x64 (chain 1.26), 64x64 (chain 1.31)
best 5: 64x64 (chain 1.31), 32x64 (chain 1.26), 64x256 (chain 1.01), 128x128 (chain 0.95)
