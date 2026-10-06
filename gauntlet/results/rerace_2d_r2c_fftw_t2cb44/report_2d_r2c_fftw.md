# gauntlet report (2D)

run: `rerace_2d_r2c_fftw_t2cb44`  contract: 2D c2c interleaved, natural, out of place, K=1_r2c_fftw  cells: 5 listed, 5 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 1.355..1.474


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     512x256  2^9          chain  raced         138387     146853    1.04     40.3  3.7e-16
    1024x512  2^10         chain  raced         692575     782131    1.13     36.0  4.1e-16
   1024x1024  2^10         chain  raced        1418038    1549387    1.08     37.0  4.5e-16
   2048x1024  2^11         chain  raced        4089588    4640337    1.12     26.9  4.3e-16
   2048x2048  2^11         chain  raced        9026800   11932062    1.31     25.6  5.3e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            5      0      0   1.04   1.12   1.31    1.13
 ALL                              5      0      0   1.04   1.12   1.31    1.13
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0   1.04   1.12   1.31    1.13
 ALL                              5      0      0   1.04   1.12   1.31    1.13
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   5      0      0   1.04   1.12   1.31    1.13
 ALL                              5      0      0   1.04   1.12   1.31    1.13
```


worst 10: 512x256 (chain 1.04), 1024x1024 (chain 1.08), 2048x1024 (chain 1.12), 1024x512 (chain 1.13), 2048x2048 (chain 1.31)
best 5: 2048x2048 (chain 1.31), 1024x512 (chain 1.13), 2048x1024 (chain 1.12), 1024x1024 (chain 1.08), 512x256 (chain 1.04)
