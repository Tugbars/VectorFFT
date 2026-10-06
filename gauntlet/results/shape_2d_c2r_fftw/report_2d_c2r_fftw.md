# gauntlet report (2D)

run: `shape_2d_c2r_fftw`  contract: 2D c2c interleaved, natural, out of place, K=1_c2r_fftw  cells: 5 listed, 5 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 1.297..1.378


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     512x256  2^9          chain  raced         138257     147888    1.03     40.3  1.1e-15
    1024x512  2^10         chain  raced         640700     738237    1.10     38.9  1.2e-15
   1024x1024  2^10         chain  raced        1507188    1509393    0.98     34.8  1.3e-15
   2048x1024  2^11         chain  raced        4227200    4610362    1.08     26.0  1.3e-15
   2048x2048  2^11         chain  raced        9253037   11312175    1.21     24.9  1.5e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            5      0      1   0.98   1.08   1.21    1.08
 ALL                              5      0      1   0.98   1.08   1.21    1.08
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      1   0.98   1.08   1.21    1.08
 ALL                              5      0      1   0.98   1.08   1.21    1.08
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   5      0      1   0.98   1.08   1.21    1.08
 ALL                              5      0      1   0.98   1.08   1.21    1.08
```


worst 10: 1024x1024 (chain 0.98), 512x256 (chain 1.03), 2048x1024 (chain 1.08), 1024x512 (chain 1.10), 2048x2048 (chain 1.21)
best 5: 2048x2048 (chain 1.21), 1024x512 (chain 1.10), 2048x1024 (chain 1.08), 512x256 (chain 1.03), 1024x1024 (chain 0.98)
