# gauntlet report (2D)

run: `race_2d_c2c_mkl_2026-10-06`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 5 listed, 5 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.076..1.356


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     512x256  2^9          chain  raced         243813     361307    1.46     45.7  3.4e-16
    1024x512  2^10         chain  raced        1204788    2202756    1.76     41.3  1.2e-15
   1024x1024  2^10         chain  raced        2975725    4748775    1.49     35.2  1.3e-15
   2048x1024  2^11         chain  raced        6969850   12916337    1.57     31.6  1.5e-15
   2048x2048  2^11         chain  raced       16092712   24028631    1.47     28.7  3.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            5      0      0   1.46   1.49   1.76    1.55
 ALL                              5      0      0   1.46   1.49   1.76    1.55
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0   1.46   1.49   1.76    1.55
 ALL                              5      0      0   1.46   1.49   1.76    1.55
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   5      0      0   1.46   1.49   1.76    1.55
 ALL                              5      0      0   1.46   1.49   1.76    1.55
```


worst 10: 512x256 (chain 1.46), 2048x2048 (chain 1.47), 1024x1024 (chain 1.49), 2048x1024 (chain 1.57), 1024x512 (chain 1.76)
best 5: 1024x512 (chain 1.76), 2048x1024 (chain 1.57), 1024x1024 (chain 1.49), 2048x2048 (chain 1.47), 512x256 (chain 1.46)
