# gauntlet report (2D)

run: `rerace_2d_c2c_mkl_t2cb44`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 5 listed, 5 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.075..1.340


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     512x256  2^9          chain  raced         261907     354853    1.32     42.5  4.6e-16
    1024x512  2^10         chain  raced        1189050    2197037    1.81     41.9  1.2e-15
   1024x1024  2^10         chain  raced        2893687    4364762    1.49     36.2  1.3e-15
   2048x1024  2^11         chain  raced        6962037   12725750    1.53     31.6  1.5e-15
   2048x2048  2^11         chain  raced       15920900   33031181    1.52     29.0  2.9e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            5      0      0   1.32   1.52   1.81    1.53
 ALL                              5      0      0   1.32   1.52   1.81    1.53
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0   1.32   1.52   1.81    1.53
 ALL                              5      0      0   1.32   1.52   1.81    1.53
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   5      0      0   1.32   1.52   1.81    1.53
 ALL                              5      0      0   1.32   1.52   1.81    1.53
```


worst 10: 512x256 (chain 1.32), 1024x1024 (chain 1.49), 2048x2048 (chain 1.52), 2048x1024 (chain 1.53), 1024x512 (chain 1.81)
best 5: 1024x512 (chain 1.81), 2048x1024 (chain 1.53), 2048x2048 (chain 1.52), 1024x1024 (chain 1.49), 512x256 (chain 1.32)
