# gauntlet report (2D)

run: `rerace_2d_c2r_fftw_t2cb44`  contract: 2D c2c interleaved, natural, out of place, K=1_c2r_fftw  cells: 5 listed, 5 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 1.235..1.336


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     512x256  2^9          chain  raced         143407     148625    1.03     38.8  1.1e-15
    1024x512  2^10         chain  raced         645612     726787    1.11     38.6  1.1e-15
   1024x1024  2^10         chain  raced        1290687    1518725    1.08     40.6  1.4e-15
   2048x1024  2^11         chain  raced        4465187    4569375    1.00     24.7  1.3e-15
   2048x2048  2^11         chain  raced        9251300   11374112    1.20     24.9  1.7e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            5      0      0   1.00   1.08   1.20    1.08
 ALL                              5      0      0   1.00   1.08   1.20    1.08
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0   1.00   1.08   1.20    1.08
 ALL                              5      0      0   1.00   1.08   1.20    1.08
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   5      0      0   1.00   1.08   1.20    1.08
 ALL                              5      0      0   1.00   1.08   1.20    1.08
```


worst 10: 2048x1024 (chain 1.00), 512x256 (chain 1.03), 1024x1024 (chain 1.08), 1024x512 (chain 1.11), 2048x2048 (chain 1.20)
best 5: 2048x2048 (chain 1.20), 1024x512 (chain 1.11), 1024x1024 (chain 1.08), 512x256 (chain 1.03), 2048x1024 (chain 1.00)
