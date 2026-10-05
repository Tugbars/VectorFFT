# gauntlet report (2D)

run: `fftw_2d_c2r_2026-10-05`  contract: 2D c2c interleaved, natural, out of place, K=1_c2r_fftw  cells: 5 listed, 4 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 1.215..1.345


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       32x63  2^5          2p     raced              -          -       -        - not benched
       32x64  2^5          chain  raced           1189       1466    1.20     47.4  6.1e-16
       64x64  2^6          chain  raced           2746       3613    1.18     44.8  8.9e-16
      64x256  2^6          chain  raced          12095      11406    0.87     47.4  8.9e-16
     128x128  2^7          chain  raced          14110      13041    0.92     40.6  7.8e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            4      0      2   0.87   1.05   1.20    1.03
 ALL                              4      0      2   0.87   1.05   1.20    1.03
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      4      0      2   0.87   1.05   1.20    1.03
 ALL                              4      0      2   0.87   1.05   1.20    1.03
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 1025..4096                       2      0      0   1.18   1.19   1.20    1.19
 4097..65536                      2      0      2   0.87   0.89   0.92    0.89
 ALL                              4      0      2   0.87   1.05   1.20    1.03
```


worst 10: 64x256 (chain 0.87), 128x128 (chain 0.92), 64x64 (chain 1.18), 32x64 (chain 1.20)
best 5: 32x64 (chain 1.20), 64x64 (chain 1.18), 128x128 (chain 0.92), 64x256 (chain 0.87)
