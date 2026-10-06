# gauntlet report (2D)

run: `race_2d_c2r_fftw_2026-10-06`  contract: 2D c2c interleaved, natural, out of place, K=1_c2r_fftw  cells: 5 listed, 5 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 1.242..1.293


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     512x256  2^9          chain  raced         141443     146796    1.02     39.4  1.0e-15
    1024x512  2^10         chain  raced         653787     739662    1.11     38.1  1.2e-15
   1024x1024  2^10         chain  raced        1317925    1516031    1.13     39.8  1.3e-15
   2048x1024  2^11         chain  raced        3953225    4303775    1.06     27.9  1.6e-15
   2048x2048  2^11         chain  raced        9406725   11540937    1.20     24.5  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            5      0      0   1.02   1.11   1.20    1.10
 ALL                              5      0      0   1.02   1.11   1.20    1.10
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0   1.02   1.11   1.20    1.10
 ALL                              5      0      0   1.02   1.11   1.20    1.10
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   5      0      0   1.02   1.11   1.20    1.10
 ALL                              5      0      0   1.02   1.11   1.20    1.10
```


worst 10: 512x256 (chain 1.02), 2048x1024 (chain 1.06), 1024x512 (chain 1.11), 1024x1024 (chain 1.13), 2048x2048 (chain 1.20)
best 5: 2048x2048 (chain 1.20), 1024x1024 (chain 1.13), 1024x512 (chain 1.11), 2048x1024 (chain 1.06), 512x256 (chain 1.02)
