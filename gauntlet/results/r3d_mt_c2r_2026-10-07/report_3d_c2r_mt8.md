# gauntlet report (3D)

run: `r3d_mt_c2r_2026-10-07`  contract: 3D c2r interleaved, natural, out of place, K=1_c2r_mt8  cells: 4 listed, 4 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.034..1.093

threaded plans that ran serial (engaged = 0 at both flips): 0


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  8x128x2048  2^3          payonce raced         472350     789331    1.53    233.1  1.3e-15
  16x256x256  2^4          payonce raced         176112     246712    1.34    297.7  1.4e-15
    64x64x64  2^6          band   raced          50100      53870    1.07    235.5  1.1e-15
 128x128x128  2^7          band   raced         479150     605781    0.97    229.8  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 payonce                          2      0      0   1.34   1.43   1.53    1.43
 band                             2      0      1   0.97   1.02   1.07    1.02
 ALL                              4      0      1   0.97   1.20   1.53    1.21
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      4      0      1   0.97   1.20   1.53    1.21
 ALL                              4      0      1   0.97   1.20   1.53    1.21
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   4      0      1   0.97   1.20   1.53    1.21
 ALL                              4      0      1   0.97   1.20   1.53    1.21
```


worst 10: 128x128x128 (band 0.97), 64x64x64 (band 1.07), 16x256x256 (payonce 1.34), 8x128x2048 (payonce 1.53)
best 5: 8x128x2048 (payonce 1.53), 16x256x256 (payonce 1.34), 64x64x64 (band 1.07), 128x128x128 (band 0.97)
