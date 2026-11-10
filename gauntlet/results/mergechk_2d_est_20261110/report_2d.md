# gauntlet report (2D)

run: `mergechk_2d_est_20261110`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 5 listed, 5 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.391..1.399


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       64x64  2^6          chain+rb replayed        5119       7145    1.39     48.0  3.4e-16
    128x2048  2^7          chain  replayed      586000     819931    1.38     40.3  6.2e-16
     256x256  2^8          chain  replayed      110493     128695    1.16     47.5  3.3e-16
   1024x1024  2^10         chain  replayed     3494012    7821094    1.85     30.0  1.3e-15
    4096x256  2^12         chain  replayed     3364537    7810262    2.17     31.2  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            4      0      0   1.16   1.61   2.17    1.59
 chain+rb                         1      0      0   1.39   1.39   1.39    1.39
 ALL                              5      0      0   1.16   1.39   2.17    1.55
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0   1.16   1.39   2.17    1.55
 ALL                              5      0      0   1.16   1.39   2.17    1.55
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   3      0      0   1.38   1.85   2.17    1.77
 1025..4096                       1      0      0   1.39   1.39   1.39    1.39
 4097..65536                      1      0      0   1.16   1.16   1.16    1.16
 ALL                              5      0      0   1.16   1.39   2.17    1.55
```


worst 10: 256x256 (chain 1.16), 128x2048 (chain 1.38), 64x64 (chain+rb 1.39), 1024x1024 (chain 1.85), 4096x256 (chain 2.17)
best 5: 4096x256 (chain 2.17), 1024x1024 (chain 1.85), 64x64 (chain+rb 1.39), 128x2048 (chain 1.38), 256x256 (chain 1.16)
