# gauntlet report (2D)

run: `rb_2d_c2c_mkl_replay`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 10 listed, 10 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.142..1.348


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       16x16  2^4          csk+rb2 replayed         153        147    0.93     66.7  2.5e-16
       16x64  2^4          csk    replayed         746        719    0.96     68.6  3.0e-16
      16x128  2^4          csk    replayed        1764       1612    0.82     63.9  3.1e-16
       32x16  2^5          chain+rb replayed         346        341    0.98     66.6  2.3e-16
       32x32  2^5          chain+rb2 replayed         733        720    0.98     69.9  2.4e-16
       32x64  2^5          chain+rb2 replayed        2731       1878    0.68     41.3  2.8e-16
       64x32  2^6          chain+rb2 replayed        2058       1997    0.76     54.7  3.0e-16
      64x128  2^6          csk    replayed       11027      13942    1.22     48.3  4.6e-16
      128x16  2^7          turn   replayed        2536       1984    0.75     44.4  2.3e-16
    256x8192  2^8          chain  replayed     8088400    8180737    0.96     27.2  1.6e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 csk                              3      0      2   0.82   0.96   1.22    0.98
 chain+rb2                        3      2      3   0.68   0.76   0.98    0.80
 csk+rb2                          1      0      1   0.93   0.93   0.93    0.93
 chain+rb                         1      0      1   0.98   0.98   0.98    0.98
 turn                             1      1      1   0.75   0.75   0.75    0.75
 chain                            1      0      1   0.96   0.96   0.96    0.96
 ALL                             10      3      9   0.75   0.94   1.22    0.89
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      3      9   0.75   0.94   1.22    0.89
 ALL                             10      3      9   0.75   0.94   1.22    0.89
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 1025..4096                       4      3      4   0.68   0.75   0.82    0.75
 257..1024                        3      0      3   0.96   0.98   0.98    0.97
 <= 256 points                    1      0      1   0.93   0.93   0.93    0.93
 4097..65536                      1      0      0   1.22   1.22   1.22    1.22
 > 65536 points                   1      0      1   0.96   0.96   0.96    0.96
 ALL                             10      3      9   0.75   0.94   1.22    0.89
```


worst 10: 32x64 (chain+rb2 0.68), 128x16 (turn 0.75), 64x32 (chain+rb2 0.76), 16x128 (csk 0.82), 16x16 (csk+rb2 0.93), 16x64 (csk 0.96), 256x8192 (chain 0.96), 32x32 (chain+rb2 0.98), 32x16 (chain+rb 0.98), 64x128 (csk 1.22)
best 5: 64x128 (csk 1.22), 32x16 (chain+rb 0.98), 32x32 (chain+rb2 0.98), 16x64 (csk 0.96), 256x8192 (chain 0.96)
