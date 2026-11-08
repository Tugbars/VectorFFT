# gauntlet report (2D)

run: `gnr_2d_retime3_20261108`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 8 listed, 8 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.951..1.056


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x256  2            csk    replayed         368       2242    5.54     62.6  2.1e-16
      2x4096  2            csk    replayed       12783      35496    2.72     41.7  3.3e-16
         4x2  2^2          csk+rb replayed          16        129    7.61      7.4  1.4e-16
       64x64  2^6          chain+rb2 replayed        5719       5675    0.96     43.0  3.7e-16
     256x256  2^8          chain+rb2 replayed      185290     139481    0.71     28.3  4.4e-16
     512x512  2^9          chain  replayed      970051    1208344    1.08     24.3  4.1e-16
    1024x256  2^10         chain  replayed     1103244    1269983    0.85     21.4  4.3e-16
   1024x1024  2^10         chain  replayed     4635451    7809729    1.04     22.6  4.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            3      0      1   0.85   1.04   1.08    0.98
 csk                              2      0      0   2.72   4.13   5.54    3.88
 chain+rb2                        2      1      2   0.71   0.83   0.96    0.82
 csk+rb                           1      0      0   7.61   7.61   7.61    7.61
 ALL                              8      1      3   0.71   1.06   7.61    1.71
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      1      3   0.71   1.06   7.61    1.71
 ALL                              8      1      3   0.71   1.06   7.61    1.71
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   3      0      1   0.85   1.04   1.08    0.98
 4097..65536                      2      1      1   0.71   1.71   2.72    1.38
 257..1024                        1      0      0   5.54   5.54   5.54    5.54
 <= 256 points                    1      0      0   7.61   7.61   7.61    7.61
 1025..4096                       1      0      1   0.96   0.96   0.96    0.96
 ALL                              8      1      3   0.71   1.06   7.61    1.71
```


worst 10: 256x256 (chain+rb2 0.71), 1024x256 (chain 0.85), 64x64 (chain+rb2 0.96), 1024x1024 (chain 1.04), 512x512 (chain 1.08), 2x4096 (csk 2.72), 2x256 (csk 5.54), 4x2 (csk+rb 7.61)
best 5: 4x2 (csk+rb 7.61), 2x256 (csk 5.54), 2x4096 (csk 2.72), 512x512 (chain 1.08), 1024x1024 (chain 1.04)
