# gauntlet report (2D)

run: `rb_2d_c2c_mkl_rerace`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 10 listed, 10 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.098..1.289


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       16x16  2^4          csk+rb2 raced            154        146    0.94     66.4  2.5e-16
       16x64  2^4          chain+rb2 raced            717        714    1.00     71.5  2.7e-16
      16x128  2^4          csk    raced           1816       1611    0.86     62.0  3.1e-16
       32x16  2^5          chain+rb raced            346        340    0.98     66.6  2.3e-16
       32x32  2^5          chain+rb2 raced            711        726    1.01     72.0  2.4e-16
       32x64  2^5          csk    raced           2230       1930    0.84     50.5  2.7e-16
       64x32  2^6          chain+rb2 raced           2032       1998    0.98     55.4  3.0e-16
      64x128  2^6          chain  raced          11352      14000    1.22     46.9  3.1e-16
      128x16  2^7          turn   raced           2621       1991    0.67     43.0  2.3e-16
    256x8192  2^8          chain  raced        6986388    8244081    1.16     31.5  1.5e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain+rb2                        3      0      2   0.98   1.00   1.01    1.00
 csk                              2      0      2   0.84   0.85   0.86    0.85
 chain                            2      0      0   1.16   1.19   1.22    1.19
 csk+rb2                          1      0      1   0.94   0.94   0.94    0.94
 chain+rb                         1      0      1   0.98   0.98   0.98    0.98
 turn                             1      1      1   0.67   0.67   0.67    0.67
 ALL                             10      1      7   0.84   0.98   1.22    0.95
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      1      7   0.84   0.98   1.22    0.95
 ALL                             10      1      7   0.84   0.98   1.22    0.95
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 1025..4096                       4      1      4   0.67   0.85   0.98    0.83
 257..1024                        3      0      2   0.98   1.00   1.01    1.00
 <= 256 points                    1      0      1   0.94   0.94   0.94    0.94
 4097..65536                      1      0      0   1.22   1.22   1.22    1.22
 > 65536 points                   1      0      0   1.16   1.16   1.16    1.16
 ALL                             10      1      7   0.84   0.98   1.22    0.95
```


worst 10: 128x16 (turn 0.67), 32x64 (csk 0.84), 16x128 (csk 0.86), 16x16 (csk+rb2 0.94), 64x32 (chain+rb2 0.98), 32x16 (chain+rb 0.98), 16x64 (chain+rb2 1.00), 32x32 (chain+rb2 1.01), 256x8192 (chain 1.16), 64x128 (chain 1.22)
best 5: 64x128 (chain 1.22), 256x8192 (chain 1.16), 32x32 (chain+rb2 1.01), 16x64 (chain+rb2 1.00), 32x16 (chain+rb 0.98)
