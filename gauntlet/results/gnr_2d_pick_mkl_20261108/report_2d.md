# gauntlet report (2D)

run: `gnr_2d_pick_mkl_20261108`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 8 listed, 8 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.943..1.035


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x256  2            csk    raced            382       2286    5.97     60.4  2.1e-16
      2x4096  2            csk    raced          13266      36131    2.71     40.1  3.3e-16
         4x2  2^2          csk+rb raced             17        136    7.92      7.0  1.4e-16
       64x64  2^6          chain+rb2 raced           5495       5601    1.00     44.7  3.7e-16
     256x256  2^8          chain+rb2 raced         277416     292354    0.58     18.9  4.4e-16
     512x512  2^9          chain  raced        1418518    1851182    1.00     16.6  4.1e-16
    1024x256  2^10         chain  raced        2539208    2516473    0.93      9.3  4.3e-16
   1024x1024  2^10         chain  raced        4596826    6780162    1.09     22.8  4.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            3      0      2   0.93   1.00   1.09    1.00
 csk                              2      0      0   2.71   4.34   5.97    4.02
 chain+rb2                        2      1      2   0.58   0.79   1.00    0.76
 csk+rb                           1      0      0   7.92   7.92   7.92    7.92
 ALL                              8      1      4   0.58   1.04   7.92    1.71
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      1      4   0.58   1.04   7.92    1.71
 ALL                              8      1      4   0.58   1.04   7.92    1.71
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   3      0      2   0.93   1.00   1.09    1.00
 4097..65536                      2      1      1   0.58   1.65   2.71    1.25
 257..1024                        1      0      0   5.97   5.97   5.97    5.97
 <= 256 points                    1      0      0   7.92   7.92   7.92    7.92
 1025..4096                       1      0      1   1.00   1.00   1.00    1.00
 ALL                              8      1      4   0.58   1.04   7.92    1.71
```


worst 10: 256x256 (chain+rb2 0.58), 1024x256 (chain 0.93), 64x64 (chain+rb2 1.00), 512x512 (chain 1.00), 1024x1024 (chain 1.09), 2x4096 (csk 2.71), 2x256 (csk 5.97), 4x2 (csk+rb 7.92)
best 5: 4x2 (csk+rb 7.92), 2x256 (csk 5.97), 2x4096 (csk 2.71), 1024x1024 (chain 1.09), 512x512 (chain 1.00)
