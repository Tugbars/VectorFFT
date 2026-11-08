# gauntlet report (2D)

run: `gnr_2d_retime1_20261108`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 8 listed, 8 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.904..0.992


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x256  2            csk    replayed         372       2441    5.97     62.0  2.1e-16
      2x4096  2            csk    replayed       12997      35460    2.69     41.0  3.3e-16
         4x2  2^2          csk+rb replayed          17        134    7.78      7.1  1.4e-16
       64x64  2^6          chain+rb2 replayed        5735       5540    0.94     42.9  3.7e-16
     256x256  2^8          chain+rb2 replayed      185155     135651    0.73     28.3  4.4e-16
     512x512  2^9          chain  replayed      982541    1248664    1.11     24.0  4.1e-16
    1024x256  2^10         chain  replayed     1436146    1262335    0.87     16.4  4.3e-16
   1024x1024  2^10         chain  replayed     4571755    5776521    0.95     22.9  4.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            3      0      2   0.87   0.95   1.11    0.97
 csk                              2      0      0   2.69   4.33   5.97    4.01
 chain+rb2                        2      1      2   0.73   0.84   0.94    0.83
 csk+rb                           1      0      0   7.78   7.78   7.78    7.78
 ALL                              8      1      4   0.73   1.03   7.78    1.73
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      1      4   0.73   1.03   7.78    1.73
 ALL                              8      1      4   0.73   1.03   7.78    1.73
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   3      0      2   0.87   0.95   1.11    0.97
 4097..65536                      2      1      1   0.73   1.71   2.69    1.40
 257..1024                        1      0      0   5.97   5.97   5.97    5.97
 <= 256 points                    1      0      0   7.78   7.78   7.78    7.78
 1025..4096                       1      0      1   0.94   0.94   0.94    0.94
 ALL                              8      1      4   0.73   1.03   7.78    1.73
```


worst 10: 256x256 (chain+rb2 0.73), 1024x256 (chain 0.87), 64x64 (chain+rb2 0.94), 1024x1024 (chain 0.95), 512x512 (chain 1.11), 2x4096 (csk 2.69), 2x256 (csk 5.97), 4x2 (csk+rb 7.78)
best 5: 4x2 (csk+rb 7.78), 2x256 (csk 5.97), 2x4096 (csk 2.69), 512x512 (chain 1.11), 1024x1024 (chain 0.95)
