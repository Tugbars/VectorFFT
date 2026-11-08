# gauntlet report (2D)

run: `gnr_2d_retime2_20261108`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 8 listed, 8 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.933..1.040


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x256  2            csk    replayed         376       2200    5.76     61.3  2.1e-16
      2x4096  2            csk    replayed       13182      34713    2.57     40.4  3.3e-16
         4x2  2^2          csk+rb replayed          16        130    7.60      7.6  1.4e-16
       64x64  2^6          chain+rb2 replayed        5316       5553    0.99     46.2  3.7e-16
     256x256  2^8          chain+rb2 replayed      212266     162120    0.60     24.7  4.4e-16
     512x512  2^9          chain  replayed      936764    1225027    1.08     25.2  4.1e-16
    1024x256  2^10         chain  replayed     1325068    1212218    0.87     17.8  4.3e-16
   1024x1024  2^10         chain  replayed     4572222    6670471    1.14     22.9  4.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            3      0      1   0.87   1.08   1.14    1.02
 csk                              2      0      0   2.57   4.16   5.76    3.84
 chain+rb2                        2      1      2   0.60   0.79   0.99    0.77
 csk+rb                           1      0      0   7.60   7.60   7.60    7.60
 ALL                              8      1      3   0.60   1.11   7.60    1.70
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      1      3   0.60   1.11   7.60    1.70
 ALL                              8      1      3   0.60   1.11   7.60    1.70
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   3      0      1   0.87   1.08   1.14    1.02
 4097..65536                      2      1      1   0.60   1.58   2.57    1.24
 257..1024                        1      0      0   5.76   5.76   5.76    5.76
 <= 256 points                    1      0      0   7.60   7.60   7.60    7.60
 1025..4096                       1      0      1   0.99   0.99   0.99    0.99
 ALL                              8      1      3   0.60   1.11   7.60    1.70
```


worst 10: 256x256 (chain+rb2 0.60), 1024x256 (chain 0.87), 64x64 (chain+rb2 0.99), 512x512 (chain 1.08), 1024x1024 (chain 1.14), 2x4096 (csk 2.57), 2x256 (csk 5.76), 4x2 (csk+rb 7.60)
best 5: 4x2 (csk+rb 7.60), 2x256 (csk 5.76), 2x4096 (csk 2.57), 1024x1024 (chain 1.14), 512x512 (chain 1.08)
