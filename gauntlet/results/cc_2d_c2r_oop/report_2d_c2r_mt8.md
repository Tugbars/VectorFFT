# gauntlet report (2D)

run: `cc_2d_c2r_oop`  contract: 2D c2r interleaved, natural, out of place, K=1_c2r_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 3 readings, 0.310..0.365


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            286        328    1.15     16.6  4.9e-16
     16x1000  2^4          chain  raced          21071       6441    0.30     26.5  9.9e-16
     16x1024  2^4          chain  raced           5369       8075    1.50    106.8  1.1e-15
       17x64  17           chain  raced           1528       2505    1.16     18.0  1.0e-15
     32x1024  2^5          chain  raced           7529      11428    0.81    163.2  8.9e-16
       64x15  2^6          chain  raced           2006       4381    2.18     11.9  6.4e-16
       64x30  2^6          chain  raced           2677       3457    1.27     19.6  8.0e-16
       64x64  2^6          chain  raced           5836       2046    0.28     21.1  7.8e-16
      64x256  2^6          chain  raced           9793       5749    0.51     58.6  7.8e-16
     128x128  2^7          chain  raced          24127       6121    0.25     23.8  1.0e-15
     128x512  2^7          chain  raced          13828      17800    1.29    189.6  1.0e-15
     256x256  2^8          chain  raced          17280      19233    1.11    151.7  1.0e-15
    256x1024  2^8          chain  raced          59973      68333    1.06    196.7  1.1e-15
     512x512  2^9          chain  raced          62273      75373    1.12    189.4  1.3e-15
   1024x1024  2^10         chain  raced         258825     335863    1.30    202.6  1.3e-15
   2048x2048  2^11         chain  raced        4429475    2754600    0.62     52.1  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      5      6   0.28   1.12   1.50    0.84
 ALL                             16      5      6   0.28   1.12   1.50    0.84
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      5      6   0.28   1.09   1.50    0.81
 odd column                       1      0      0   1.15   1.15   1.15    1.15
 prime column                     1      0      0   1.16   1.16   1.16    1.16
 ALL                             16      5      6   0.28   1.12   1.50    0.84
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      3      4   0.25   0.81   1.50    0.68
 > 65536 points                   4      1      1   0.62   1.09   1.30    0.99
 1025..4096                       3      1      1   0.28   1.16   1.27    0.74
 <= 256 points                    1      0      0   1.15   1.15   1.15    1.15
 257..1024                        1      0      0   2.18   2.18   2.18    2.18
 ALL                             16      5      6   0.28   1.12   1.50    0.84
```


worst 10: 128x128 (chain 0.25), 64x64 (chain 0.28), 16x1000 (chain 0.30), 64x256 (chain 0.51), 2048x2048 (chain 0.62), 32x1024 (chain 0.81), 256x1024 (chain 1.06), 256x256 (chain 1.11), 512x512 (chain 1.12), 15x16 (chain 1.15)
best 5: 64x15 (chain 2.18), 16x1024 (chain 1.50), 1024x1024 (chain 1.30), 128x512 (chain 1.29), 64x30 (chain 1.27)
