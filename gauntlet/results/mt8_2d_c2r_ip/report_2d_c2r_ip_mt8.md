# gauntlet report (2D)

run: `mt8_2d_c2r_ip`  contract: 2D c2r interleaved, natural, out of place, K=1_c2r_ip_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 3 readings, 0.327..0.547


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            806        407    0.38      5.9  4.9e-16
     16x1000  2^4          chain  raced          35577       5053    0.14     15.7  9.9e-16
     16x1024  2^4          chain  raced          17951       4375    0.24     31.9  7.8e-16
       17x64  17           chain  raced           2009       1862    0.93     13.7  1.0e-15
     32x1024  2^5          chain  raced           9110      10543    1.03    134.9  1.0e-15
       64x15  2^6          chain  raced           1920       4117    2.14     12.4  5.9e-16
       64x30  2^6          chain  raced           4014       1933    0.47     13.0  7.7e-16
       64x64  2^6          chain  raced           6198       3221    0.48     19.8  7.8e-16
      64x256  2^6          chain  raced          23595       4743    0.13     24.3  8.9e-16
     128x128  2^7          chain  raced           9440       5371    0.55     60.7  8.9e-16
     128x512  2^7          chain  raced          20326      16100    0.58    129.0  1.0e-15
     256x256  2^8          chain  raced          21046      23340    0.90    124.6  1.0e-15
    256x1024  2^8          chain  raced         162987      55556    0.32     72.4  1.2e-15
     512x512  2^9          chain  raced         142393      71280    0.48     82.8  1.1e-15
   1024x1024  2^10         chain  raced         194163     243050    1.18    270.0  1.3e-15
   2048x2048  2^11         chain  raced        3600212    1012262    0.28     64.1  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16     11     13   0.14   0.48   1.18    0.49
 ALL                             16     11     13   0.14   0.48   1.18    0.49
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14     10     11   0.14   0.48   1.18    0.48
 odd column                       1      1      1   0.38   0.38   0.38    0.38
 prime column                     1      0      1   0.93   0.93   0.93    0.93
 ALL                             16     11     13   0.14   0.48   1.18    0.49
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      5      6   0.13   0.55   1.03    0.39
 > 65536 points                   4      3      3   0.28   0.40   1.18    0.47
 1025..4096                       3      2      3   0.47   0.48   0.93    0.59
 <= 256 points                    1      1      1   0.38   0.38   0.38    0.38
 257..1024                        1      0      0   2.14   2.14   2.14    2.14
 ALL                             16     11     13   0.14   0.48   1.18    0.49
```


worst 10: 64x256 (chain 0.13), 16x1000 (chain 0.14), 16x1024 (chain 0.24), 2048x2048 (chain 0.28), 256x1024 (chain 0.32), 15x16 (chain 0.38), 64x30 (chain 0.47), 512x512 (chain 0.48), 64x64 (chain 0.48), 128x128 (chain 0.55)
best 5: 64x15 (chain 2.14), 1024x1024 (chain 1.18), 32x1024 (chain 1.03), 17x64 (chain 0.93), 256x256 (chain 0.90)
