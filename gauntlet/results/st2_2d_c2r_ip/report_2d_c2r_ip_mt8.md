# gauntlet report (2D)

run: `st2_2d_c2r_ip`  contract: 2D c2r interleaved, natural, out of place, K=1_c2r_ip_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.042..1.193


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced           1107        409    0.37      4.3  8.5e-16
     16x1000  2^4          chain  raced           8377       5154    0.61     66.7  9.9e-16
     16x1024  2^4          chain  raced          17930       4408    0.12     32.0  1.1e-15
       17x64  17           chain  raced           1579       1867    1.10     17.4  1.1e-15
     32x1024  2^5          chain  raced          31900       8760    0.28     38.5  1.0e-15
       64x15  2^6          chain  raced           1933       4199    2.17     12.3  6.4e-16
       64x30  2^6          chain  raced           2259       1933    0.73     23.2  7.3e-16
       64x64  2^6          chain  raced           2800       3297    1.16     43.9  8.9e-16
      64x256  2^6          chain  raced           7668       4839    0.63     74.8  8.9e-16
     128x128  2^7          chain  raced           9720       6114    0.59     59.0  1.0e-15
     128x512  2^7          chain  raced          15689      17967    0.89    167.1  1.0e-15
     256x256  2^8          chain  raced          22772      24295    1.07    115.1  1.0e-15
    256x1024  2^8          chain  raced          62473      54533    0.87    188.8  1.2e-15
     512x512  2^9          chain  raced          71627      70816    0.97    164.7  1.2e-15
   1024x1024  2^10         chain  raced         196525     256899    1.05    266.8  1.3e-15
   2048x2048  2^11         chain  raced        2295212     983612    0.43    100.5  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      8     11   0.28   0.80   1.16    0.68
 ALL                             16      8     11   0.28   0.80   1.16    0.68
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      7     10   0.28   0.80   1.16    0.69
 odd column                       1      1      1   0.37   0.37   0.37    0.37
 prime column                     1      0      0   1.10   1.10   1.10    1.10
 ALL                             16      8     11   0.28   0.80   1.16    0.68
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      5      6   0.12   0.61   1.07    0.49
 > 65536 points                   4      1      3   0.43   0.92   1.05    0.79
 1025..4096                       3      1      1   0.73   1.10   1.16    0.98
 <= 256 points                    1      1      1   0.37   0.37   0.37    0.37
 257..1024                        1      0      0   2.17   2.17   2.17    2.17
 ALL                             16      8     11   0.28   0.80   1.16    0.68
```


worst 10: 16x1024 (chain 0.12), 32x1024 (chain 0.28), 15x16 (chain 0.37), 2048x2048 (chain 0.43), 128x128 (chain 0.59), 16x1000 (chain 0.61), 64x256 (chain 0.63), 64x30 (chain 0.73), 256x1024 (chain 0.87), 128x512 (chain 0.89)
best 5: 64x15 (chain 2.17), 64x64 (chain 1.16), 17x64 (chain 1.10), 256x256 (chain 1.07), 1024x1024 (chain 1.05)
