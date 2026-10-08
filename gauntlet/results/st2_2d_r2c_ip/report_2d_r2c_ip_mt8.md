# gauntlet report (2D)

run: `st2_2d_r2c_ip`  contract: 2D r2c interleaved, natural, out of place, K=1_r2c_ip_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.975..1.231


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            952        436    0.35      5.0  3.1e-16
     16x1000  2^4          chain  raced           5214       5319    0.97    107.1  4.6e-16
     16x1024  2^4          chain  raced           4458       5362    1.15    128.6  3.9e-16
       17x64  17           chain  raced           1809       2025    1.12     15.2  6.0e-16
     32x1024  2^5          chain  raced           8270       9706    1.17    148.6  4.7e-16
       64x15  2^6          chain  raced           1775       4717    2.06     13.4  3.7e-16
       64x30  2^6          chain  raced          20985       1964    0.09      2.5  3.3e-16
       64x64  2^6          chain  raced           3008       3689    1.23     40.9  2.7e-16
      64x256  2^6          chain  raced           6453      19065    0.15     88.9  3.9e-16
     128x128  2^7          chain  raced           6683       7324    1.01     85.8  3.3e-16
     128x512  2^7          chain  raced          17487      18446    1.02    149.9  3.8e-16
     256x256  2^8          chain  raced          21782      24049    1.10    120.3  4.7e-16
    256x1024  2^8          chain  raced          57293      64060    1.04    205.9  4.9e-16
     512x512  2^9          chain  raced          72313      78440    1.08    163.1  5.1e-16
   1024x1024  2^10         chain  raced         289025     285763    0.99    181.4  5.6e-16
   2048x2048  2^11         chain  raced        1661688    1710537    1.03    138.8  5.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      3      5   0.15   1.04   1.23    0.79
 ALL                             16      3      5   0.15   1.04   1.23    0.79
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      2      4   0.15   1.04   1.23    0.82
 odd column                       1      1      1   0.35   0.35   0.35    0.35
 prime column                     1      0      0   1.12   1.12   1.12    1.12
 ALL                             16      3      5   0.15   1.04   1.23    0.79
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      1      2   0.15   1.02   1.17    0.80
 > 65536 points                   4      0      1   0.99   1.04   1.08    1.04
 1025..4096                       3      1      1   0.09   1.12   1.23    0.51
 <= 256 points                    1      1      1   0.35   0.35   0.35    0.35
 257..1024                        1      0      0   2.06   2.06   2.06    2.06
 ALL                             16      3      5   0.15   1.04   1.23    0.79
```


worst 10: 64x30 (chain 0.09), 64x256 (chain 0.15), 15x16 (chain 0.35), 16x1000 (chain 0.97), 1024x1024 (chain 0.99), 128x128 (chain 1.01), 128x512 (chain 1.02), 2048x2048 (chain 1.03), 256x1024 (chain 1.04), 512x512 (chain 1.08)
best 5: 64x15 (chain 2.06), 64x64 (chain 1.23), 32x1024 (chain 1.17), 16x1024 (chain 1.15), 17x64 (chain 1.12)
