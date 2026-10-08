# gauntlet report (2D)

run: `cr_2d_r2c_oop`  contract: 2D r2c interleaved, natural, out of place, K=1_r2c_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 3 readings, 0.777..0.784


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            229        376    1.64     20.7  2.8e-16
     16x1000  2^4          chain  raced           4614       6430    1.24    121.1  4.6e-16
     16x1024  2^4          chain  raced           3967       7456    1.39    144.6  5.2e-16
       17x64  17           chain  raced           1593       1832    0.86     17.2  6.4e-16
     32x1024  2^5          chain  raced           7080      15693    1.55    173.6  4.7e-16
       64x15  2^6          chain  raced           1439       3781    2.63     16.5  3.7e-16
       64x30  2^6          chain  raced           2318       1652    0.71     22.6  3.3e-16
       64x64  2^6          chain  raced           4008       3220    0.80     30.7  3.0e-16
      64x256  2^6          chain  raced           4960       5333    0.88    115.6  3.9e-16
     128x128  2^7          chain  raced           7534       7528    0.96     76.1  3.3e-16
     128x512  2^7          chain  raced          15507      16254    0.76    169.1  5.0e-16
     256x256  2^8          chain  raced          19736      23805    1.08    132.8  4.3e-16
    256x1024  2^8          chain  raced          56800      62503    0.93    207.7  6.1e-16
     512x512  2^9          chain  raced          60767      69973    1.08    194.1  5.1e-16
   1024x1024  2^10         chain  raced         188837     286894    1.24    277.6  5.1e-16
   2048x2048  2^11         chain  raced        2722187    1462050    0.54     84.7  5.6e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      4      8   0.71   1.02   1.64    1.06
 ALL                             16      4      8   0.71   1.02   1.64    1.06
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      4      7   0.71   1.02   1.55    1.05
 odd column                       1      0      0   1.64   1.64   1.64    1.64
 prime column                     1      0      1   0.86   0.86   0.86    0.86
 ALL                             16      4      8   0.71   1.02   1.64    1.06
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      1      3   0.76   1.08   1.55    1.09
 > 65536 points                   4      1      2   0.54   1.01   1.24    0.90
 1025..4096                       3      2      3   0.71   0.80   0.86    0.79
 <= 256 points                    1      0      0   1.64   1.64   1.64    1.64
 257..1024                        1      0      0   2.63   2.63   2.63    2.63
 ALL                             16      4      8   0.71   1.02   1.64    1.06
```


worst 10: 2048x2048 (chain 0.54), 64x30 (chain 0.71), 128x512 (chain 0.76), 64x64 (chain 0.80), 17x64 (chain 0.86), 64x256 (chain 0.88), 256x1024 (chain 0.93), 128x128 (chain 0.96), 512x512 (chain 1.08), 256x256 (chain 1.08)
best 5: 64x15 (chain 2.63), 15x16 (chain 1.64), 32x1024 (chain 1.55), 16x1024 (chain 1.39), 16x1000 (chain 1.24)
