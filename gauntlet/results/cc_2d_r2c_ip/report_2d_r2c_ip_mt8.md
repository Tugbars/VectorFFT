# gauntlet report (2D)

run: `cc_2d_r2c_ip`  contract: 2D r2c interleaved, natural, out of place, K=1_r2c_ip_mt8  cells: 16 listed, 16 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 3 readings, 0.740..0.767


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            803        388    0.47      5.9  2.7e-16
     16x1000  2^4          chain  raced          12373       5783    0.47     45.1  4.6e-16
     16x1024  2^4          chain  raced           4179       4604    1.10    137.2  5.2e-16
       17x64  17           chain  raced           2064       1845    0.86     13.3  6.4e-16
     32x1024  2^5          chain  raced           8656      11724    1.04    142.0  4.7e-16
       64x15  2^6          chain  raced           1573       4058    2.58     15.1  4.6e-16
       64x30  2^6          chain  raced           2560       1835    0.71     20.4  3.3e-16
       64x64  2^6          chain  raced           4452       3523    0.74     27.6  3.3e-16
      64x256  2^6          chain  raced          11568       4514    0.38     49.6  5.2e-16
     128x128  2^7          chain  raced           5690       6438    1.13    100.8  3.6e-16
     128x512  2^7          chain  raced          21111      17251    0.82    124.2  4.1e-16
     256x256  2^8          chain  raced          19520      24422    1.19    134.3  4.1e-16
    256x1024  2^8          chain  raced          64533      57300    0.89    182.8  4.9e-16
     512x512  2^9          chain  raced          63580      71070    1.05    185.5  5.1e-16
   1024x1024  2^10         chain  raced         211200     272112    1.22    248.2  5.6e-16
   2048x2048  2^11         chain  raced        2956075    1062500    0.36     78.0  4.8e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      6      9   0.38   0.87   1.22    0.83
 ALL                             16      6      9   0.38   0.87   1.22    0.83
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      5      7   0.38   0.96   1.22    0.86
 odd column                       1      1      1   0.47   0.47   0.47    0.47
 prime column                     1      0      1   0.86   0.86   0.86    0.86
 ALL                             16      6      9   0.38   0.87   1.22    0.83
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      2      3   0.38   1.04   1.19    0.81
 > 65536 points                   4      1      2   0.36   0.97   1.22    0.80
 1025..4096                       3      2      3   0.71   0.74   0.86    0.77
 <= 256 points                    1      1      1   0.47   0.47   0.47    0.47
 257..1024                        1      0      0   2.58   2.58   2.58    2.58
 ALL                             16      6      9   0.38   0.87   1.22    0.83
```


worst 10: 2048x2048 (chain 0.36), 64x256 (chain 0.38), 16x1000 (chain 0.47), 15x16 (chain 0.47), 64x30 (chain 0.71), 64x64 (chain 0.74), 128x512 (chain 0.82), 17x64 (chain 0.86), 256x1024 (chain 0.89), 32x1024 (chain 1.04)
best 5: 64x15 (chain 2.58), 1024x1024 (chain 1.22), 256x256 (chain 1.19), 128x128 (chain 1.13), 16x1024 (chain 1.10)
