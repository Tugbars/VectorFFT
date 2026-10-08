# gauntlet report (2D)

run: `ip2d_c2r_ipo_fftw`  contract: 2D c2r interleaved, natural, out of place, K=1_c2r_ipo_fftw  cells: 16 listed, 16 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 1.254..1.300


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            224        146    0.60     21.1  6.8e-16
     16x1000  2^4          chain  raced          13139      13835    1.04     42.5  1.0e-15
     16x1024  2^4          chain  raced           9884      11760    1.18     58.0  7.8e-16
       17x64  17           chain  raced            711       2040    2.80     38.6  9.1e-16
     32x1024  2^5          chain  raced          21578      24548    1.13     56.9  1.3e-15
       64x15  2^6          chain  raced           1919        653    0.34     12.4  7.1e-16
       64x30  2^6          chain  raced           2084       1660    0.80     25.1  7.4e-16
       64x64  2^6          chain  raced           2473       3269    1.28     49.7  8.1e-16
      64x256  2^6          chain  raced          11198      12636    1.12     51.2  8.9e-16
     128x128  2^7          chain  raced          11602      13570    1.16     49.4  8.9e-16
     128x512  2^7          chain  raced          49233      54071    1.10     53.2  1.3e-15
     256x256  2^8          chain  raced          65608      73158    1.08     40.0  1.0e-15
    256x1024  2^8          chain  raced         288460     311220    1.05     40.9  1.1e-15
     512x512  2^9          chain  replayed      284480     337940    1.07     41.5  1.1e-15
   1024x1024  2^10         chain  raced        1163500    1490343    1.27     45.1  1.4e-15
   2048x2048  2^11         chain  raced        7298188    9875806    1.31     31.6  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      3      3   0.60   1.11   1.31    1.05
 ALL                             16      3      3   0.60   1.11   1.31    1.05
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      2      2   0.80   1.11   1.28    1.02
 odd column                       1      1      1   0.60   0.60   0.60    0.60
 prime column                     1      0      0   2.80   2.80   2.80    2.80
 ALL                             16      3      3   0.60   1.11   1.31    1.05
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      0      0   1.04   1.12   1.18    1.11
 > 65536 points                   4      0      0   1.05   1.17   1.31    1.17
 1025..4096                       3      1      1   0.80   1.28   2.80    1.42
 <= 256 points                    1      1      1   0.60   0.60   0.60    0.60
 257..1024                        1      1      1   0.34   0.34   0.34    0.34
 ALL                             16      3      3   0.60   1.11   1.31    1.05
```


worst 10: 64x15 (chain 0.34), 15x16 (chain 0.60), 64x30 (chain 0.80), 16x1000 (chain 1.04), 256x1024 (chain 1.05), 512x512 (chain 1.07), 256x256 (chain 1.08), 128x512 (chain 1.10), 64x256 (chain 1.12), 32x1024 (chain 1.13)
best 5: 17x64 (chain 2.80), 2048x2048 (chain 1.31), 64x64 (chain 1.28), 1024x1024 (chain 1.27), 16x1024 (chain 1.18)
