# gauntlet report (2D)

run: `ip2d_c2r_ip_fftw`  contract: 2D c2r interleaved, natural, out of place, K=1_c2r_ip_fftw  cells: 16 listed, 16 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 1.237..1.282


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            229        147    0.64     20.7  5.5e-16
     16x1000  2^4          chain  raced          13219      13691    0.89     42.3  1.0e-15
     16x1024  2^4          chain  raced          13361      11785    0.86     42.9  7.8e-16
       17x64  17           chain  raced            758       2016    2.63     36.2  9.0e-16
     32x1024  2^5          chain  raced          21948      24640    1.08     56.0  1.3e-15
       64x15  2^6          chain  raced           1908        653    0.34     12.5  7.1e-16
       64x30  2^6          chain  raced           2085       1671    0.79     25.1  9.1e-16
       64x64  2^6          chain  raced           2552       3238    1.24     48.1  7.8e-16
      64x256  2^6          chain  raced          11125      12531    1.10     51.5  8.9e-16
     128x128  2^7          chain  raced          11512      13450    1.16     49.8  8.9e-16
     128x512  2^7          chain  raced          48710      53814    1.07     53.8  1.3e-15
     256x256  2^8          chain  raced          65851      73064    1.08     39.8  1.0e-15
    256x1024  2^8          chain  raced         274993     305666    1.03     42.9  1.2e-15
     512x512  2^9          chain  raced         274440     353163    1.21     43.0  1.2e-15
   1024x1024  2^10         chain  raced        1221375    1551487    1.22     42.9  1.4e-15
   2048x2048  2^11         chain  raced        7177625    9786662    1.36     32.1  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      3      5   0.64   1.08   1.36    1.02
 ALL                             16      3      5   0.64   1.08   1.36    1.02
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      2      4   0.79   1.08   1.24    0.99
 odd column                       1      1      1   0.64   0.64   0.64    0.64
 prime column                     1      0      0   2.63   2.63   2.63    2.63
 ALL                             16      3      5   0.64   1.08   1.36    1.02
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      0      2   0.86   1.08   1.16    1.03
 > 65536 points                   4      0      0   1.03   1.21   1.36    1.20
 1025..4096                       3      1      1   0.79   1.24   2.63    1.37
 <= 256 points                    1      1      1   0.64   0.64   0.64    0.64
 257..1024                        1      1      1   0.34   0.34   0.34    0.34
 ALL                             16      3      5   0.64   1.08   1.36    1.02
```


worst 10: 64x15 (chain 0.34), 15x16 (chain 0.64), 64x30 (chain 0.79), 16x1024 (chain 0.86), 16x1000 (chain 0.89), 256x1024 (chain 1.03), 128x512 (chain 1.07), 32x1024 (chain 1.08), 256x256 (chain 1.08), 64x256 (chain 1.10)
best 5: 17x64 (chain 2.63), 2048x2048 (chain 1.36), 64x64 (chain 1.24), 1024x1024 (chain 1.22), 512x512 (chain 1.21)
