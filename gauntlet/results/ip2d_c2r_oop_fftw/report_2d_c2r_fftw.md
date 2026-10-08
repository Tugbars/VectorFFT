# gauntlet report (2D)

run: `ip2d_c2r_oop_fftw`  contract: 2D c2r interleaved, natural, out of place, K=1_c2r_fftw  cells: 16 listed, 16 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 1.201..1.293


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            142        146    1.03     33.3  6.8e-16
     16x1000  2^4          chain  raced          15457      12751    0.75     36.1  1.0e-15
     16x1024  2^4          chain  raced          12305      10500    0.83     46.6  8.9e-16
       17x64  17           chain  raced            711       2110    2.94     38.6  9.0e-16
     32x1024  2^5          chain  raced          21678      22266    1.02     56.7  8.9e-16
       64x15  2^6          chain  raced           1902        655    0.34     12.5  5.3e-16
       64x30  2^6          chain  raced           2133       1712    0.79     24.5  9.1e-16
       64x64  2^6          chain  raced           2608       3332    1.27     47.1  8.9e-16
      64x256  2^6          chain  raced          10905      11545    1.04     52.6  8.9e-16
     128x128  2^7          chain  raced          11703      12403    1.05     49.0  8.3e-16
     128x512  2^7          chain  raced          48634      49541    1.01     53.9  1.2e-15
     256x256  2^8          chain  raced          58400      69977    1.09     44.9  1.1e-15
    256x1024  2^8          chain  raced         277280     312406    1.12     42.5  1.2e-15
     512x512  2^9          chain  raced         302960     350720    1.13     38.9  1.1e-15
   1024x1024  2^10         chain  raced        1175988    1497750    1.15     44.6  1.4e-15
   2048x2048  2^11         chain  raced        7008062   11412344    1.56     32.9  1.7e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      3      4   0.75   1.05   1.56    1.04
 ALL                             16      3      4   0.75   1.05   1.56    1.04
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      3      4   0.75   1.05   1.27    0.96
 odd column                       1      0      0   1.03   1.03   1.03    1.03
 prime column                     1      0      0   2.94   2.94   2.94    2.94
 ALL                             16      3      4   0.75   1.05   1.56    1.04
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      1      2   0.75   1.02   1.09    0.96
 > 65536 points                   4      0      0   1.12   1.14   1.56    1.23
 1025..4096                       3      1      1   0.79   1.27   2.94    1.43
 <= 256 points                    1      0      0   1.03   1.03   1.03    1.03
 257..1024                        1      1      1   0.34   0.34   0.34    0.34
 ALL                             16      3      4   0.75   1.05   1.56    1.04
```


worst 10: 64x15 (chain 0.34), 16x1000 (chain 0.75), 64x30 (chain 0.79), 16x1024 (chain 0.83), 128x512 (chain 1.01), 32x1024 (chain 1.02), 15x16 (chain 1.03), 64x256 (chain 1.04), 128x128 (chain 1.05), 256x256 (chain 1.09)
best 5: 17x64 (chain 2.94), 2048x2048 (chain 1.56), 64x64 (chain 1.27), 1024x1024 (chain 1.15), 512x512 (chain 1.13)
