# gauntlet report (2D)

run: `ip2d_r2c_ip_fftw`  contract: 2D r2c interleaved, natural, out of place, K=1_r2c_ip_fftw  cells: 16 listed, 16 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 1.376..1.434


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            243        150    0.62     19.5  3.1e-16
     16x1000  2^4          chain  raced          11503      14240    1.23     48.6  4.7e-16
     16x1024  2^4          chain  raced          11605      11561    0.95     49.4  2.8e-16
       17x64  17           chain  raced            722       2075    2.80     38.0  4.8e-16
     32x1024  2^5          chain  raced          21077      24425    1.16     58.3  4.5e-16
       64x15  2^6          chain  raced           1111        659    0.59     21.4  3.7e-16
       64x30  2^6          chain  raced           2108       1710    0.81     24.8  4.4e-16
       64x64  2^6          chain  raced           2493       3570    1.40     49.3  3.0e-16
      64x256  2^6          chain  raced          10954      12474    1.13     52.4  3.9e-16
     128x128  2^7          chain  raced          11545      12905    1.12     49.7  3.6e-16
     128x512  2^7          chain  raced          46364      53174    1.13     56.5  4.5e-16
     256x256  2^8          chain  raced          60713      73366    1.20     43.2  4.1e-16
    256x1024  2^8          chain  raced         288473     305086    1.05     40.9  4.9e-16
     512x512  2^9          chain  raced         316900     335443    1.04     37.2  5.1e-16
   1024x1024  2^10         chain  raced        1401975    1517569    1.01     37.4  5.6e-16
   2048x2048  2^11         chain  raced        8459225   10200181    1.20     27.3  5.6e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      2      4   0.62   1.12   1.40    1.08
 ALL                             16      2      4   0.62   1.12   1.40    1.08
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      1      3   0.81   1.12   1.23    1.05
 odd column                       1      1      1   0.62   0.62   0.62    0.62
 prime column                     1      0      0   2.80   2.80   2.80    2.80
 ALL                             16      2      4   0.62   1.12   1.40    1.08
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      0      1   0.95   1.13   1.23    1.12
 > 65536 points                   4      0      0   1.01   1.05   1.20    1.07
 1025..4096                       3      0      1   0.81   1.40   2.80    1.47
 <= 256 points                    1      1      1   0.62   0.62   0.62    0.62
 257..1024                        1      1      1   0.59   0.59   0.59    0.59
 ALL                             16      2      4   0.62   1.12   1.40    1.08
```


worst 10: 64x15 (chain 0.59), 15x16 (chain 0.62), 64x30 (chain 0.81), 16x1024 (chain 0.95), 1024x1024 (chain 1.01), 512x512 (chain 1.04), 256x1024 (chain 1.05), 128x128 (chain 1.12), 64x256 (chain 1.13), 128x512 (chain 1.13)
best 5: 17x64 (chain 2.80), 64x64 (chain 1.40), 16x1000 (chain 1.23), 2048x2048 (chain 1.20), 256x256 (chain 1.20)
