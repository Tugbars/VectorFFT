# gauntlet report (2D)

run: `ip2d_r2c_ipo_fftw`  contract: 2D r2c interleaved, natural, out of place, K=1_r2c_ipo_fftw  cells: 16 listed, 16 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 1.399..1.453


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x16  3.5          chain  raced            213        151    0.71     22.3  3.1e-16
     16x1000  2^4          chain  raced          11916      14120    1.16     46.9  4.7e-16
     16x1024  2^4          chain  raced           9489      11539    1.21     60.4  2.8e-16
       17x64  17           chain  raced            731       2117    2.76     37.5  5.6e-16
     32x1024  2^5          chain  raced          21249      24123    1.08     57.8  3.8e-16
       64x15  2^6          chain  raced           1111        634    0.57     21.4  3.7e-16
       64x30  2^6          chain  raced           2110       1720    0.81     24.8  4.0e-16
       64x64  2^6          chain  raced           2504       3718    1.47     49.1  3.0e-16
      64x256  2^6          chain  raced          11096      12577    1.12     51.7  3.9e-16
     128x128  2^7          chain  raced          11773      12942    1.09     48.7  3.6e-16
     128x512  2^7          chain  raced          47292      54834    1.16     55.4  4.1e-16
     256x256  2^8          chain  raced          60890      73498    1.20     43.1  4.1e-16
    256x1024  2^8          chain  raced         300427     304990    1.01     39.3  4.3e-16
     512x512  2^9          chain  replayed      297260     326786    1.10     39.7  5.1e-16
   1024x1024  2^10         chain  raced        1172225    1485774    1.26     44.7  5.2e-16
   2048x2048  2^11         chain  raced        8928937    9885212    1.09     25.8  5.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      2      3   0.71   1.11   1.47    1.11
 ALL                             16      2      3   0.71   1.11   1.47    1.11
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     14      1      2   0.81   1.11   1.26    1.07
 odd column                       1      1      1   0.71   0.71   0.71    0.71
 prime column                     1      0      0   2.76   2.76   2.76    2.76
 ALL                             16      2      3   0.71   1.11   1.47    1.11
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      7      0      0   1.08   1.16   1.21    1.14
 > 65536 points                   4      0      0   1.01   1.10   1.26    1.11
 1025..4096                       3      0      1   0.81   1.47   2.76    1.48
 <= 256 points                    1      1      1   0.71   0.71   0.71    0.71
 257..1024                        1      1      1   0.57   0.57   0.57    0.57
 ALL                             16      2      3   0.71   1.11   1.47    1.11
```


worst 10: 64x15 (chain 0.57), 15x16 (chain 0.71), 64x30 (chain 0.81), 256x1024 (chain 1.01), 32x1024 (chain 1.08), 128x128 (chain 1.09), 2048x2048 (chain 1.09), 512x512 (chain 1.10), 64x256 (chain 1.12), 128x512 (chain 1.16)
best 5: 17x64 (chain 2.76), 64x64 (chain 1.47), 1024x1024 (chain 1.26), 16x1024 (chain 1.21), 256x256 (chain 1.20)
