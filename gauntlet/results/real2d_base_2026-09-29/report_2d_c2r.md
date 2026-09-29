# gauntlet report (2D)

run: `real2d_base_2026-09-29`  contract: 2D c2c interleaved, natural, out of place, K=1_c2r  cells: 67 listed, 67 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.209..1.297


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       16x16  2^4          chain  replayed         334        138    0.41     15.3  3.3e-16
       16x32  2^4          chain  replayed         429        501    1.16     26.8  4.5e-16
       16x64  2^4          chain  replayed         653        797    1.21     39.2  5.6e-16
      16x128  2^4          chain  replayed        1252       1431    1.11     45.0  6.1e-16
      16x256  2^4          chain  replayed        3264       3610    0.94     37.6  7.8e-16
      16x512  2^4          chain  replayed        6616       9250    1.38     40.2  7.7e-16
     16x1024  2^4          chain  replayed       15716      18488    1.17     36.5  8.9e-16
     16x4096  2^4          chain  replayed       61970      83200    1.11     42.3  1.0e-15
       23x64  23           chain  replayed        1309       3240    2.47     29.6  1.3e-15
       29x64  29           chain  replayed        2043       3436    1.64     24.7  1.0e-15
       32x16  2^5          chain  replayed         394        533    1.35     29.2  5.6e-16
       32x32  2^5          chain  replayed         893        664    0.73     28.7  6.7e-16
       32x64  2^5          chain  replayed        1386       1667    1.18     40.6  6.7e-16
      32x128  2^5          chain  replayed        3047       3657    1.19     40.3  7.8e-16
      32x256  2^5          chain  replayed        6695       9429    1.37     39.8  8.9e-16
      32x512  2^5          chain  replayed       12033      19271    1.59     47.7  8.9e-16
     32x1024  2^5          chain  replayed       28779      38693    1.33     42.7  1.0e-15
       47x64  47           chain  replayed        3477       7782    2.21     25.0  2.6e-15
       48x48  2^4.3        chain  replayed        2519       2697    1.07     25.5  1.2e-15
       63x64  3^2.7        chain  replayed        3777       3992    1.05     32.0  9.5e-16
       64x16  2^6          chain  replayed         866       1028    1.16     29.6  7.8e-16
       64x30  2^6          chain  replayed        2188       1963    0.89     23.9  8.4e-16
       64x32  2^6          chain  replayed        1944       2009    0.98     29.0  5.6e-16
       64x50  2^6          chain  replayed        3035       3759    1.19     30.7  8.0e-16
       64x62  2^6          chain  replayed        6291      10366    1.42     18.9  1.1e-15
       64x64  2^6          chain  replayed        3212       4092    1.20     38.3  7.8e-16
      64x128  2^6          chain  replayed        6347       8651    1.33     41.9  9.4e-16
      64x256  2^6          chain  replayed       14227      17686    1.23     40.3  8.9e-16
      64x512  2^6          chain  replayed       26174      39223    1.45     46.9  1.0e-15
     64x1024  2^6          chain  replayed       62585      80170    1.14     41.9  1.0e-15
     64x4096  2^6          chain  replayed      283567     369546    1.22     41.6  1.1e-15
       96x96  2^5.3        chain  replayed        8212       9319    1.11     37.0  1.0e-15
      100x64  2^2.5^2      chain  replayed        6591       5944    0.89     30.7  8.9e-16
      128x16  2^7          chain  replayed        3126       2289    0.73     18.0  6.1e-16
      128x32  2^7          chain  replayed        5426       4641    0.85     22.6  6.7e-16
      128x64  2^7          chain  replayed        7980      10131    1.24     33.4  8.3e-16
     128x128  2^7          chain  replayed       15176      18776    1.22     37.8  1.0e-15
     128x256  2^7          chain  replayed       35611      38398    1.07     34.5  9.4e-16
     128x512  2^7          chain  replayed       66756      77170    1.15     39.3  1.0e-15
    128x1024  2^7          chain  replayed      159690     161135    1.00     34.9  1.3e-15
     192x192  2^6.3        chain  replayed       36661      47370    1.28     38.1  1.3e-15
      256x16  2^8          chain  replayed        6700       5056    0.75     18.3  6.7e-16
      256x32  2^8          chain  replayed       10011      11747    1.17     26.6  7.8e-16
      256x64  2^8          chain  replayed       17052      20904    1.15     33.6  8.3e-16
     256x128  2^8          chain  replayed       32598      39451    1.21     37.7  1.0e-15
     256x256  2^8          chain  replayed       71602      80765    1.11     36.6  1.2e-15
     256x512  2^8          chain  replayed      145840     163768    1.11     38.2  1.1e-15
    256x1024  2^8          chain  replayed      321660     384893    1.11     36.7  1.1e-15
     480x480  2^5.3.5      chain  replayed      280753     403794    1.35     36.5  1.4e-15
      512x16  2^9          chain  replayed       14047      12793    0.91     19.0  7.8e-16
      512x32  2^9          chain  replayed       20680      25236    1.20     27.7  8.9e-16
      512x64  2^9          chain  replayed       36966      45725    1.23     33.2  8.9e-16
     512x128  2^9          chain  replayed       70025      87855    1.24     37.4  1.1e-15
     512x256  2^9          chain  replayed      170567     183450    1.04     32.7  1.1e-15
     512x512  2^9          chain  replayed      323053     429573    1.32     36.5  1.1e-15
    512x1024  2^9          chain  replayed      657438     865481    1.26     37.9  1.3e-15
   1000x1000  2^3.5^3      chain  replayed     1836138    1742393    0.94     27.1  1.7e-15
     1024x16  2^10         chain  replayed       28037      26398    0.93     20.5  7.8e-16
     1024x32  2^10         chain  replayed       42275      52228    1.23     29.1  1.1e-15
     1024x64  2^10         chain  replayed       73775      95638    1.26     35.5  1.1e-15
    1024x128  2^10         chain  replayed      159167     190180    1.19     35.0  1.3e-15
    1024x256  2^10         chain  replayed      340153     423140    1.24     34.7  1.1e-15
    1024x512  2^10         chain  replayed      663962     932062    1.39     37.5  1.3e-15
   1024x1024  2^10         chain  replayed     1489187    1887643    1.25     35.2  1.4e-15
   2048x2048  2^11         chain  replayed     9211225   14606450    1.58     25.0  1.3e-15
     4096x16  2^12         chain  replayed      144903     108119    0.73     18.1  1.0e-15
     4096x64  2^12         chain  replayed      414773     446897    1.06     28.4  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           67      5     13   0.89   1.19   1.42    1.15
 ALL                             67      5     13   0.89   1.19   1.42    1.15
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     57      5     11   0.85   1.19   1.38    1.12
 even column                      6      0      2   0.89   1.09   1.35    1.09
 prime column                     3      0      0   1.64   2.21   2.47    2.07
 odd column                       1      0      0   1.05   1.05   1.05    1.05
 ALL                             67      5     13   0.89   1.19   1.42    1.15
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     29      1      4   0.91   1.21   1.38    1.17
 1025..4096                      17      2      6   0.75   1.11   2.21    1.16
 > 65536 points                  15      0      1   1.00   1.22   1.39    1.19
 257..1024                        5      1      1   0.73   1.16   1.35    1.10
 <= 256 points                    1      1      1   0.41   0.41   0.41    0.41
 ALL                             67      5     13   0.89   1.19   1.42    1.15
```


worst 10: 16x16 (chain 0.41), 128x16 (chain 0.73), 4096x16 (chain 0.73), 32x32 (chain 0.73), 256x16 (chain 0.75), 128x32 (chain 0.85), 100x64 (chain 0.89), 64x30 (chain 0.89), 512x16 (chain 0.91), 1024x16 (chain 0.93)
best 5: 23x64 (chain 2.47), 47x64 (chain 2.21), 29x64 (chain 1.64), 32x512 (chain 1.59), 2048x2048 (chain 1.58)
