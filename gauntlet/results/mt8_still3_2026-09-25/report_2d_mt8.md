# gauntlet report (2D)

run: `mt8_still3_2026-09-25`  contract: 2D c2c interleaved, natural, out of place, K=1_mt8  cells: 22 listed, 22 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.066..1.420

threaded plans that ran serial (engaged = 0 at both flips): 3: 16x16, 32x16, 32x32


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       16x16  2^4          csk+rb2 raced            158        145    0.92     64.8  2.5e-16
       32x16  2^5          chain+rb2 raced            355        340    0.95     64.8  2.5e-16
       32x32  2^5          chain+rb2 raced            724        717    0.98     70.8  2.4e-16
      128x64  2^7          turn   raced           5762       4992    0.85     92.4  4.3e-16
     128x128  2^7          chain  raced          15176      12282    0.64     75.6  3.8e-16
     128x256  2^7          chain+rb2 raced          29741      14952    0.46     82.6  4.0e-16
     128x512  2^7          chain  raced          59410      26801    0.40     88.2  3.2e-16
      256x32  2^8          turn   raced           5914       5829    0.97     90.0  3.7e-16
      256x64  2^8          turn   raced          13545      14405    0.95     84.7  4.8e-16
     256x128  2^8          chain  raced          42826      18185    0.38     57.4  3.8e-16
     256x256  2^8          chain+rb2 raced          55240      33180    0.50     94.9  4.1e-16
    256x8192  2^8          chain  raced        1399737    1970881    0.81    157.3  1.5e-15
      512x64  2^9          turn   raced          21882      19107    0.79    112.3  4.9e-16
     512x128  2^9          chain  raced          57063      33277    0.49     91.9  3.5e-16
     1024x64  2^10         turn   raced          34390      33533    0.96    152.5  4.6e-16
   2048x2048  2^11         chain  raced        7144225    6146806    0.83     64.6  3.1e-15
    4096x128  2^12         chain  raced         302300     282743    0.79    164.8  1.2e-15
    4096x256  2^12         chain+rb2 raced         709275     659475    0.89    147.8  1.3e-15
   4096x1024  2^12         chain  raced        6426525   10057519    1.10     71.8  3.1e-15
    8192x128  2^13         chain  raced         921362     748294    0.75    113.8  1.2e-15
    8192x256  2^13         chain  raced        3154937    2375937    0.71     69.8  1.6e-15
    8192x512  2^13         chain  raced        6641300    7619068    1.10     69.5  3.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           11      7      9   0.40   0.75   1.10    0.69
 chain+rb2                        5      2      5   0.46   0.89   0.98    0.72
 turn                             5      1      5   0.79   0.95   0.97    0.90
 csk+rb2                          1      0      1   0.92   0.92   0.92    0.92
 ALL                             22     10     20   0.46   0.82   0.98    0.75
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     22     10     20   0.46   0.82   0.98    0.75
 ALL                             22     10     20   0.46   0.82   0.98    0.75
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     11      7     11   0.40   0.64   0.96    0.63
 > 65536 points                   8      3      6   0.71   0.82   1.10    0.86
 257..1024                        2      0      2   0.95   0.96   0.98    0.96
 <= 256 points                    1      0      1   0.92   0.92   0.92    0.92
 ALL                             22     10     20   0.46   0.82   0.98    0.75
```


worst 10: 256x128 (chain 0.38), 128x512 (chain 0.40), 128x256 (chain+rb2 0.46), 512x128 (chain 0.49), 256x256 (chain+rb2 0.50), 128x128 (chain 0.64), 8192x256 (chain 0.71), 8192x128 (chain 0.75), 512x64 (turn 0.79), 4096x128 (chain 0.79)
best 5: 8192x512 (chain 1.10), 4096x1024 (chain 1.10), 32x32 (chain+rb2 0.98), 256x32 (turn 0.97), 1024x64 (turn 0.96)
