# gauntlet report (2D)

run: `mt8_still_2026-09-24`  contract: 2D c2c interleaved, natural, out of place, K=1_mt8  cells: 22 listed, 22 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 0.600..1.395

threaded plans that ran serial (engaged = 0 at both flips): 3: 16x16, 32x16, 32x32


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       16x16  2^4          chain+rb2 raced            159        145    0.87     64.6  2.5e-16
       32x16  2^5          chain+rb2 raced            358        340    0.68     64.3  2.3e-16
       32x32  2^5          chain+rb2 raced            726        719    0.87     70.5  2.4e-16
      128x64  2^7          turn   raced           5926       5148    0.79     89.9  4.3e-16
     128x128  2^7          chain  raced          17847      12579    0.65     64.3  3.8e-16
     128x256  2^7          chain+rb2 raced          25433      14883    0.54     96.6  4.0e-16
     128x512  2^7          chain  raced          39603      28348    0.66    132.4  3.2e-16
      256x32  2^8          turn   raced           5978       5611    0.92     89.1  3.7e-16
      256x64  2^8          turn   raced          13314      14450    0.98     86.1  4.8e-16
     256x128  2^8          chain  raced          26911      18689    0.66     91.3  3.8e-16
     256x256  2^8          chain+rb2 raced          35260      31102    0.87    148.7  4.1e-16
    256x8192  2^8          chain  raced        1281050    1429006    1.05    171.9  1.5e-15
      512x64  2^9          turn   raced          20838      18940    0.85    117.9  4.9e-16
     512x128  2^9          chain  raced          36587      32920    0.86    143.3  3.5e-16
     1024x64  2^10         turn   raced          32983      33196    0.92    159.0  4.6e-16
   2048x2048  2^11         chain  raced        5571338    5865437    1.01     82.8  3.1e-15
    4096x128  2^12         chain  raced         278600     286950    1.00    178.8  1.2e-15
    4096x256  2^12         chain+rb2 raced         650650     630462    0.96    161.2  1.3e-15
   4096x1024  2^12         chain  raced        6167850    6847831    1.06     74.8  3.1e-15
    8192x128  2^13         chain  raced         890237     713143    0.65    117.8  1.1e-15
    8192x256  2^13         chain+rb2 raced        2986613    2731687    0.79     73.7  1.2e-15
    8192x512  2^13         chain  raced        8035850    7933437    0.96     57.4  2.9e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           10      4      6   0.65   0.91   1.06    0.84
 chain+rb2                        7      3      7   0.54   0.87   0.96    0.78
 turn                             5      1      5   0.79   0.92   0.98    0.89
 ALL                             22      8     18   0.65   0.87   1.01    0.83
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     22      8     18   0.65   0.87   1.01    0.83
 ALL                             22      8     18   0.65   0.87   1.01    0.83
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     11      5     11   0.65   0.85   0.92    0.78
 > 65536 points                   8      2      4   0.65   0.98   1.06    0.92
 257..1024                        2      1      2   0.68   0.77   0.87    0.77
 <= 256 points                    1      0      1   0.87   0.87   0.87    0.87
 ALL                             22      8     18   0.65   0.87   1.01    0.83
```


worst 10: 128x256 (chain+rb2 0.54), 128x128 (chain 0.65), 8192x128 (chain 0.65), 256x128 (chain 0.66), 128x512 (chain 0.66), 32x16 (chain+rb2 0.68), 8192x256 (chain+rb2 0.79), 128x64 (turn 0.79), 512x64 (turn 0.85), 512x128 (chain 0.86)
best 5: 4096x1024 (chain 1.06), 256x8192 (chain 1.05), 2048x2048 (chain 1.01), 4096x128 (chain 1.00), 256x64 (turn 0.98)
