# gauntlet report (2D)

run: `mt8_still4_2026-09-25`  contract: 2D c2c interleaved, natural, out of place, K=1_mt8  cells: 22 listed, 22 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.325..1.763

threaded plans that ran serial (engaged = 0 at both flips): 3: 16x16, 32x16, 32x32


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       16x16  2^4          csk+rb raced            158        145    0.92     64.8  2.5e-16
       32x16  2^5          turn   raced            502        340    0.67     45.9  2.3e-16
       32x32  2^5          chain+rb2 raced            739        720    0.97     69.3  2.4e-16
      128x64  2^7          chain+rb2 raced           4178       4934    1.12    127.5  2.6e-16
     128x128  2^7          chain  raced           9313      12838    1.22    123.1  3.8e-16
     128x256  2^7          chain+rb2 raced          11911      14837    1.01    206.3  4.0e-16
     128x512  2^7          chain  raced          26847      27290    0.97    195.3  3.2e-16
      256x32  2^8          chain+rb2 raced           4764       5590    1.13    111.8  3.0e-16
      256x64  2^8          chain+rb2 raced          10173      13961    1.29    112.7  3.8e-16
     256x128  2^8          chain  raced          17559      17982    1.02    140.0  3.8e-16
     256x256  2^8          chain+rb2 raced          26467      32635    1.18    198.1  4.1e-16
    256x8192  2^8          chain  raced        1303613    1654650    1.13    168.9  1.5e-15
      512x64  2^9          chain+rb2 raced          18333      19203    1.01    134.1  3.7e-16
     512x128  2^9          chain  raced          28593      32555    1.07    183.4  3.5e-16
     1024x64  2^10         chain+rb2 raced          35953      33408    0.91    145.8  3.5e-16
   2048x2048  2^11         chain  raced        5514712    5945400    1.07     83.7  3.1e-15
    4096x128  2^12         chain  raced         265200     280069    0.98    187.8  1.2e-15
    4096x256  2^12         chain+rb2 raced         705437     671462    0.91    148.6  1.3e-15
   4096x1024  2^12         chain  raced        6728825    7065369    1.03     68.6  3.1e-15
    8192x128  2^13         chain  raced         889587     694744    0.70    117.9  1.2e-15
    8192x256  2^13         chain+rb2 raced        3011750    2372150    0.74     73.1  1.6e-15
    8192x512  2^13         chain  raced        7731987    7606900    0.97     59.7  2.9e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain+rb2                       10      1      4   0.91   1.01   1.29    1.02
 chain                           10      1      4   0.97   1.02   1.22    1.01
 csk+rb                           1      0      1   0.92   0.92   0.92    0.92
 turn                             1      1      1   0.67   0.67   0.67    0.67
 ALL                             22      3     10   0.74   1.01   1.18    0.99
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     22      3     10   0.74   1.01   1.18    0.99
 ALL                             22      3     10   0.74   1.01   1.18    0.99
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     11      0      2   0.97   1.07   1.22    1.08
 > 65536 points                   8      2      5   0.70   0.98   1.13    0.93
 257..1024                        2      1      2   0.67   0.82   0.97    0.81
 <= 256 points                    1      0      1   0.92   0.92   0.92    0.92
 ALL                             22      3     10   0.74   1.01   1.18    0.99
```


worst 10: 32x16 (turn 0.67), 8192x128 (chain 0.70), 8192x256 (chain+rb2 0.74), 1024x64 (chain+rb2 0.91), 4096x256 (chain+rb2 0.91), 16x16 (csk+rb 0.92), 8192x512 (chain 0.97), 32x32 (chain+rb2 0.97), 128x512 (chain 0.97), 4096x128 (chain 0.98)
best 5: 256x64 (chain+rb2 1.29), 128x128 (chain 1.22), 256x256 (chain+rb2 1.18), 256x8192 (chain 1.13), 256x32 (chain+rb2 1.13)
