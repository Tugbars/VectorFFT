# gauntlet report (2D)

run: `mt8_still2_2026-09-25`  contract: 2D c2c interleaved, natural, out of place, K=1_mt8  cells: 22 listed, 22 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.123..1.147

threaded plans that ran serial (engaged = 0 at both flips): 2: 16x16, 32x16


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       16x16  2^4          chain+rb2 raced            163        147    0.89     62.8  2.5e-16
       32x16  2^5          turn   raced            498        338    0.68     46.3  2.3e-16
       32x32  2^5          chain+rb2 raced           1297        721    0.50     39.5  2.4e-16
      128x64  2^7          turn   raced           5868       5016    0.84     90.7  4.3e-16
     128x128  2^7          chain  raced          19062      12806    0.66     60.2  3.8e-16
     128x256  2^7          chain  raced          34216      14782    0.42     71.8  4.0e-16
     128x512  2^7          chain  raced          49807      27590    0.52    105.3  3.2e-16
      256x32  2^8          turn   raced           6212       5585    0.85     85.7  3.7e-16
      256x64  2^8          turn   raced          13642      14267    0.97     84.1  4.8e-16
     256x128  2^8          chain  raced          43590      18042    0.36     56.4  3.8e-16
     256x256  2^8          chain+rb2 raced          50300      32446    0.63    104.2  4.1e-16
    256x8192  2^8          chain  raced        1260138    1466456    1.12    174.7  1.5e-15
      512x64  2^9          turn   raced          23905      19512    0.79    102.8  4.9e-16
     512x128  2^9          chain  raced          54893      32928    0.49     95.5  3.5e-16
     1024x64  2^10         turn   raced          34353      34541    0.92    152.6  4.6e-16
   2048x2048  2^11         chain  raced        5543037    5719069    1.00     83.2  3.1e-15
    4096x128  2^12         chain  raced         290587     284862    0.93    171.4  1.2e-15
    4096x256  2^12         chain  raced         667813     630400    0.92    157.0  1.3e-15
   4096x1024  2^12         chain  raced        6168212    6448469    0.99     74.8  3.1e-15
    8192x128  2^13         chain  raced         971763     752793    0.77    107.9  1.2e-15
    8192x256  2^13         chain+rb2 raced        2999000    2281100    0.70     73.4  1.6e-15
    8192x512  2^13         chain  raced        7815188    7367469    0.93     59.0  2.9e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           12      6     11   0.42   0.84   1.00    0.71
 turn                             6      2      6   0.68   0.85   0.97    0.84
 chain+rb2                        4      3      4   0.50   0.67   0.89    0.67
 ALL                             22     11     21   0.49   0.82   0.99    0.74
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     22     11     21   0.49   0.82   0.99    0.74
 ALL                             22     11     21   0.49   0.82   0.99    0.74
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     11      7     11   0.42   0.66   0.92    0.65
 > 65536 points                   8      2      7   0.70   0.93   1.12    0.91
 257..1024                        2      2      2   0.50   0.59   0.68    0.58
 <= 256 points                    1      0      1   0.89   0.89   0.89    0.89
 ALL                             22     11     21   0.49   0.82   0.99    0.74
```


worst 10: 256x128 (chain 0.36), 128x256 (chain 0.42), 512x128 (chain 0.49), 32x32 (chain+rb2 0.50), 128x512 (chain 0.52), 256x256 (chain+rb2 0.63), 128x128 (chain 0.66), 32x16 (turn 0.68), 8192x256 (chain+rb2 0.70), 8192x128 (chain 0.77)
best 5: 256x8192 (chain 1.12), 2048x2048 (chain 1.00), 4096x1024 (chain 0.99), 256x64 (turn 0.97), 4096x128 (chain 0.93)
