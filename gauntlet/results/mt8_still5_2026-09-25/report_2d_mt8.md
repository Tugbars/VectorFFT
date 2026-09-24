# gauntlet report (2D)

run: `mt8_still5_2026-09-25`  contract: 2D c2c interleaved, natural, out of place, K=1_mt8  cells: 9 listed, 9 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.415..1.676

threaded plans that ran serial (engaged = 0 at both flips): 3: 16x16, 32x16, 32x32


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       16x16  2^4          csk+rb2 raced            156        145    0.91     65.5  2.5e-16
       32x16  2^5          chain+rb2 raced            347        339    0.96     66.4  2.3e-16
       32x32  2^5          chain+rb2 raced            789        718    0.87     64.9  2.4e-16
     128x512  2^7          chain  raced          26940      26830    0.85    194.6  3.2e-16
     1024x64  2^10         chain+rb2 raced          31527      33701    0.92    166.3  3.5e-16
    4096x256  2^12         chain  raced         680950     614875    0.88    154.0  1.3e-15
    8192x128  2^13         chain  raced         872012     689974    0.72    120.2  1.2e-15
    8192x256  2^13         chain+rb2 raced        2690450    2495275    0.89     81.8  1.2e-15
    8192x512  2^13         chain  raced        6919525    7754262    1.07     66.7  3.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain+rb2                        4      0      4   0.87   0.90   0.96    0.91
 chain                            4      1      3   0.72   0.86   1.07    0.87
 csk+rb2                          1      0      1   0.91   0.91   0.91    0.91
 ALL                              9      1      8   0.72   0.89   1.07    0.89
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      9      1      8   0.72   0.89   1.07    0.89
 ALL                              9      1      8   0.72   0.89   1.07    0.89
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   4      1      3   0.72   0.88   1.07    0.88
 257..1024                        2      0      2   0.87   0.92   0.96    0.92
 4097..65536                      2      0      2   0.85   0.88   0.92    0.88
 <= 256 points                    1      0      1   0.91   0.91   0.91    0.91
 ALL                              9      1      8   0.72   0.89   1.07    0.89
```


worst 10: 8192x128 (chain 0.72), 128x512 (chain 0.85), 32x32 (chain+rb2 0.87), 4096x256 (chain 0.88), 8192x256 (chain+rb2 0.89), 16x16 (csk+rb2 0.91), 1024x64 (chain+rb2 0.92), 32x16 (chain+rb2 0.96), 8192x512 (chain 1.07)
best 5: 8192x512 (chain 1.07), 32x16 (chain+rb2 0.96), 1024x64 (chain+rb2 0.92), 16x16 (csk+rb2 0.91), 8192x256 (chain+rb2 0.89)
