# gauntlet report (2D)

run: `ct_trial_2d_avx2_2026-11-24`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 4 listed, 4 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.438..1.508


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       75x21  3.5^2        chain+rb raced           2851       2565    0.89     29.3  4.4e-16
     75x1029  3.5^2        chain  raced         187700     179240    0.94     33.4  6.9e-16
      135x35  3^3.5        chain  raced           8753       7929    0.90     32.9  5.5e-16
    150x1050  2.3.5^2      chain  raced         376500     355808    0.92     36.1  8.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            3      0      3   0.90   0.92   0.94    0.92
 chain+rb                         1      0      1   0.89   0.89   0.89    0.89
 ALL                              4      0      4   0.89   0.91   0.94    0.91
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 odd column                       3      0      3   0.89   0.90   0.94    0.91
 even column                      1      0      1   0.92   0.92   0.92    0.92
 ALL                              4      0      4   0.89   0.91   0.94    0.91
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   2      0      2   0.92   0.93   0.94    0.93
 1025..4096                       1      0      1   0.89   0.89   0.89    0.89
 4097..65536                      1      0      1   0.90   0.90   0.90    0.90
 ALL                              4      0      4   0.89   0.91   0.94    0.91
```


worst 10: 75x21 (chain+rb 0.89), 135x35 (chain 0.90), 150x1050 (chain 0.92), 75x1029 (chain 0.94)
best 5: 75x1029 (chain 0.94), 150x1050 (chain 0.92), 135x35 (chain 0.90), 75x21 (chain+rb 0.89)
