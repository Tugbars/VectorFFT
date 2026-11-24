# gauntlet report (2D)

run: `ct_default_2d_avx2_2026-11-24`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 13 listed, 13 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.427..1.509


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     50x1050  2.5^2        chain  raced         104768     103966    0.99     39.3  5.6e-16
     54x1050  2.3^3        chain  raced         114560     115205    1.00     39.1  7.3e-16
     63x1050  3^2.7        chain  raced         138463     140863    1.00     38.3  5.6e-16
       75x21  3.5^2        chain+rb raced           2799       2573    0.86     29.9  4.9e-16
     75x1029  3.5^2        chain  raced         186292     181552    0.86     33.6  6.6e-16
      100x42  2^2.5^2      chain  raced           6722       6209    0.91     37.6  4.8e-16
    100x1050  2^2.5^2      chain  raced         261326     224250    0.69     33.5  6.3e-16
      105x42  3.5.7        chain  raced           7289       6910    0.94     36.6  7.1e-16
    105x1050  3.5.7        chain  raced         260111     244653    0.93     35.5  6.3e-16
    108x1050  2^2.3^3      chain  raced         260824     248420    0.93     36.5  6.5e-16
      135x21  3^3.5        chain+rb raced           4971       4787    0.96     32.7  7.2e-16
      135x35  3^3.5        chain  raced           8560       7946    0.93     33.7  5.1e-16
    150x1050  2.3.5^2      chain  raced         387567     357396    0.92     35.1  8.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           11      1      9   0.86   0.93   1.00    0.91
 chain+rb                         2      0      2   0.86   0.91   0.96    0.91
 ALL                             13      1     11   0.86   0.93   1.00    0.91
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 odd column                       7      0      6   0.86   0.93   1.00    0.92
 even column                      6      1      5   0.69   0.92   1.00    0.90
 ALL                             13      1     11   0.86   0.93   1.00    0.91
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      1      5   0.69   0.92   1.00    0.88
 4097..65536                      5      0      4   0.91   0.94   1.00    0.95
 1025..4096                       2      0      2   0.86   0.91   0.96    0.91
 ALL                             13      1     11   0.86   0.93   1.00    0.91
```


worst 10: 100x1050 (chain 0.69), 75x21 (chain+rb 0.86), 75x1029 (chain 0.86), 100x42 (chain 0.91), 150x1050 (chain 0.92), 135x35 (chain 0.93), 105x1050 (chain 0.93), 108x1050 (chain 0.93), 105x42 (chain 0.94), 135x21 (chain+rb 0.96)
best 5: 54x1050 (chain 1.00), 63x1050 (chain 1.00), 50x1050 (chain 0.99), 135x21 (chain+rb 0.96), 105x42 (chain 0.94)
