# gauntlet report (2D)

run: `2d_odds_subset_avx2_2026-11-23`  contract: 2D c2c interleaved, natural, out of place, K=1_mt8  cells: 17 listed, 17 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.233..1.611

threaded plans that ran serial (engaged = 0 at both flips): 0


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
      21x135  3.7          chain  raced           2472       3751    1.49     65.8  7.3e-16
      21x315  3.7          chain  raced           4862       7059    1.33     86.3  7.4e-16
      21x525  3.7          chain  raced           6289       9941    1.45    117.7  6.5e-16
     21x2187  3.7          chain  raced          22198      32019    1.05    160.2  7.7e-16
     21x4375  3.7          chain  raced          38014      63259    1.27    199.2  5.9e-16
     35x1029  5.7          chain  raced          22191      28742    1.12    122.8  5.0e-16
       75x21  3.5^2        chain+rb raced           2923       3696    1.12     28.6  4.4e-16
       75x35  3.5^2        chain  raced           3175       3897    1.18     47.0  4.2e-16
     75x1029  3.5^2        chain  raced          28776      50308    1.27    217.7  5.6e-16
    135x1029  3^3.5        chain  raced          49307      90968    1.39    240.7  6.1e-16
    135x4375  3^3.5        chain  raced         258737     754200    1.20    218.8  6.8e-16
     525x525  3.5^2.7      chain  raced          98387     180550    1.16    253.1  7.1e-16
   1029x1029  3.7^3        chain  raced         489062    1011150    1.32    216.7  6.8e-16
    2187x525  3^7          chain  raced         632087     864187    0.79    182.8  8.8e-16
   2187x2187  3^7          chain  raced        6370600    6493887    0.96     83.3  9.7e-16
     4375x21  5^4.7        chain  replayed       52252     121552    1.78    144.9  6.1e-16
    4375x315  5^4.7        chain  raced         911462     939262    0.79    154.2  8.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           16      2      3   0.79   1.23   1.49    1.20
 chain+rb                         1      0      0   1.12   1.12   1.12    1.12
 ALL                             17      2      3   0.79   1.20   1.49    1.19
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 odd column                      17      2      3   0.79   1.20   1.49    1.19
 ALL                             17      2      3   0.79   1.20   1.49    1.19
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  10      2      3   0.79   1.23   1.78    1.16
 4097..65536                      4      0      0   1.05   1.22   1.45    1.23
 1025..4096                       3      0      0   1.12   1.18   1.49    1.25
 ALL                             17      2      3   0.79   1.20   1.49    1.19
```


worst 10: 2187x525 (chain 0.79), 4375x315 (chain 0.79), 2187x2187 (chain 0.96), 21x2187 (chain 1.05), 35x1029 (chain 1.12), 75x21 (chain+rb 1.12), 525x525 (chain 1.16), 75x35 (chain 1.18), 135x4375 (chain 1.20), 75x1029 (chain 1.27)
best 5: 4375x21 (chain 1.78), 21x135 (chain 1.49), 21x525 (chain 1.45), 135x1029 (chain 1.39), 21x315 (chain 1.33)
