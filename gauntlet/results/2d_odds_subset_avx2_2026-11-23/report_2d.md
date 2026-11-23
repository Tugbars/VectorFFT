# gauntlet report (2D)

run: `2d_odds_subset_avx2_2026-11-23`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 17 listed, 17 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.410..1.492


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
      21x135  3.7          chain  raced           4207       6201    1.42     38.6  7.3e-16
      21x315  3.7          csk    raced          10808      15069    1.39     38.8  7.6e-16
      21x525  3.7          chain  raced          19891      26335    1.31     37.2  6.5e-16
     21x2187  3.7          chain  raced         108930     138001    1.26     32.6  8.2e-16
     21x4375  3.7          chain  raced         243267     285211    1.17     31.1  5.9e-16
     35x1029  5.7          chain  raced          86178      78085    0.89     31.6  5.7e-16
       75x21  3.5^2        chain+rb raced           2843       2559    0.76     29.4  4.4e-16
       75x35  3.5^2        chain  raced           4836       4285    0.87     30.8  4.6e-16
     75x1029  3.5^2        chain  raced         207028     177882    0.85     30.3  6.6e-16
    135x1029  3^3.5        chain  raced         385164     341932    0.89     30.8  6.0e-16
    135x4375  3^3.5        chain  raced        2011875    1856450    0.91     28.1  6.8e-16
     525x525  3.5^2.7      chain  raced         772125     870681    1.12     32.3  7.1e-16
   1029x1029  3.7^3        chain  raced        3995712    5012593    1.21     26.5  6.8e-16
    2187x525  3^7          chain  raced        4152162    5635162    1.27     27.8  7.9e-16
   2187x2187  3^7          chain  raced       22480300   34757737    1.43     23.6  9.1e-16
     4375x21  5^4.7        chain+rb raced         269433     270240    0.99     28.1  6.1e-16
    4375x315  5^4.7        chain  raced        5121612    7206168    1.38     27.4  9.4e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           14      0      5   0.87   1.19   1.42    1.12
 chain+rb                         2      1      2   0.76   0.88   0.99    0.87
 csk                              1      0      0   1.39   1.39   1.39    1.39
 ALL                             17      1      7   0.85   1.17   1.42    1.10
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 odd column                      17      1      7   0.85   1.17   1.42    1.10
 ALL                             17      1      7   0.85   1.17   1.42    1.10
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  10      0      4   0.89   1.15   1.43    1.11
 4097..65536                      4      0      1   0.89   1.29   1.39    1.20
 1025..4096                       3      1      2   0.76   0.87   1.42    0.98
 ALL                             17      1      7   0.85   1.17   1.42    1.10
```


worst 10: 75x21 (chain+rb 0.76), 75x1029 (chain 0.85), 75x35 (chain 0.87), 135x1029 (chain 0.89), 35x1029 (chain 0.89), 135x4375 (chain 0.91), 4375x21 (chain+rb 0.99), 525x525 (chain 1.12), 21x4375 (chain 1.17), 1029x1029 (chain 1.21)
best 5: 2187x2187 (chain 1.43), 21x135 (chain 1.42), 21x315 (csk 1.39), 4375x315 (chain 1.38), 21x525 (chain 1.31)
