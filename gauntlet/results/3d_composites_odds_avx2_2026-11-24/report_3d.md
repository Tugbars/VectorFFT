# gauntlet report (3D)

run: `3d_composites_odds_avx2_2026-11-24`  contract: 3D c2c interleaved, natural, out of place, K=1  cells: 22 listed, 22 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.405..1.443


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     6x10x18  2.3          2p     raced           1400       3141    2.21     38.9  4.5e-16
  6x150x1050  2.3          chain3 raced        3084025    4350725    1.39     30.4  6.6e-16
     9x21x35  3^2          2p     raced          11154      20131    1.80     37.6  4.6e-16
   9x75x1029  3^2          flat   raced        2071550    2385488    1.11     32.5  6.9e-16
  10x18x1050  2.5          chain3 raced         448750     507880    1.13     36.9  6.4e-16
    10x42x70  2.5          2p     raced          45854      74526    1.62     47.6  5.5e-16
    15x35x75  3.5          2p     raced          73836      97209    1.31     40.7  7.3e-16
    18x18x18  2.3^2        2p     raced          10377      13315    1.28     35.2  7.2e-16
  18x42x1050  2.3^2        chain3 raced        2252038    2505993    1.09     34.5  7.3e-16
   18x70x150  2.3^2        2p     raced         419220     492810    1.15     39.5  6.4e-16
    21x21x21  3.7          2p     replayed       18667      33301    1.78     32.7  5.3e-16
    21x35x75  3.7          2p     raced         119242     168257    1.40     36.4  6.2e-16
   21x75x315  3.7          2p     raced        1329300    1778006    1.33     35.3  9.3e-16
   35x35x315  5.7          2p     raced         927900    1215912    1.30     38.6  1.0e-15
  35x75x1029  5.7          flat   raced       10427125   13155600    1.25     27.7  6.6e-16
   42x42x150  2.3.7        2p     raced         564950     736681    1.29     42.2  6.0e-16
    42x70x70  2.3.7        2p     raced         444078     574483    1.28     40.9  5.1e-16
  42x150x150  2.3.7        2p     raced        2445950    3286562    1.29     38.3  8.5e-16
    70x70x70  2.5.7        2p     raced         738787     957131    1.28     42.7  5.4e-16
    75x75x75  3.5^2        2p     replayed     1096638    1235168    1.12     35.9  8.1e-16
 135x135x135  3^3.5        2p     raced        8563500   11202606    1.29     30.5  8.3e-16
 150x150x150  2.3.5^2      2p     raced       12035838   14333519    1.18     30.4  9.2e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 2p                              17      0      0   1.15   1.29   1.80    1.38
 chain3                           3      0      0   1.09   1.13   1.39    1.20
 flat                             2      0      0   1.11   1.18   1.25    1.18
 ALL                             22      0      0   1.12   1.29   1.78    1.34
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 even column                     12      0      0   1.13   1.28   1.62    1.32
 odd column                      10      0      0   1.12   1.30   1.80    1.35
 ALL                             22      0      0   1.12   1.29   1.78    1.34
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  15      0      0   1.11   1.28   1.33    1.23
 4097..65536                      6      0      0   1.28   1.51   1.80    1.52
 1025..4096                       1      0      0   2.21   2.21   2.21    2.21
 ALL                             22      0      0   1.12   1.29   1.78    1.34
```


worst 10: 18x42x1050 (chain3 1.09), 9x75x1029 (flat 1.11), 75x75x75 (2p 1.12), 10x18x1050 (chain3 1.13), 18x70x150 (2p 1.15), 150x150x150 (2p 1.18), 35x75x1029 (flat 1.25), 18x18x18 (2p 1.28), 70x70x70 (2p 1.28), 42x70x70 (2p 1.28)
best 5: 6x10x18 (2p 2.21), 9x21x35 (2p 1.80), 21x21x21 (2p 1.78), 10x42x70 (2p 1.62), 21x35x75 (2p 1.40)
