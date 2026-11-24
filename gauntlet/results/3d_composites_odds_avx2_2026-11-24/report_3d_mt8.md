# gauntlet report (3D)

run: `3d_composites_odds_avx2_2026-11-24`  contract: 3D c2c interleaved, natural, out of place, K=1_mt8  cells: 22 listed, 22 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.706..1.880

threaded plans that ran serial (engaged = 0 at both flips): 1: 6x10x18


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     6x10x18  2.3          2p     raced           1388       4678    3.34     39.2  4.5e-16
  6x150x1050  2.3          chain3 raced         554638     738462    1.13    169.1  6.6e-16
     9x21x35  3^2          2p     raced           5695       8496    1.24     73.7  4.6e-16
   9x75x1029  3^2          flat   raced         436950     521462    1.10    154.2  6.9e-16
  10x18x1050  2.5          chain3 raced          98180     139195    0.97    168.7  6.4e-16
    10x42x70  2.5          2p     raced          20854      22057    1.00    104.6  5.1e-16
    15x35x75  3.5          2p     raced          19402      24720    1.09    154.9  8.5e-16
    18x18x18  2.3^2        2p     raced           5402       5189    0.86     67.5  7.2e-16
  18x42x1050  2.3^2        chain3 raced         387237     400300    0.89    200.9  7.3e-16
   18x70x150  2.3^2        2p     raced          76940      83075    0.94    215.3  6.4e-16
    21x21x21  3.7          2p     raced           7775       9211    1.13     78.5  4.5e-16
    21x35x75  3.7          2p     raced          22822      36607    1.18    190.2  6.2e-16
   21x75x315  3.7          chain3 raced         194487     268368    1.01    241.3  7.9e-16
   35x35x315  5.7          2p     raced         151612     254050    1.18    236.2  9.0e-16
  35x75x1029  5.7          flat   raced        2130413    1971293    0.88    135.4  6.9e-16
   42x42x150  2.3.7        2p     raced          97075     116637    1.09    245.5  6.1e-16
    42x70x70  2.3.7        2p     raced          74922      89638    1.04    242.4  6.0e-16
  42x150x150  2.3.7        2p     raced         360025     423169    1.13    260.5  7.2e-16
    70x70x70  2.5.7        2p     raced         113237     138544    1.10    278.5  5.1e-16
    75x75x75  3.5^2        2p     raced         186538     186700    0.82    211.3  7.4e-16
 135x135x135  3^3.5        2p     raced        1701775    1652918    0.94    153.5  7.4e-16
 150x150x150  2.3.5^2      2p     raced        3028062    2738125    0.90    120.9  8.3e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 2p                              16      0      5   0.86   1.09   1.24    1.11
 chain3                           4      0      2   0.89   0.99   1.13    1.00
 flat                             2      0      1   0.88   0.99   1.10    0.98
 ALL                             22      0      8   0.88   1.07   1.18    1.08
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 even column                     12      0      5   0.89   1.02   1.13    1.11
 odd column                      10      0      3   0.88   1.10   1.24    1.05
 ALL                             22      0      8   0.88   1.07   1.18    1.08
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  15      0      7   0.88   1.01   1.13    1.00
 4097..65536                      6      0      1   0.86   1.11   1.24    1.08
 1025..4096                       1      0      0   3.34   3.34   3.34    3.34
 ALL                             22      0      8   0.88   1.07   1.18    1.08
```


worst 10: 75x75x75 (2p 0.82), 18x18x18 (2p 0.86), 35x75x1029 (flat 0.88), 18x42x1050 (chain3 0.89), 150x150x150 (2p 0.90), 18x70x150 (2p 0.94), 135x135x135 (2p 0.94), 10x18x1050 (chain3 0.97), 10x42x70 (2p 1.00), 21x75x315 (chain3 1.01)
best 5: 6x10x18 (2p 3.34), 9x21x35 (2p 1.24), 21x35x75 (2p 1.18), 35x35x315 (2p 1.18), 21x21x21 (2p 1.13)
