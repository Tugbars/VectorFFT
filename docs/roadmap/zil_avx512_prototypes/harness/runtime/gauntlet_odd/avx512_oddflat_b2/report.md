# gauntlet report

run: `avx512_oddflat_b2`  contract file suffix: `(oop, T=1)`  cells: 7 listed, 7 benched, comparator: MKL

control cell: 4 readings, 0.922..1.669 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      1125  3^2.5^3          chain3   replayed        2063       2238    0.95  8.7e-16  flips differ 1.28x
      1215  3^5.5            chain3   replayed        2768       2433    0.74  5.5e-16  
      1575  3^2.5^2.7        chain3   replayed        3259       2498    0.59  8.2e-16  flips differ 1.36x
      2025  3^4.5^2          chain3   replayed        3751       5998    1.23  5.9e-16  
      2187  3^7              chain3   replayed        4196       6337    1.30  5.4e-16  flips differ 1.27x
      2401  7^4              flat     replayed        7815       5704    0.63  6.5e-16  
      2835  3^4.5.7          chain3   replayed        6544       7885    1.18  5.3e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 chain3       6      2      3   0.59   1.06   1.30    0.96
 flat         1      1      1   0.63   0.63   0.63    0.63
 ALL          7      3      4   0.59   0.95   1.30    0.90
```


## by size
```
 band               cells median   <1.0   <0.8
 512..2047              4   0.84      3      2
 2048..2835             3   1.18      1      1
```


## by family
```
 family                                       cells median   <1.0  gmean
 chain3                                           6   1.06      3   0.96
 flat                                             1   0.63      1   0.63
```


flip agreement: our two readings more than 25% apart at 3 of 7 cells.

worst 10: 1575 (chain3 0.59), 2401 (flat 0.63), 1215 (chain3 0.74), 1125 (chain3 0.95), 2835 (chain3 1.18), 2025 (chain3 1.23), 2187 (chain3 1.30)
best 5: 2187 (chain3 1.30), 2025 (chain3 1.23), 2835 (chain3 1.18), 1125 (chain3 0.95), 1215 (chain3 0.74)
