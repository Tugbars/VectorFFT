# gauntlet report

run: `avx512_oddflat_scratch`  contract file suffix: `(oop, T=1)`  cells: 7 listed, 7 benched, comparator: MKL

control cell: 4 readings, 0.931..1.557 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      1125  3^2.5^3          chain3   raced           2042       2229    0.74  8.7e-16  flips differ 1.30x
      1215  3^5.5            chain3   raced           2755       2859    0.99  5.5e-16  
      1575  3^2.5^2.7        chain3   raced           3310       2391    0.62  8.2e-16  
      2025  3^4.5^2          chain3   raced           3452       6308    1.42  5.9e-16  flips differ 1.31x
      2187  3^7              chain3   raced           4209       5412    1.01  5.4e-16  flips differ 1.27x
      2401  7^4              flat     raced           6824       5620    0.71  6.5e-16  
      2835  3^4.5.7          chain3   raced           6495       7324    0.98  5.3e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 chain3       6      2      4   0.62   0.99   1.42    0.93
 flat         1      1      1   0.71   0.71   0.71    0.71
 ALL          7      3      5   0.62   0.98   1.42    0.90
```


## by size
```
 band               cells median   <1.0   <0.8
 512..2047              4   0.87      3      2
 2048..2835             3   0.98      2      1
```


## by family
```
 family                                       cells median   <1.0  gmean
 chain3                                           6   0.99      4   0.93
 flat                                             1   0.71      1   0.71
```


flip agreement: our two readings more than 25% apart at 3 of 7 cells.

worst 10: 1575 (chain3 0.62), 2401 (flat 0.71), 1125 (chain3 0.74), 2835 (chain3 0.98), 1215 (chain3 0.99), 2187 (chain3 1.01), 2025 (chain3 1.42)
best 5: 2025 (chain3 1.42), 2187 (chain3 1.01), 1215 (chain3 0.99), 2835 (chain3 0.98), 1125 (chain3 0.74)
