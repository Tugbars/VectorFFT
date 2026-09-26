# gauntlet report

run: `pow2_2p23_2p24_2026-09-26`  contract file suffix: `(oop, T=1)`  cells: 2 listed, 2 benched, comparator: MKL

control cell: 4 readings, 1.048..1.078 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
   8388608  2^23             fs       raced       46026088   62206631    1.34  1.2e-15  
  16777216  2^24             fs       raced       99645425  127755906    1.27  3.2e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 fs           2      0      0   1.27   1.31   1.34    1.31
 ALL          2      0      0   1.27   1.31   1.34    1.31
```


## by size
```
 band               cells median   <1.0   <0.8
 8388608..16777216      2   1.31      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             2   1.31      0   1.31
```


flip agreement: our two readings more than 25% apart at 0 of 2 cells.

worst 10: 16777216 (fs 1.27), 8388608 (fs 1.34)
best 5: 8388608 (fs 1.34), 16777216 (fs 1.27)
