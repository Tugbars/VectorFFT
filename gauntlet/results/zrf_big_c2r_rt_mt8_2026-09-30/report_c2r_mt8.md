# gauntlet report

run: `zrf_big_c2r_rt_mt8_2026-09-30`  contract file suffix: `_c2r_mt8`  cells: 7 listed, 7 benched, comparator: MKL

control cell: 2 readings, 1.269..1.322 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
    390625  5^8              -        raced         228380    1700105    7.14  1.9e-15  
    531441  3^12             -        raced         281962    2603731    9.11  2.7e-15  
    759375  3^5.5^5          -        raced         359688    3909575   10.87  2.1e-15  
   1265625  3^4.5^6          -        raced         659725    7996944   11.92  2.5e-15  
   1594323  3^13             -        raced         949850   10473249   10.66  2.8e-15  
   1953125  5^9              -        raced        1947025   12878956    6.33  4.5e-15  
   4782969  3^14             -        raced        4970712   48945187    9.85  5.2e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            7      0      0   6.33   9.85  11.92    9.21
 ALL          7      0      0   6.33   9.85  11.92    9.21
```


## by size
```
 band               cells median   <1.0   <0.8
 131072..524287         1   7.14      0      0
 524288..2097151        5  10.66      0      0
 2097152..4782969       1   9.85      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                                7   9.85      0   9.21
```


flip agreement: our two readings more than 25% apart at 0 of 7 cells.

worst 10: 1953125 (- 6.33), 390625 (- 7.14), 531441 (- 9.11), 4782969 (- 9.85), 1594323 (- 10.66), 759375 (- 10.87), 1265625 (- 11.92)
best 5: 1265625 (- 11.92), 759375 (- 10.87), 1594323 (- 10.66), 4782969 (- 9.85), 531441 (- 9.11)
