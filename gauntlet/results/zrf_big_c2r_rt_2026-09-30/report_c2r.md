# gauntlet report

run: `zrf_big_c2r_rt_2026-09-30`  contract file suffix: `_c2r`  cells: 7 listed, 7 benched, comparator: MKL

control cell: 4 readings, 1.267..1.276 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
    390625  5^8              -        raced         658880    1692645    2.56  1.8e-15  
    531441  3^12             -        raced         900025    2594669    2.88  2.7e-15  
    759375  3^5.5^5          -        raced        1418475    3835531    2.61  2.1e-15  
   1265625  3^4.5^6          -        raced        2771550    8109912    2.81  2.7e-15  
   1594323  3^13             -        raced        4082088   10685825    2.61  2.9e-15  
   1953125  5^9              -        raced        6584238   12938143    1.95  4.5e-15  
   4782969  3^14             -        raced       16742475   51920644    3.09  3.2e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            7      0      0   1.95   2.61   3.09    2.62
 ALL          7      0      0   1.95   2.61   3.09    2.62
```


## by size
```
 band               cells median   <1.0   <0.8
 131072..524287         1   2.56      0      0
 524288..2097151        5   2.61      0      0
 2097152..4782969       1   3.09      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                                7   2.61      0   2.62
```


flip agreement: our two readings more than 25% apart at 0 of 7 cells.

worst 10: 1953125 (- 1.95), 390625 (- 2.56), 759375 (- 2.61), 1594323 (- 2.61), 1265625 (- 2.81), 531441 (- 2.88), 4782969 (- 3.09)
best 5: 4782969 (- 3.09), 531441 (- 2.88), 1265625 (- 2.81), 759375 (- 2.61), 1594323 (- 2.61)
