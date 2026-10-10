# gauntlet report

run: `kfr_smoke_c2r_1024`  contract file suffix: `_c2r_kfr`  cells: 1 listed, 1 benched, comparator: KFR

control cell: 4 readings, 1.147..1.161 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      1024  2^10             -        replayed         474        495    1.04  6.7e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            1      0      0   1.04   1.04   1.04    1.04
 ALL          1      0      0   1.04   1.04   1.04    1.04
```


## by size
```
 band               cells median   <1.0   <0.8
 512..1024              1   1.04      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             1   1.04      0   1.04
```


flip agreement: our two readings more than 25% apart at 0 of 1 cells.

worst 10: 1024 (- 1.04)
best 5: 1024 (- 1.04)
