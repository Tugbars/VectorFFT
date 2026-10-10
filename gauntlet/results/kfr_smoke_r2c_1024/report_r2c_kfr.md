# gauntlet report

run: `kfr_smoke_r2c_1024`  contract file suffix: `_r2c_kfr`  cells: 1 listed, 1 benched, comparator: KFR

control cell: 4 readings, 1.196..1.203 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      1024  2^10             -        replayed         438        466    1.06  3.4e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            1      0      0   1.06   1.06   1.06    1.06
 ALL          1      0      0   1.06   1.06   1.06    1.06
```


## by size
```
 band               cells median   <1.0   <0.8
 512..1024              1   1.06      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             1   1.06      0   1.06
```


flip agreement: our two readings more than 25% apart at 0 of 1 cells.

worst 10: 1024 (- 1.06)
best 5: 1024 (- 1.06)
