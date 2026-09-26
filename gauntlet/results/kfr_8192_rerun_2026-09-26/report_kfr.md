# gauntlet report

run: `kfr_8192_rerun_2026-09-26`  contract file suffix: `_kfr`  cells: 1 listed, 1 benched, comparator: MKL

control cell: 4 readings, 1.320..1.342 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      8192  2^13             ztt      replayed        7950      10609    1.33  4.1e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt          1      0      0   1.33   1.33   1.33    1.33
 ALL          1      0      0   1.33   1.33   1.33    1.33
```


## by size
```
 band               cells median   <1.0   <0.8
 8192..8192             1   1.33      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             1   1.33      0   1.33
```


flip agreement: our two readings more than 25% apart at 0 of 1 cells.

worst 10: 8192 (ztt 1.33)
best 5: 8192 (ztt 1.33)
