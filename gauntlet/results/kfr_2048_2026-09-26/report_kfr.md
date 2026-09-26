# gauntlet report

run: `kfr_2048_2026-09-26`  contract file suffix: `_kfr`  cells: 1 listed, 1 benched, comparator: MKL

control cell: 4 readings, 1.321..1.328 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      2048  2^11             ztt      replayed        1603       1998    1.24  3.1e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt          1      0      0   1.24   1.24   1.24    1.24
 ALL          1      0      0   1.24   1.24   1.24    1.24
```


## by size
```
 band               cells median   <1.0   <0.8
 2048..2048             1   1.24      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             1   1.24      0   1.24
```


flip agreement: our two readings more than 25% apart at 0 of 1 cells.

worst 10: 2048 (ztt 1.24)
best 5: 2048 (ztt 1.24)
