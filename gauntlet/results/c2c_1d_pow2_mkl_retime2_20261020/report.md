# gauntlet report

run: `pow2rec_t1_mkl_retime2_2026-10-20`  contract file suffix: `(oop, T=1)`  cells: 1 listed, 1 benched, comparator: MKL

control cell: 4 readings, 1.067..1.550 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
    131072  2^17             ztt      replayed      225013     301403    1.28  4.6e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt          1      0      0   1.28   1.28   1.28    1.28
 ALL          1      0      0   1.28   1.28   1.28    1.28
```


## by size
```
 band               cells median   <1.0   <0.8
 131072..131072         1   1.28      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             1   1.28      0   1.28
```


flip agreement: our two readings more than 25% apart at 0 of 1 cells.

worst 10: 131072 (ztt 1.28)
best 5: 131072 (ztt 1.28)
