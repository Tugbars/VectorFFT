# gauntlet report

run: `pow2rec_t1_mkl_retime_2026-10-20`  contract file suffix: `(oop, T=1)`  cells: 2 listed, 2 benched, comparator: MKL

control cell: 4 readings, 0.647..1.084 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      1024  2^10             ztt      replayed         721        840    1.14  4.0e-16  
     65536  2^16             ztt      replayed       90667     114461    1.25  4.8e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt          2      0      0   1.14   1.20   1.25    1.19
 ALL          2      0      0   1.14   1.20   1.25    1.19
```


## by size
```
 band               cells median   <1.0   <0.8
 512..2047              1   1.14      0      0
 32768..65536           1   1.25      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             2   1.20      0   1.19
```


flip agreement: our two readings more than 25% apart at 0 of 2 cells.

worst 10: 1024 (ztt 1.14), 65536 (ztt 1.25)
best 5: 65536 (ztt 1.25), 1024 (ztt 1.14)
