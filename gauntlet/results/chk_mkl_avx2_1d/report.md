# gauntlet report

run: `chk_mkl_avx2_1d`  contract file suffix: `(oop, T=1)`  cells: 2 listed, 2 benched, comparator: MKL

control cell: 4 readings, 1.073..1.083 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        64  2^6              2p       replayed          30         31    1.04  1.5e-16  
      4096  2^12             ztt      replayed        3552       3945    1.08  3.9e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 2p           1      0      0   1.04   1.04   1.04    1.04
 ztt          1      0      0   1.08   1.08   1.08    1.08
 ALL          2      0      0   1.04   1.06   1.08    1.06
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                1   1.04      0      0
 2048..4096             1   1.08      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             2   1.06      0   1.06
```


flip agreement: our two readings more than 25% apart at 0 of 2 cells.

worst 10: 64 (2p 1.04), 4096 (ztt 1.08)
best 5: 4096 (ztt 1.08), 64 (2p 1.04)
