# gauntlet report

run: `mergechk_1d_est_20261110`  contract file suffix: `(oop, T=1)`  cells: 5 listed, 5 benched, comparator: MKL

control cell: 4 readings, 1.076..1.086 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      1024  2^10             ztt      replayed         741        836    1.10  2.5e-16  
      4096  2^12             ztt      replayed        3552       3841    1.08  3.9e-16  
     65536  2^16             ztt      replayed       91150     115570    1.20  4.8e-16  
   1048576  2^20             fs       replayed     3954487    5094900    1.28  4.6e-16  
   4194304  2^22             fs       replayed    21574613   29479231    1.35  9.4e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt          3      0      0   1.08   1.10   1.20    1.13
 fs           2      0      0   1.28   1.31   1.35    1.31
 ALL          5      0      0   1.08   1.20   1.35    1.20
```


## by size
```
 band               cells median   <1.0   <0.8
 512..2047              1   1.10      0      0
 2048..8191             1   1.08      0      0
 32768..131071          1   1.20      0      0
 524288..2097151        1   1.28      0      0
 2097152..4194304       1   1.35      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             5   1.20      0   1.20
```


flip agreement: our two readings more than 25% apart at 0 of 5 cells.

worst 10: 4096 (ztt 1.08), 1024 (ztt 1.10), 65536 (ztt 1.20), 1048576 (fs 1.28), 4194304 (fs 1.35)
best 5: 4194304 (fs 1.35), 1048576 (fs 1.28), 65536 (ztt 1.20), 1024 (ztt 1.10), 4096 (ztt 1.08)
