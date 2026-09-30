# gauntlet report

run: `zfsr_top_r2c_mkl_2026-09-30`  contract file suffix: `_r2c`  cells: 5 listed, 5 benched, comparator: MKL

control cell: 4 readings, 1.195..1.197 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
    524288  2^19             -        raced         609325     862012    1.36  4.4e-16  
   1048576  2^20             -        raced        1451825    1891206    1.28  5.6e-16  
   2097152  2^21             -        raced        3420962    5214206    1.51  5.4e-16  
   4194304  2^22             -        raced        8691037   12885387    1.46  1.0e-15  
   8388608  2^23             -        raced       21783975   30078869    1.35  7.5e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            5      0      0   1.28   1.36   1.51    1.39
 ALL          5      0      0   1.28   1.36   1.51    1.39
```


## by size
```
 band               cells median   <1.0   <0.8
 524288..2097151        2   1.32      0      0
 2097152..8388607       2   1.49      0      0
 8388608..8388608       1   1.35      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             5   1.36      0   1.39
```


flip agreement: our two readings more than 25% apart at 0 of 5 cells.

worst 10: 1048576 (- 1.28), 8388608 (- 1.35), 524288 (- 1.36), 4194304 (- 1.46), 2097152 (- 1.51)
best 5: 2097152 (- 1.51), 4194304 (- 1.46), 524288 (- 1.36), 8388608 (- 1.35), 1048576 (- 1.28)
