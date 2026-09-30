# gauntlet report

run: `zfsr_top_c2r_mkl_2026-09-30`  contract file suffix: `_c2r`  cells: 5 listed, 5 benched, comparator: MKL

control cell: 4 readings, 1.248..1.272 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
    524288  2^19             -        raced         728500     851937    1.16  1.2e-15  
   1048576  2^20             -        raced        1927350    1822918    0.93  1.2e-15  
   2097152  2^21             -        raced        4806275    4695775    0.96  1.4e-15  
   4194304  2^22             -        raced       12065788   11973100    0.99  1.4e-15  
   8388608  2^23             -        raced       26195912   27705225    1.05  1.4e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            5      0      3   0.93   0.99   1.16    1.02
 ALL          5      0      3   0.93   0.99   1.16    1.02
```


## by size
```
 band               cells median   <1.0   <0.8
 524288..2097151        2   1.05      1      0
 2097152..8388607       2   0.97      2      0
 8388608..8388608       1   1.05      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             5   0.99      3   1.02
```


flip agreement: our two readings more than 25% apart at 0 of 5 cells.

worst 10: 1048576 (- 0.93), 2097152 (- 0.96), 4194304 (- 0.99), 8388608 (- 1.05), 524288 (- 1.16)
best 5: 524288 (- 1.16), 8388608 (- 1.05), 4194304 (- 0.99), 2097152 (- 0.96), 1048576 (- 0.93)
