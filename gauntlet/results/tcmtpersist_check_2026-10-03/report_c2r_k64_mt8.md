# gauntlet report

run: `tcmtpersist_check_2026-10-03`  contract file suffix: `_c2r_k64_mt8`  cells: 2 listed, 2 benched, comparator: MKL

control cell: 4 readings, 1.299..1.536 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        45  3^2.5            -        raced            548        955    1.70  1.1e-15  
        97  97               -        raced           2524       6383    2.53  1.3e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            2      0      0   1.70   2.12   2.53    2.08
 ALL          2      0      0   1.70   2.12   2.53    2.08
```


## by size
```
 band               cells median   <1.0   <0.8
 32..97                 2   2.12      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                                2   2.12      0   2.08
```


flip agreement: our two readings more than 25% apart at 0 of 2 cells.

worst 10: 45 (- 1.70), 97 (- 2.53)
best 5: 97 (- 2.53), 45 (- 1.70)
