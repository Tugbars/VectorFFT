# gauntlet report

run: `tcbmt2_c2r_k64_t8_2026-10-03`  contract file suffix: `_c2r_k64_mt8`  cells: 15 listed, 15 benched, comparator: MKL

control cell: 3 readings, 1.306..1.375 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        45  3^2.5            -        replayed         573        982    1.62  1.1e-15  
        97  97               -        replayed        2494       6283    2.52  1.3e-15  
       105  3.5.7            -        replayed         893       1545    1.73  1.1e-15  
       159  3.53             -        replayed        3750      13378    3.54  1.1e-15  
       225  3^2.5^2          -        replayed        1641       2633    1.59  1.2e-15  
       251  251              -        replayed        6224      11115    1.70  1.8e-15  
       315  3^2.5.7          -        replayed        2205       3671    1.47  1.4e-15  
       509  509              -        replayed       14585      23708    1.56  1.9e-15  
      1001  7.11.13          -        replayed        6429      15360    2.35  1.8e-15  
      1009  1009             -        replayed       41010      60240    1.41  1.4e-15  
      2003  2003             -        replayed       67861     124621    1.80  1.6e-15  
      2025  3^4.5^2          -        replayed       15050      33527    2.23  1.8e-15  
      3465  3^2.5.7.11       -        replayed       27728      57306    2.07  1.4e-15  
      4099  4099             -        replayed      179593     512353    2.85  1.8e-15  
      6561  3^8              -        replayed       70278     134167    1.91  2.2e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           15      0      0   1.47   1.80   2.85    1.95
 ALL         15      0      0   1.47   1.80   2.85    1.95
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                3   1.73      0      0
 128..511               5   1.59      0      0
 512..2047              4   2.01      0      0
 2048..6561             3   2.07      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                               15   1.80      0   1.95
```


flip agreement: our two readings more than 25% apart at 0 of 15 cells.

worst 10: 1009 (- 1.41), 315 (- 1.47), 509 (- 1.56), 225 (- 1.59), 45 (- 1.62), 251 (- 1.70), 105 (- 1.73), 2003 (- 1.80), 6561 (- 1.91), 3465 (- 2.07)
best 5: 159 (- 3.54), 4099 (- 2.85), 97 (- 2.52), 1001 (- 2.35), 2025 (- 2.23)
