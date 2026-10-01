# gauntlet report

run: `zrb_big_r2c_2026-10-01`  contract file suffix: `_r2c`  cells: 12 listed, 9 benched, comparator: MKL

control cell: 4 readings, 1.178..1.194 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      4099  4099             -        raced          19905      54641    2.26  5.8e-16  
      6563  6563             -        raced          27907      57283    2.04  6.5e-16  
     10007  10007            -        raced          48053     111348    2.30  7.7e-16  
     16411  16411            -        raced          95689     301828    3.13  5.5e-16  
     30011  30011            -        raced         175974     314854    1.78  7.8e-16  
     50021  50021            -        raced         317278     736343    2.06  7.5e-16  
     65537  65537            -        raced         390777    1737005    4.43  3.1e-15  
    100003  100003           -        raced         773444    1777729    2.27  7.7e-16  
    131071  131071           -        raced         958007    1819645    1.88  7.4e-16  
    262147  262147           -        refused            -          -       -        -  not benched
    524287  524287           -        refused            -          -       -        -  not benched
   1000003  1000003          -        refused            -          -       -        -  not benched
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            9      0      0   1.78   2.26   4.43    2.37
 ALL          9      0      0   1.78   2.26   4.43    2.37
```


## by size
```
 band               cells median   <1.0   <0.8
 2048..8191             2   2.15      0      0
 8192..32767            3   2.30      0      0
 32768..131071          4   2.17      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                                9   2.26      0   2.37
```


flip agreement: our two readings more than 25% apart at 0 of 9 cells.

worst 10: 30011 (- 1.78), 131071 (- 1.88), 6563 (- 2.04), 50021 (- 2.06), 4099 (- 2.26), 100003 (- 2.27), 10007 (- 2.30), 16411 (- 3.13), 65537 (- 4.43)
best 5: 65537 (- 4.43), 16411 (- 3.13), 10007 (- 2.30), 100003 (- 2.27), 4099 (- 2.26)
