# gauntlet report

run: `zrb_big_c2r_2026-10-01`  contract file suffix: `_c2r`  cells: 12 listed, 9 benched, comparator: MKL

control cell: 4 readings, 1.261..1.277 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      4099  4099             -        raced          21980      55626    2.35  1.2e-15  
      6563  6563             -        raced          28194      58697    2.08  1.3e-15  
     10007  10007            -        raced          46790     113734    2.37  1.5e-15  
     16411  16411            -        raced          95285     311004    3.06  1.3e-15  
     30011  30011            -        raced         176555     327801    1.55  1.7e-15  
     50021  50021            -        raced         332349     750033    2.25  1.7e-15  
     65537  65537            -        raced         411174    1776972    4.27  1.4e-15  
    100003  100003           -        raced         811418    1805649    2.20  1.7e-15  
    131071  131071           -        raced         956260    1846471    1.87  2.1e-15  
    262147  262147           -        refused            -          -       -        -  not benched
    524287  524287           -        refused            -          -       -        -  not benched
   1000003  1000003          -        refused            -          -       -        -  not benched
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            9      0      0   1.55   2.25   4.27    2.35
 ALL          9      0      0   1.55   2.25   4.27    2.35
```


## by size
```
 band               cells median   <1.0   <0.8
 2048..8191             2   2.21      0      0
 8192..32767            3   2.37      0      0
 32768..131071          4   2.22      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                                9   2.25      0   2.35
```


flip agreement: our two readings more than 25% apart at 0 of 9 cells.

worst 10: 30011 (- 1.55), 131071 (- 1.87), 6563 (- 2.08), 100003 (- 2.20), 50021 (- 2.25), 4099 (- 2.35), 10007 (- 2.37), 16411 (- 3.06), 65537 (- 4.27)
best 5: 65537 (- 4.27), 16411 (- 3.06), 10007 (- 2.37), 4099 (- 2.35), 50021 (- 2.25)
