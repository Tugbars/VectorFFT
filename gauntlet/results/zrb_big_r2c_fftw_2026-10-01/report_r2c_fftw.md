# gauntlet report

run: `zrb_big_r2c_fftw_2026-10-01`  contract file suffix: `_r2c_fftw`  cells: 12 listed, 9 benched, comparator: FFTW

control cell: 4 readings, 1.144..1.153 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      4099  4099             -        replayed       19761      70142    3.53  7.2e-16  
      6563  6563             -        replayed       28066     119513    4.13  6.5e-16  
     10007  10007            -        replayed       46952     186737    3.91  7.7e-16  
     16411  16411            -        replayed       94977     323464    3.40  6.8e-16  
     30011  30011            -        replayed      175931     607607    3.43  6.9e-16  
     50021  50021            -        replayed      320846    1122178    3.22  7.5e-16  
     65537  65537            -        raced         409449     942812    2.30  3.0e-15  
    100003  100003           -        replayed      786090    3633663    4.57  7.7e-16  
    131071  131071           -        replayed      954837    5010085    5.18  7.4e-16  
    262147  262147           -        refused            -          -       -        -  not benched
    524287  524287           -        refused            -          -       -        -  not benched
   1000003  1000003          -        refused            -          -       -        -  not benched
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            9      0      0   2.30   3.53   5.18    3.66
 ALL          9      0      0   2.30   3.53   5.18    3.66
```


## by size
```
 band               cells median   <1.0   <0.8
 2048..8191             2   3.83      0      0
 8192..32767            3   3.43      0      0
 32768..131071          4   3.89      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                                9   3.53      0   3.66
```


flip agreement: our two readings more than 25% apart at 0 of 9 cells.

worst 10: 65537 (- 2.30), 50021 (- 3.22), 16411 (- 3.40), 30011 (- 3.43), 4099 (- 3.53), 10007 (- 3.91), 6563 (- 4.13), 100003 (- 4.57), 131071 (- 5.18)
best 5: 131071 (- 5.18), 100003 (- 4.57), 6563 (- 4.13), 10007 (- 3.91), 4099 (- 3.53)
