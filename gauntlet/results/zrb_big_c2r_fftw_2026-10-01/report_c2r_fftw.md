# gauntlet report

run: `zrb_big_c2r_fftw_2026-10-01`  contract file suffix: `_c2r_fftw`  cells: 12 listed, 9 benched, comparator: FFTW

control cell: 4 readings, 1.086..1.095 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      4099  4099             -        replayed       20107      70976    3.51  1.4e-15  
      6563  6563             -        replayed       28213     120536    3.58  1.3e-15  
     10007  10007            -        replayed       46338     188878    3.97  1.7e-15  
     16411  16411            -        replayed       95044     326807    3.43  1.5e-15  
     30011  30011            -        replayed      178404     610960    3.41  2.0e-15  
     50021  50021            -        replayed      338794    1175805    3.11  1.7e-15  
     65537  65537            -        raced         412126     953997    2.21  1.6e-15  
    100003  100003           -        replayed      801600    3651597    4.51  1.8e-15  
    131071  131071           -        replayed      988410    5008995    5.02  2.0e-15  
    262147  262147           -        refused            -          -       -        -  not benched
    524287  524287           -        refused            -          -       -        -  not benched
   1000003  1000003          -        refused            -          -       -        -  not benched
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            9      0      0   2.21   3.51   5.02    3.56
 ALL          9      0      0   2.21   3.51   5.02    3.56
```


## by size
```
 band               cells median   <1.0   <0.8
 2048..8191             2   3.55      0      0
 8192..32767            3   3.43      0      0
 32768..131071          4   3.81      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                                9   3.51      0   3.56
```


flip agreement: our two readings more than 25% apart at 0 of 9 cells.

worst 10: 65537 (- 2.21), 50021 (- 3.11), 30011 (- 3.41), 16411 (- 3.43), 4099 (- 3.51), 6563 (- 3.58), 10007 (- 3.97), 100003 (- 4.51), 131071 (- 5.02)
best 5: 131071 (- 5.02), 100003 (- 4.51), 10007 (- 3.97), 6563 (- 3.58), 4099 (- 3.51)
