# gauntlet report

run: `zrf2_big_r2c_fftw_2026-10-01`  contract file suffix: `_r2c_fftw`  cells: 19 listed, 19 benched, comparator: FFTW

control cell: 4 readings, 1.121..1.144 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      3375  3^3.5^3          -        replayed        3067       6724    2.18  5.4e-16  
      6561  3^8              -        replayed        7037      17388    2.44  9.0e-16  
     10125  3^4.5^3          -        replayed       10682      23296    2.09  6.7e-16  
     16875  3^3.5^4          -        replayed       18184      41762    2.27  6.5e-16  
     30375  3^5.5^3          -        replayed       33753      80877    2.36  6.6e-16  
     50625  3^4.5^4          -        replayed       60191     139524    1.74  6.8e-16  flips differ 1.34x
     59049  3^10             -        replayed       73496     197153    2.59  9.1e-16  
     78125  5^7              -        replayed      111094     260631    2.33  7.0e-16  
    117649  7^6              -        replayed      189700     480092    2.50  6.4e-16  
    151875  3^5.5^4          -        replayed      222465     678684    2.99  7.8e-16  
    177147  3^11             -        replayed      285736     891431    3.04  9.6e-16  
    253125  3^4.5^5          -        replayed      434187    1113726    2.40  6.9e-16  
    390625  5^8              -        replayed      697030    1922910    2.75  6.7e-16  
    531441  3^12             -        replayed     1004775    3210487    3.15  2.4e-15  
    759375  3^5.5^5          -        replayed     1467600    3906412    2.66  2.6e-15  
   1265625  3^4.5^6          -        replayed     2799650    7697818    2.71  2.2e-15  
   1594323  3^13             -        replayed     3926088   13053437    3.29  3.1e-15  
   1953125  5^9              -        replayed     5861362   14672718    2.49  2.5e-15  
   4782969  3^14             -        replayed    15583550   60896012    3.72  2.7e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           19      0      0   2.09   2.50   3.29    2.58
 ALL         19      0      0   2.09   2.50   3.29    2.58
```


## by size
```
 band               cells median   <1.0   <0.8
 2048..8191             2   2.31      0      0
 8192..32767            3   2.27      0      0
 32768..131071          4   2.41      0      0
 131072..524287         4   2.87      0      0
 524288..2097151        5   2.71      0      0
 2097152..4782969       1   3.72      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                               19   2.50      0   2.58
```


flip agreement: our two readings more than 25% apart at 1 of 19 cells.

worst 10: 50625 (- 1.74), 10125 (- 2.09), 3375 (- 2.18), 16875 (- 2.27), 78125 (- 2.33), 30375 (- 2.36), 253125 (- 2.40), 6561 (- 2.44), 1953125 (- 2.49), 117649 (- 2.50)
best 5: 4782969 (- 3.72), 1594323 (- 3.29), 531441 (- 3.15), 177147 (- 3.04), 151875 (- 2.99)
