# gauntlet report

run: `zrf2_big_c2r_fftw_2026-10-01`  contract file suffix: `_c2r_fftw`  cells: 19 listed, 19 benched, comparator: FFTW

control cell: 4 readings, 1.092..1.109 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      3375  3^3.5^3          -        replayed        2900       7486    2.49  1.4e-15  
      6561  3^8              -        replayed        7128      18900    2.59  1.8e-15  
     10125  3^4.5^3          -        replayed        9889      28026    2.83  1.4e-15  
     16875  3^3.5^4          -        replayed       16930      49979    2.92  1.2e-15  
     30375  3^5.5^3          -        replayed       31879      96407    2.99  1.6e-15  
     50625  3^4.5^4          -        replayed       56415     165886    2.90  1.6e-15  
     59049  3^10             -        replayed       69566     207309    2.88  2.1e-15  
     78125  5^7              -        replayed       97257     273896    2.80  1.4e-15  
    117649  7^6              -        replayed      165933     510554    2.97  1.9e-15  
    151875  3^5.5^4          -        replayed      200458     765180    3.81  1.7e-15  
    177147  3^11             -        replayed      259482     967366    3.65  2.4e-15  
    253125  3^4.5^5          -        replayed      363567    1334400    3.65  1.8e-15  
    390625  5^8              -        replayed      619930    2147645    3.44  1.6e-15  
    531441  3^12             -        replayed      852837    3396331    3.96  2.7e-15  
    759375  3^5.5^5          -        replayed     1334250    4710675    3.45  2.0e-15  
   1265625  3^4.5^6          -        replayed     3097175    8935068    2.88  1.9e-15  
   1594323  3^13             -        replayed     4372313   13354194    2.98  5.1e-15  
   1953125  5^9              -        replayed     6797563   14762237    2.15  1.9e-15  
   4782969  3^14             -        replayed    18294062   48556062    2.58  5.9e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           19      0      0   2.49   2.92   3.81    3.01
 ALL         19      0      0   2.49   2.92   3.81    3.01
```


## by size
```
 band               cells median   <1.0   <0.8
 2048..8191             2   2.54      0      0
 8192..32767            3   2.92      0      0
 32768..131071          4   2.89      0      0
 131072..524287         4   3.65      0      0
 524288..2097151        5   2.98      0      0
 2097152..4782969       1   2.58      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                               19   2.92      0   3.01
```


flip agreement: our two readings more than 25% apart at 0 of 19 cells.

worst 10: 1953125 (- 2.15), 3375 (- 2.49), 4782969 (- 2.58), 6561 (- 2.59), 78125 (- 2.80), 10125 (- 2.83), 59049 (- 2.88), 1265625 (- 2.88), 50625 (- 2.90), 16875 (- 2.92)
best 5: 531441 (- 3.96), 151875 (- 3.81), 177147 (- 3.65), 253125 (- 3.65), 759375 (- 3.45)
