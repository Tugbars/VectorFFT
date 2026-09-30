# gauntlet report

run: `zrf_big_c2r_2026-09-30`  contract file suffix: `_c2r`  cells: 19 listed, 12 benched, comparator: MKL

control cell: 4 readings, 1.265..1.278 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      3375  3^3.5^3          -        raced           3333       6988    2.09  1.3e-15  
      6561  3^8              -        raced           7227      16039    2.21  1.8e-15  
     10125  3^4.5^3          -        raced          10635      25618    2.38  1.6e-15  
     16875  3^3.5^4          -        raced          18395      44804    2.42  1.5e-15  
     30375  3^5.5^3          -        raced          34799      83878    2.16  1.8e-15  
     50625  3^4.5^4          -        raced          60303     145438    2.32  1.6e-15  
     59049  3^10             -        raced          76364     179485    2.35  2.0e-15  
     78125  5^7              -        raced         108000     231449    2.12  1.6e-15  
    117649  7^6              -        raced         179021     361538    2.02  1.8e-15  
    151875  3^5.5^4          -        raced         221523     561107    2.47  1.8e-15  
    177147  3^11             -        raced         281241     750127    2.59  2.8e-15  
    253125  3^4.5^5          -        raced         403693    1217953    3.01  2.0e-15  
    390625  5^8              -        refused            -          -       -        -  not benched
    531441  3^12             -        refused            -          -       -        -  not benched
    759375  3^5.5^5          -        refused            -          -       -        -  not benched
   1265625  3^4.5^6          -        refused            -          -       -        -  not benched
   1594323  3^13             -        refused            -          -       -        -  not benched
   1953125  5^9              -        refused            -          -       -        -  not benched
   4782969  3^14             -        refused            -          -       -        -  not benched
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           12      0      0   2.09   2.33   2.59    2.33
 ALL         12      0      0   2.09   2.33   2.59    2.33
```


## by size
```
 band               cells median   <1.0   <0.8
 2048..8191             2   2.15      0      0
 8192..32767            3   2.38      0      0
 32768..131071          4   2.22      0      0
 131072..253125         3   2.59      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                               12   2.33      0   2.33
```


flip agreement: our two readings more than 25% apart at 0 of 12 cells.

worst 10: 117649 (- 2.02), 3375 (- 2.09), 78125 (- 2.12), 30375 (- 2.16), 6561 (- 2.21), 50625 (- 2.32), 59049 (- 2.35), 10125 (- 2.38), 16875 (- 2.42), 151875 (- 2.47)
best 5: 253125 (- 3.01), 177147 (- 2.59), 151875 (- 2.47), 16875 (- 2.42), 10125 (- 2.38)
