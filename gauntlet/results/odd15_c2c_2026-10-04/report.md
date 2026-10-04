# gauntlet report

run: `odd15_c2c_2026-10-04`  contract file suffix: `(oop, T=1)`  cells: 15 listed, 15 benched, comparator: MKL

control cell: 4 readings, 1.070..1.085 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      3375  3^3.5^3          chain3   raced           5115       6924    1.26  5.9e-16  
      6561  3^8              flat     raced          13764      15680    1.13  9.0e-16  
     10125  3^4.5^3          flat     raced          19464      23701    1.14  7.6e-16  
     15625  5^6              chain3   raced          32328      35902    0.94  6.9e-16  
     19683  3^9              flat     raced          44024      52536    1.19  1.1e-15  
     30375  3^5.5^3          flat     raced          71638      79987    1.11  1.4e-15  
     45927  3^8.7            flat     raced         113423     128253    1.11  1.0e-15  
     50625  3^4.5^4          flat     raced         137977     140675    1.02  1.3e-15  
     59049  3^10             flat     raced         152330     182153    1.18  1.1e-15  
     99225  3^4.5^2.7^2      flat     raced         258075     330227    1.28  7.1e-16  
    117649  7^6              flat     raced         322456     360015    1.10  9.1e-16  
    151875  3^5.5^4          flat     raced         439700     571507    1.28  9.5e-16  
    177147  3^11             flat     raced         581309     719482    1.14  1.2e-15  
    225225  3^2.5^2.7.11.13  flat     raced         658913     949268    1.17  9.8e-16  
    253125  3^4.5^5          flat     raced         748500    1032987    1.14  1.1e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 flat        13      0      0   1.10   1.14   1.28    1.15
 chain3       2      0      1   0.94   1.10   1.26    1.09
 ALL         15      0      1   1.02   1.14   1.28    1.14
```


## by size
```
 band               cells median   <1.0   <0.8
 2048..8191             2   1.19      0      0
 8192..32767            4   1.13      1      0
 32768..131071          5   1.11      0      0
 131072..253125         4   1.15      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 flat                                            13   1.14      0   1.15
 chain3                                           2   1.10      1   1.09
```


flip agreement: our two readings more than 25% apart at 0 of 15 cells.

worst 10: 15625 (chain3 0.94), 50625 (flat 1.02), 117649 (flat 1.10), 45927 (flat 1.11), 30375 (flat 1.11), 6561 (flat 1.13), 253125 (flat 1.14), 10125 (flat 1.14), 177147 (flat 1.14), 225225 (flat 1.17)
best 5: 99225 (flat 1.28), 151875 (flat 1.28), 3375 (chain3 1.26), 19683 (flat 1.19), 59049 (flat 1.18)
