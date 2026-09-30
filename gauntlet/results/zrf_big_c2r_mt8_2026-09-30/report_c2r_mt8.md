# gauntlet report

run: `zrf_big_c2r_mt8_2026-09-30`  contract file suffix: `_c2r_mt8`  cells: 19 listed, 12 benched, comparator: MKL

control cell: 2 readings, 1.246..1.251 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      3375  3^3.5^3          -        raced           2097       8729    3.69  1.4e-15  
      6561  3^8              -        raced           3473      17284    4.24  1.8e-15  
     10125  3^4.5^3          -        raced           5511      25510    4.56  1.6e-15  
     16875  3^3.5^4          -        raced           7615      44539    5.75  1.4e-15  
     30375  3^5.5^3          -        raced          14930      83813    4.79  1.8e-15  
     50625  3^4.5^4          -        raced          21657     145499    6.26  1.8e-15  
     59049  3^10             -        raced          24867     185245    7.12  2.0e-15  
     78125  5^7              -        raced          50180     232614    4.43  1.6e-15  
    117649  7^6              -        raced          73573     363688    4.94  1.8e-15  
    151875  3^5.5^4          -        raced          83612     567346    6.55  1.9e-15  
    177147  3^11             -        raced          91177     778548    7.96  2.6e-15  
    253125  3^4.5^5          -        raced         126587    1235513    9.48  2.0e-15  
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
 -           12      0      0   4.24   5.35   7.96    5.60
 ALL         12      0      0   4.24   5.35   7.96    5.60
```


## by size
```
 band               cells median   <1.0   <0.8
 2048..8191             2   3.97      0      0
 8192..32767            3   4.79      0      0
 32768..131071          4   5.60      0      0
 131072..253125         3   7.96      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                               12   5.35      0   5.60
```


flip agreement: our two readings more than 25% apart at 0 of 12 cells.

worst 10: 3375 (- 3.69), 6561 (- 4.24), 78125 (- 4.43), 10125 (- 4.56), 30375 (- 4.79), 117649 (- 4.94), 16875 (- 5.75), 50625 (- 6.26), 151875 (- 6.55), 59049 (- 7.12)
best 5: 253125 (- 9.48), 177147 (- 7.96), 59049 (- 7.12), 151875 (- 6.55), 50625 (- 6.26)
