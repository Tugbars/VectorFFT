# gauntlet report

run: `tcbmt_c2r_k64_t8_2026-10-03`  contract file suffix: `_c2r_k64_mt8`  cells: 15 listed, 15 benched, comparator: MKL

control cell: 3 readings, 1.291..1.462 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        45  3^2.5            -        raced            566        985    1.58  1.1e-15  
        97  97               -        raced           2699       6072    2.25  1.3e-15  
       105  3.5.7            -        raced            913       1444    1.37  1.1e-15  
       159  3.53             -        raced           3776      13291    3.48  1.1e-15  
       225  3^2.5^2          -        raced           1693       2588    1.23  1.2e-15  
       251  251              -        raced           6279      10872    1.72  1.8e-15  
       315  3^2.5.7          -        raced           2506       3589    1.38  1.4e-15  
       509  509              -        raced          14827      23675    1.44  1.9e-15  
      1001  7.11.13          -        raced           6558      15235    1.69  1.8e-15  flips differ 1.39x
      1009  1009             -        raced          39195      54292    1.16  1.4e-15  
      2003  2003             -        raced          64252     113022    1.63  1.6e-15  
      2025  3^4.5^2          -        raced          14103      32138    1.87  1.8e-15  
      3465  3^2.5.7.11       -        raced         247022      62480    0.20  1.4e-15  
      4099  4099             -        raced         169700     458187    2.58  1.8e-15  
      6561  3^8              -        replayed      482822     133722    0.28  2.2e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           15      2      2   0.28   1.58   2.58    1.31
 ALL         15      2      2   0.28   1.58   2.58    1.31
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                3   1.58      0      0
 128..511               5   1.44      0      0
 512..2047              4   1.66      0      0
 2048..6561             3   0.28      2      2
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                               15   1.58      2   1.31
```


flip agreement: our two readings more than 25% apart at 1 of 15 cells.

worst 10: 3465 (- 0.20), 6561 (- 0.28), 1009 (- 1.16), 225 (- 1.23), 105 (- 1.37), 315 (- 1.38), 509 (- 1.44), 45 (- 1.58), 2003 (- 1.63), 1001 (- 1.69)
best 5: 159 (- 3.48), 4099 (- 2.58), 97 (- 2.25), 2025 (- 1.87), 251 (- 1.72)
