# gauntlet report

run: `zrf2_big_c2r_mt8_2026-10-01`  contract file suffix: `_c2r_mt8`  cells: 19 listed, 19 benched, comparator: MKL

control cell: 2 readings, 1.264..1.273 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      3375  3^3.5^3          -        raced           3456       6924    1.58  1.4e-15  flips differ 1.27x
      6561  3^8              -        raced           6255      16021    2.53  1.4e-15  
     10125  3^4.5^3          -        raced           5717      25551    3.58  1.6e-15  flips differ 1.25x
     16875  3^3.5^4          -        raced          10162      44494    4.05  1.5e-15  
     30375  3^5.5^3          -        raced          16320      83664    5.11  2.0e-15  
     50625  3^4.5^4          -        raced          17386     145876    6.54  1.8e-15  flips differ 1.28x
     59049  3^10             -        raced          25299     180664    7.14  2.0e-15  
     78125  5^7              -        raced          37712     246039    5.77  1.5e-15  
    117649  7^6              -        raced          69815     366410    5.05  1.8e-15  
    151875  3^5.5^4          -        raced          45369     565331   12.46  1.8e-15  
    177147  3^11             -        raced          65041     745916    8.65  2.6e-15  flips differ 1.31x
    253125  3^4.5^5          -        raced         130973    1253040    9.40  2.4e-15  
    390625  5^8              -        raced         250620    1716630    6.85  2.0e-15  
    531441  3^12             -        raced         282700    2603825    9.21  2.7e-15  
    759375  3^5.5^5          -        raced         244250    3898625   15.96  2.1e-15  
   1265625  3^4.5^6          -        raced         642512    7904137   11.67  2.2e-15  
   1594323  3^13             -        raced         819763   10352606   12.57  3.1e-15  
   1953125  5^9              -        raced        1846025   12689750    6.87  2.0e-15  
   4782969  3^14             -        raced        4837800   48532224    9.75  5.4e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           19      0      0   2.53   6.87  12.57    6.63
 ALL         19      0      0   2.53   6.87  12.57    6.63
```


## by size
```
 band               cells median   <1.0   <0.8
 2048..8191             2   2.05      0      0
 8192..32767            3   4.05      0      0
 32768..131071          4   6.15      0      0
 131072..524287         4   9.03      0      0
 524288..2097151        5  11.67      0      0
 2097152..4782969       1   9.75      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                               19   6.87      0   6.63
```


flip agreement: our two readings more than 25% apart at 4 of 19 cells.

worst 10: 3375 (- 1.58), 6561 (- 2.53), 10125 (- 3.58), 16875 (- 4.05), 117649 (- 5.05), 30375 (- 5.11), 78125 (- 5.77), 50625 (- 6.54), 390625 (- 6.85), 1953125 (- 6.87)
best 5: 759375 (- 15.96), 1594323 (- 12.57), 151875 (- 12.46), 1265625 (- 11.67), 4782969 (- 9.75)
