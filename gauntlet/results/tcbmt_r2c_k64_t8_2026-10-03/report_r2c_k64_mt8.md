# gauntlet report

run: `tcbmt_r2c_k64_t8_2026-10-03`  contract file suffix: `_r2c_k64_mt8`  cells: 15 listed, 15 benched, comparator: MKL

control cell: 2 readings, 1.053..1.231 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        45  3^2.5            -        raced            572        914    1.49  4.3e-16  
        97  97               -        raced           2422       5799    2.39  6.0e-16  
       105  3.5.7            -        raced            949       1416    1.38  3.9e-16  
       159  3.53             -        raced           3685      13355    3.41  5.4e-16  
       225  3^2.5^2          -        raced           1575       2624    1.62  5.2e-16  
       251  251              -        raced           5900      10682    1.72  7.1e-16  
       315  3^2.5.7          -        raced           2037       3678    1.78  7.5e-16  
       509  509              -        raced          14740      25234    1.68  8.5e-16  
      1001  7.11.13          -        raced           7906      13897    1.65  1.0e-15  
      1009  1009             -        raced          39346      52874    1.33  5.7e-16  
      2003  2003             -        raced          62206     107168    1.72  6.6e-16  
      2025  3^4.5^2          -        raced          15310      28943    1.89  6.4e-16  
      3465  3^2.5.7.11       -        raced          33928      54278    1.60  5.9e-16  
      4099  4099             -        raced         163527     440203    2.68  6.0e-16  
      6561  3^8              -        replayed       68378     146250    2.11  1.1e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           15      0      0   1.38   1.72   2.68    1.83
 ALL         15      0      0   1.38   1.72   2.68    1.83
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                3   1.49      0      0
 128..511               5   1.72      0      0
 512..2047              4   1.69      0      0
 2048..6561             3   2.11      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                               15   1.72      0   1.83
```


flip agreement: our two readings more than 25% apart at 0 of 15 cells.

worst 10: 1009 (- 1.33), 105 (- 1.38), 45 (- 1.49), 3465 (- 1.60), 225 (- 1.62), 1001 (- 1.65), 509 (- 1.68), 251 (- 1.72), 2003 (- 1.72), 315 (- 1.78)
best 5: 159 (- 3.41), 4099 (- 2.68), 97 (- 2.39), 6561 (- 2.11), 2025 (- 1.89)
