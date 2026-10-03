# gauntlet report

run: `tcbmt3_r2c_k64_t8_2026-10-03`  contract file suffix: `_r2c_k64_mt8`  cells: 15 listed, 15 benched, comparator: MKL

control cell: 4 readings, 1.035..1.452 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        45  3^2.5            -        raced            625        935    1.46  4.3e-16  
        97  97               -        raced           2453       7183    2.45  6.0e-16  
       105  3.5.7            -        raced            979       1522    1.55  3.9e-16  
       159  3.53             -        raced           3647      12970    3.56  5.4e-16  
       225  3^2.5^2          -        raced           1605       2903    1.81  5.2e-16  
       251  251              -        raced           6004      11546    1.67  7.1e-16  
       315  3^2.5.7          -        raced           2028       3702    1.77  7.5e-16  
       509  509              -        raced          13987      24574    1.72  8.5e-16  
      1001  7.11.13          -        raced           7231      15023    2.08  1.0e-15  
      1009  1009             -        raced          42449      57720    1.36  5.7e-16  
      2003  2003             -        raced          64923     115092    1.71  6.6e-16  
      2025  3^4.5^2          -        raced          15010      32638    2.12  6.4e-16  
      3465  3^2.5.7.11       -        raced          27456      64239    2.34  5.9e-16  
      4099  4099             -        raced         159947     451913    2.77  6.0e-16  
      6561  3^8              -        raced          56833     145866    2.55  1.1e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           15      0      0   1.46   1.81   2.77    1.99
 ALL         15      0      0   1.46   1.81   2.77    1.99
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                3   1.55      0      0
 128..511               5   1.77      0      0
 512..2047              4   1.89      0      0
 2048..6561             3   2.55      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                               15   1.81      0   1.99
```


flip agreement: our two readings more than 25% apart at 0 of 15 cells.

worst 10: 1009 (- 1.36), 45 (- 1.46), 105 (- 1.55), 251 (- 1.67), 2003 (- 1.71), 509 (- 1.72), 315 (- 1.77), 225 (- 1.81), 1001 (- 2.08), 2025 (- 2.12)
best 5: 159 (- 3.56), 4099 (- 2.77), 6561 (- 2.55), 97 (- 2.45), 3465 (- 2.34)
