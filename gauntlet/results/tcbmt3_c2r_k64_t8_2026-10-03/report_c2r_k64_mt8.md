# gauntlet report

run: `tcbmt3_c2r_k64_t8_2026-10-03`  contract file suffix: `_c2r_k64_mt8`  cells: 15 listed, 15 benched, comparator: MKL

control cell: 4 readings, 1.313..1.359 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        45  3^2.5            -        replayed         590        999    1.69  1.1e-15  
        97  97               -        replayed        2500       5926    2.35  1.3e-15  
       105  3.5.7            -        raced            902       1488    1.60  1.1e-15  
       159  3.53             -        raced           3705      12906    3.44  1.1e-15  
       225  3^2.5^2          -        raced           1669       2555    1.47  1.2e-15  
       251  251              -        raced           6120      10766    1.72  1.8e-15  
       315  3^2.5.7          -        raced           2144       3584    1.43  1.4e-15  
       509  509              -        raced          14174      23787    1.59  1.9e-15  
      1001  7.11.13          -        raced           7895      15183    1.84  1.8e-15  
      1009  1009             -        raced          39270      56104    1.39  1.4e-15  
      2003  2003             -        raced          64210     112713    1.74  1.6e-15  
      2025  3^4.5^2          -        raced          13677      34046    2.06  1.8e-15  
      3465  3^2.5.7.11       -        raced          31561      55967    1.77  1.4e-15  
      4099  4099             -        raced         164200     449147    2.73  1.8e-15  
      6561  3^8              -        raced          69867     147689    2.11  2.2e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           15      0      0   1.43   1.74   2.73    1.87
 ALL         15      0      0   1.43   1.74   2.73    1.87
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                3   1.69      0      0
 128..511               5   1.59      0      0
 512..2047              4   1.79      0      0
 2048..6561             3   2.11      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                               15   1.74      0   1.87
```


flip agreement: our two readings more than 25% apart at 0 of 15 cells.

worst 10: 1009 (- 1.39), 315 (- 1.43), 225 (- 1.47), 509 (- 1.59), 105 (- 1.60), 45 (- 1.69), 251 (- 1.72), 2003 (- 1.74), 3465 (- 1.77), 1001 (- 1.84)
best 5: 159 (- 3.44), 4099 (- 2.73), 97 (- 2.35), 6561 (- 2.11), 2025 (- 2.06)
