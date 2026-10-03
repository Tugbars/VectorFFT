# gauntlet report

run: `tcbmt4_r2c_k64_t8_quiet_2026-10-03`  contract file suffix: `_r2c_k64_mt8`  cells: 15 listed, 15 benched, comparator: MKL

control cell: 4 readings, 1.154..1.495 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        45  3^2.5            -        raced            632        971    1.51  4.3e-16  
        97  97               -        raced           2671       6481    2.23  6.0e-16  
       105  3.5.7            -        raced            970       1433    1.44  3.9e-16  
       159  3.53             -        raced           3611      12590    3.49  5.4e-16  
       225  3^2.5^2          -        raced           1743       2650    1.51  9.1e-16  
       251  251              -        raced           5965      11168    1.83  7.1e-16  
       315  3^2.5.7          -        raced           2324       3902    1.68  7.5e-16  
       509  509              -        raced          15245      26726    1.74  8.5e-16  
      1001  7.11.13          -        raced           7297      14952    1.78  8.1e-16  
      1009  1009             -        raced          39664      52140    1.14  5.7e-16  
      2003  2003             -        raced          62403     116698    1.69  6.6e-16  
      2025  3^4.5^2          -        raced          15463      28697    1.84  6.8e-16  
      3465  3^2.5.7.11       -        raced          28789      64111    2.23  6.5e-16  
      4099  4099             -        raced         166500     445667    2.68  7.3e-16  
      6561  3^8              -        raced          70889     118978    1.68  1.1e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           15      0      0   1.44   1.74   2.68    1.83
 ALL         15      0      0   1.44   1.74   2.68    1.83
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                3   1.51      0      0
 128..511               5   1.74      0      0
 512..2047              4   1.74      0      0
 2048..6561             3   2.23      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                               15   1.74      0   1.83
```


flip agreement: our two readings more than 25% apart at 0 of 15 cells.

worst 10: 1009 (- 1.14), 105 (- 1.44), 225 (- 1.51), 45 (- 1.51), 6561 (- 1.68), 315 (- 1.68), 2003 (- 1.69), 509 (- 1.74), 1001 (- 1.78), 251 (- 1.83)
best 5: 159 (- 3.49), 4099 (- 2.68), 3465 (- 2.23), 97 (- 2.23), 2025 (- 1.84)
