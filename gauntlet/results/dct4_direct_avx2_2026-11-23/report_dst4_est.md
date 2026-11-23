# gauntlet report

run: `dct4_direct_avx2_2026-11-23`  contract file suffix: `_dst4_est`  cells: 10 listed, 10 benched, comparator: MKL

control cell: 4 readings, 2.318..2.357 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        64  2^6              -        -                 40        123    3.06  3.7e-14  
      1000  2^3.5^3          -        -                845       2013    2.36  1.3e-11  
      1024  2^10             -        -                689       1716    2.48  2.1e-11  
      4096  2^12             -        -               3176       7453    2.34  2.2e-10  
     10000  2^4.5^4          -        -              10831      23918    2.14  5.8e-09  
     10007  10007            -        -              64740     123273    1.90  6.7e-09  
     65536  2^16             -        -              74934     135649    1.81  8.0e-08  
    100000  2^5.5^5          -        -             135338     258770    1.91  3.1e-06  
    100003  100003           -        -            1058305    1896222    1.78  5.3e-07  
   1048576  2^20             -        -            3218575    3489331    1.08  1.3e-03  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           10      0      0   1.78   2.03   3.06    2.02
 ALL         10      0      0   1.78   2.03   3.06    2.02
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                1   3.06      0      0
 512..2047              2   2.42      0      0
 2048..8191             1   2.34      0      0
 8192..32767            2   2.02      0      0
 32768..131071          3   1.81      0      0
 524288..1048576        1   1.08      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             5   2.34      0   2.03
 -                                                5   1.91      0   2.01
```


flip agreement: our two readings more than 25% apart at 0 of 10 cells.

worst 10: 1048576 (- 1.08), 100003 (- 1.78), 65536 (- 1.81), 10007 (- 1.90), 100000 (- 1.91), 10000 (- 2.14), 4096 (- 2.34), 1000 (- 2.36), 1024 (- 2.48), 64 (- 3.06)
best 5: 64 (- 3.06), 1024 (- 2.48), 1000 (- 2.36), 4096 (- 2.34), 10000 (- 2.14)
