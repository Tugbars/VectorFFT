# gauntlet report

run: `dct_first_look_avx2_2026-11-23`  contract file suffix: `_dct4_est`  cells: 8 listed, 8 benched, comparator: MKL

control cell: 4 readings, 1.537..1.567 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        64  2^6              -        -                 60        118    1.93  8.1e-15  
      1000  2^3.5^3          -        -               1153       1923    1.63  4.8e-12  
      1024  2^10             -        -                986       1625    1.65  3.8e-12  
      4096  2^12             -        -               4581       7146    1.56  7.7e-11  
     10007  10007            -        -              62355     124413    1.99  6.0e-10  
     65536  2^16             -        -             105849     131305    1.23  8.2e-08  
    100003  100003           -        -            1139208    1970316    1.72  4.4e-07  
   1048576  2^20             -        -            4876725    3595088    0.73  5.6e-05  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            8      1      1   0.73   1.64   1.99    1.49
 ALL          8      1      1   0.73   1.64   1.99    1.49
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                1   1.93      0      0
 512..2047              2   1.64      0      0
 2048..8191             1   1.56      0      0
 8192..32767            1   1.99      0      0
 32768..131071          2   1.47      0      0
 524288..1048576        1   0.73      1      1
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             5   1.56      1   1.35
 -                                                3   1.72      0   1.77
```


flip agreement: our two readings more than 25% apart at 0 of 8 cells.

worst 10: 1048576 (- 0.73), 65536 (- 1.23), 4096 (- 1.56), 1000 (- 1.63), 1024 (- 1.65), 100003 (- 1.72), 64 (- 1.93), 10007 (- 1.99)
best 5: 10007 (- 1.99), 64 (- 1.93), 100003 (- 1.72), 1024 (- 1.65), 1000 (- 1.63)
