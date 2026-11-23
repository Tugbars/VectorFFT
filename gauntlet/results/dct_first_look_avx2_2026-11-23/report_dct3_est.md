# gauntlet report

run: `dct_first_look_avx2_2026-11-23`  contract file suffix: `_dct3_est`  cells: 8 listed, 8 benched, comparator: MKL

control cell: 4 readings, 1.718..1.727 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        64  2^6              -        -                 43         96    2.25  2.9e-15  
      1000  2^3.5^3          -        -               1009       1584    1.55  1.2e-13  
      1024  2^10             -        -                755       1301    1.70  3.8e-14  
      4096  2^12             -        -               3289       5655    1.72  3.2e-12  
     10007  10007            -        -              54910     132882    2.21  9.6e-11  
     65536  2^16             -        -              81805     108109    1.26  1.0e-09  
    100003  100003           -        -            1089995    1881550    1.66  1.8e-10  
   1048576  2^20             -        -            3321612    2882462    0.86  9.1e-08  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            8      0      1   0.86   1.68   2.25    1.59
 ALL          8      0      1   0.86   1.68   2.25    1.59
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                1   2.25      0      0
 512..2047              2   1.63      0      0
 2048..8191             1   1.72      0      0
 8192..32767            1   2.21      0      0
 32768..131071          2   1.46      0      0
 524288..1048576        1   0.86      1      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             5   1.70      1   1.48
 -                                                3   1.66      0   1.78
```


flip agreement: our two readings more than 25% apart at 0 of 8 cells.

worst 10: 1048576 (- 0.86), 65536 (- 1.26), 1000 (- 1.55), 100003 (- 1.66), 1024 (- 1.70), 4096 (- 1.72), 10007 (- 2.21), 64 (- 2.25)
best 5: 64 (- 2.25), 10007 (- 2.21), 4096 (- 1.72), 1024 (- 1.70), 100003 (- 1.66)
