# gauntlet report

run: `dct4_direct_avx2_2026-11-23`  contract file suffix: `_dct4_est`  cells: 10 listed, 10 benched, comparator: MKL

control cell: 4 readings, 2.214..2.243 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        64  2^6              -        -                 40        118    2.96  8.0e-15  
      1000  2^3.5^3          -        -                850       1919    2.25  4.8e-12  
      1024  2^10             -        -                690       1625    2.35  3.8e-12  
      4096  2^12             -        -               3157       7085    2.24  7.7e-11  
     10000  2^4.5^4          -        -              10859      22697    2.09  8.2e-10  
     10007  10007            -        -              62564     122319    1.95  6.0e-10  
     65536  2^16             -        -              74669     130097    1.74  8.2e-08  
    100000  2^5.5^5          -        -             135297     249011    1.83  6.1e-08  
    100003  100003           -        -            1042462    1884715    1.79  4.4e-07  
   1048576  2^20             -        -            3206125    3343344    1.04  5.6e-05  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           10      0      0   1.74   2.02   2.96    1.96
 ALL         10      0      0   1.74   2.02   2.96    1.96
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                1   2.96      0      0
 512..2047              2   2.30      0      0
 2048..8191             1   2.24      0      0
 8192..32767            2   2.02      0      0
 32768..131071          3   1.79      0      0
 524288..1048576        1   1.04      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             5   2.24      0   1.95
 -                                                5   1.95      0   1.97
```


flip agreement: our two readings more than 25% apart at 0 of 10 cells.

worst 10: 1048576 (- 1.04), 65536 (- 1.74), 100003 (- 1.79), 100000 (- 1.83), 10007 (- 1.95), 10000 (- 2.09), 4096 (- 2.24), 1000 (- 2.25), 1024 (- 2.35), 64 (- 2.96)
best 5: 64 (- 2.96), 1024 (- 2.35), 1000 (- 2.25), 4096 (- 2.24), 10000 (- 2.09)
