# gauntlet report

run: `dct_first_look_avx2_2026-11-23`  contract file suffix: `_dst4_est`  cells: 8 listed, 8 benched, comparator: MKL

control cell: 4 readings, 1.632..1.641 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        64  2^6              -        -                 60        124    2.01  3.7e-14  
      1000  2^3.5^3          -        -               1120       2009    1.78  1.3e-11  
      1024  2^10             -        -                984       1718    1.74  2.1e-11  
      4096  2^12             -        -               4561       7503    1.64  2.2e-10  
     10007  10007            -        -              64437     125853    1.92  6.7e-09  
     65536  2^16             -        -             101218     137091    1.35  8.0e-08  
    100003  100003           -        -            1116903    1912535    1.66  5.3e-07  
   1048576  2^20             -        -            4682638    3480325    0.74  1.3e-03  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            8      1      1   0.74   1.70   2.01    1.54
 ALL          8      1      1   0.74   1.70   2.01    1.54
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                1   2.01      0      0
 512..2047              2   1.76      0      0
 2048..8191             1   1.64      0      0
 8192..32767            1   1.92      0      0
 32768..131071          2   1.50      0      0
 524288..1048576        1   0.74      1      1
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             5   1.64      1   1.42
 -                                                3   1.78      0   1.78
```


flip agreement: our two readings more than 25% apart at 0 of 8 cells.

worst 10: 1048576 (- 0.74), 65536 (- 1.35), 4096 (- 1.64), 100003 (- 1.66), 1024 (- 1.74), 1000 (- 1.78), 10007 (- 1.92), 64 (- 2.01)
best 5: 64 (- 2.01), 10007 (- 1.92), 1000 (- 1.78), 1024 (- 1.74), 100003 (- 1.66)
