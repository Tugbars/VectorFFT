# gauntlet report

run: `dct_first_look_avx2_2026-11-23`  contract file suffix: `_dct2_est`  cells: 8 listed, 8 benched, comparator: MKL

control cell: 4 readings, 1.465..1.478 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        64  2^6              -        -                 44         95    2.10  5.3e-16  
      1000  2^3.5^3          -        -                793       1581    1.96  3.6e-14  
      1024  2^10             -        -                704       1288    1.82  3.5e-14  
      4096  2^12             -        -               3832       5634    1.46  2.8e-13  
     10007  10007            -        -              60587     125919    2.03  1.9e-12  
     65536  2^16             -        -              78639     103894    1.28  6.7e-12  
    100003  100003           -        -            1054223    1911443    1.74  1.3e-11  
   1048576  2^20             -        -            2767825    2889312    1.04  1.6e-09  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            8      0      0   1.04   1.78   2.10    1.64
 ALL          8      0      0   1.04   1.78   2.10    1.64
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                1   2.10      0      0
 512..2047              2   1.89      0      0
 2048..8191             1   1.46      0      0
 8192..32767            1   2.03      0      0
 32768..131071          2   1.51      0      0
 524288..1048576        1   1.04      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             5   1.46      0   1.49
 -                                                3   1.96      0   1.91
```


flip agreement: our two readings more than 25% apart at 0 of 8 cells.

worst 10: 1048576 (- 1.04), 65536 (- 1.28), 4096 (- 1.46), 100003 (- 1.74), 1024 (- 1.82), 1000 (- 1.96), 10007 (- 2.03), 64 (- 2.10)
best 5: 64 (- 2.10), 10007 (- 2.03), 1000 (- 1.96), 1024 (- 1.82), 100003 (- 1.74)
