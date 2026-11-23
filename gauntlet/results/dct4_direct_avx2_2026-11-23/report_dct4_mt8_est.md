# gauntlet report

run: `dct4_direct_avx2_2026-11-23`  contract file suffix: `_dct4_mt8_est`  cells: 5 listed, 5 benched, comparator: MKL

control cell: 4 readings, 2.182..2.241 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
     65536  2^16             -        -              69085     135675    1.95  8.2e-08  
    100000  2^5.5^5          -        -             110269     252431    2.20  6.1e-08  
    262144  2^18             -        -             280927     706820    2.51  3.8e-06  
   1048576  2^20             -        -            1663950    3430781    2.04  5.6e-05  
   4194304  2^22             -        -            9048463   19749719    1.99  2.5e-03  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            5      0      0   1.95   2.04   2.51    2.13
 ALL          5      0      0   1.95   2.04   2.51    2.13
```


## by size
```
 band               cells median   <1.0   <0.8
 32768..131071          2   2.08      0      0
 131072..524287         1   2.51      0      0
 524288..2097151        1   2.04      0      0
 2097152..4194304       1   1.99      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             4   2.02      0   2.11
 -                                                1   2.20      0   2.20
```


flip agreement: our two readings more than 25% apart at 0 of 5 cells.

worst 10: 65536 (- 1.95), 4194304 (- 1.99), 1048576 (- 2.04), 100000 (- 2.20), 262144 (- 2.51)
best 5: 262144 (- 2.51), 100000 (- 2.20), 1048576 (- 2.04), 4194304 (- 1.99), 65536 (- 1.95)
