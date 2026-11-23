# gauntlet report

run: `dct4_direct_avx2_2026-11-23`  contract file suffix: `_dst4_mt8_est`  cells: 5 listed, 5 benched, comparator: MKL

control cell: 4 readings, 2.289..2.363 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
     65536  2^16             -        -              68525     140758    2.02  8.0e-08  
    100000  2^5.5^5          -        -             110359     264525    2.36  3.1e-06  
    262144  2^18             -        -             280227     732193    2.60  1.7e-05  
   1048576  2^20             -        -            1679113    3593556    2.13  1.3e-03  
   4194304  2^22             -        -            8580950   20718162    2.40  1.1e-02  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            5      0      0   2.02   2.36   2.60    2.29
 ALL          5      0      0   2.02   2.36   2.60    2.29
```


## by size
```
 band               cells median   <1.0   <0.8
 32768..131071          2   2.19      0      0
 131072..524287         1   2.60      0      0
 524288..2097151        1   2.13      0      0
 2097152..4194304       1   2.40      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             4   2.26      0   2.28
 -                                                1   2.36      0   2.36
```


flip agreement: our two readings more than 25% apart at 0 of 5 cells.

worst 10: 65536 (- 2.02), 1048576 (- 2.13), 100000 (- 2.36), 4194304 (- 2.40), 262144 (- 2.60)
best 5: 262144 (- 2.60), 4194304 (- 2.40), 100000 (- 2.36), 1048576 (- 2.13), 65536 (- 2.02)
