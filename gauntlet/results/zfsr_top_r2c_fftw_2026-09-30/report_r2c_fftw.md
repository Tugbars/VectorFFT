# gauntlet report

run: `zfsr_top_r2c_fftw_2026-09-30`  contract file suffix: `_r2c_fftw`  cells: 5 listed, 5 benched, comparator: FFTW

control cell: 4 readings, 1.170..1.178 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
    524288  2^19             -        replayed      606937     849769    1.38  4.4e-16  
   1048576  2^20             -        replayed     1487200    2139987    1.39  5.6e-16  
   2097152  2^21             -        replayed     3380238    5950312    1.74  1.6e-15  
   4194304  2^22             -        replayed     8698250   14709931    1.68  3.1e-15  
   8388608  2^23             -        replayed    21853412   34760487    1.47  1.1e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            5      0      0   1.38   1.47   1.74    1.53
 ALL          5      0      0   1.38   1.47   1.74    1.53
```


## by size
```
 band               cells median   <1.0   <0.8
 524288..2097151        2   1.39      0      0
 2097152..8388607       2   1.71      0      0
 8388608..8388608       1   1.47      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             5   1.47      0   1.53
```


flip agreement: our two readings more than 25% apart at 0 of 5 cells.

worst 10: 524288 (- 1.38), 1048576 (- 1.39), 8388608 (- 1.47), 4194304 (- 1.68), 2097152 (- 1.74)
best 5: 2097152 (- 1.74), 4194304 (- 1.68), 8388608 (- 1.47), 1048576 (- 1.39), 524288 (- 1.38)
