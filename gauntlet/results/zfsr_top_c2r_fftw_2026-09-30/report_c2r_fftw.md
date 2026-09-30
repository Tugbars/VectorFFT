# gauntlet report

run: `zfsr_top_c2r_fftw_2026-09-30`  contract file suffix: `_c2r_fftw`  cells: 5 listed, 5 benched, comparator: FFTW

control cell: 4 readings, 1.081..1.087 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
    524288  2^19             -        replayed      731400     910056    1.21  1.4e-15  
   1048576  2^20             -        replayed     1927287    2183250    1.12  1.4e-15  
   2097152  2^21             -        replayed     4874600    5868144    1.18  1.4e-15  
   4194304  2^22             -        replayed    12073862   14758587    1.22  1.6e-15  
   8388608  2^23             -        replayed    26184313   32588518    1.21  1.5e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            5      0      0   1.12   1.21   1.22    1.19
 ALL          5      0      0   1.12   1.21   1.22    1.19
```


## by size
```
 band               cells median   <1.0   <0.8
 524288..2097151        2   1.17      0      0
 2097152..8388607       2   1.20      0      0
 8388608..8388608       1   1.21      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             5   1.21      0   1.19
```


flip agreement: our two readings more than 25% apart at 0 of 5 cells.

worst 10: 1048576 (- 1.12), 2097152 (- 1.18), 524288 (- 1.21), 8388608 (- 1.21), 4194304 (- 1.22)
best 5: 4194304 (- 1.22), 8388608 (- 1.21), 524288 (- 1.21), 2097152 (- 1.18), 1048576 (- 1.12)
