# gauntlet report

run: `pow2fs_child_t8_2026-10-04`  contract file suffix: `_mt8`  cells: 5 listed, 5 benched, comparator: MKL

control cell: 6 readings, 0.623..1.058 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
    262144  2^18             ztt      raced         106425     163662    1.47  4.0e-16  
    524288  2^19             fs       raced         279975     359350    1.25  4.3e-16  
   1048576  2^20             fs       raced         607950     891500    1.41  1.1e-15  
   2097152  2^21             fs       raced        1797738    2731087    1.48  2.7e-15  
   4194304  2^22             fs       raced        6471537    7192844    1.09  9.2e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 fs           4      0      0   1.09   1.33   1.48    1.30
 ztt          1      0      0   1.47   1.47   1.47    1.47
 ALL          5      0      0   1.09   1.41   1.48    1.33
```


## by size
```
 band               cells median   <1.0   <0.8
 131072..524287         1   1.47      0      0
 524288..2097151        2   1.33      0      0
 2097152..4194304       2   1.29      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             5   1.41      0   1.33
```


flip agreement: our two readings more than 25% apart at 0 of 5 cells.

worst 10: 4194304 (fs 1.09), 524288 (fs 1.25), 1048576 (fs 1.41), 262144 (ztt 1.47), 2097152 (fs 1.48)
best 5: 2097152 (fs 1.48), 262144 (ztt 1.47), 1048576 (fs 1.41), 524288 (fs 1.25), 4194304 (fs 1.09)
