# gauntlet report

run: `pow2fs_child_2026-10-04`  contract file suffix: `(oop, T=1)`  cells: 5 listed, 5 benched, comparator: MKL

control cell: 4 readings, 1.074..1.085 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
    262144  2^18             ztt      raced         552663     712731    1.28  4.0e-16  
    524288  2^19             fs       raced        1355663    1523094    1.08  6.8e-16  
   1048576  2^20             fs       raced        3117800    4160994    1.32  5.3e-16  
   2097152  2^21             fs       raced        7955787   10659668    1.31  7.4e-16  
   4194304  2^22             fs       raced       18325550   25601756    1.36  9.7e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 fs           4      0      0   1.08   1.32   1.36    1.27
 ztt          1      0      0   1.28   1.28   1.28    1.28
 ALL          5      0      0   1.08   1.31   1.36    1.27
```


## by size
```
 band               cells median   <1.0   <0.8
 131072..524287         1   1.28      0      0
 524288..2097151        2   1.20      0      0
 2097152..4194304       2   1.34      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             5   1.31      0   1.27
```


flip agreement: our two readings more than 25% apart at 0 of 5 cells.

worst 10: 524288 (fs 1.08), 262144 (ztt 1.28), 2097152 (fs 1.31), 1048576 (fs 1.32), 4194304 (fs 1.36)
best 5: 4194304 (fs 1.36), 1048576 (fs 1.32), 2097152 (fs 1.31), 262144 (ztt 1.28), 524288 (fs 1.08)
