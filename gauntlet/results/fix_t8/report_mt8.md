# gauntlet report

run: `fix_t8`  contract file suffix: `_mt8`  cells: 4 listed, 4 benched, comparator: MKL

control cell: 4 readings, 1.015..1.073 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      8192  2^13             ztt      replayed        5061       8210    1.49  3.4e-16  
     16384  2^14             ztt      replayed       10918      14820    1.13  3.6e-16  
     65536  2^16             ztt      replayed       25193      40271    1.53  4.0e-16  
    262144  2^18             ztt      replayed      105687     171537    1.59  4.0e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt          4      0      0   1.13   1.51   1.59    1.43
 ALL          4      0      0   1.13   1.51   1.59    1.43
```


## by size
```
 band               cells median   <1.0   <0.8
 8192..32767            2   1.31      0      0
 32768..131071          1   1.53      0      0
 131072..262144         1   1.59      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             4   1.51      0   1.43
```


flip agreement: our two readings more than 25% apart at 0 of 4 cells.

worst 10: 16384 (ztt 1.13), 8192 (ztt 1.49), 65536 (ztt 1.53), 262144 (ztt 1.59)
best 5: 262144 (ztt 1.59), 65536 (ztt 1.53), 8192 (ztt 1.49), 16384 (ztt 1.13)
