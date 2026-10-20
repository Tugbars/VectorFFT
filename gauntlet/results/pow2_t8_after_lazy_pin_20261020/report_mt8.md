# gauntlet report

run: `pow2_t8_after_lazy_pin_20261020`  contract file suffix: `_mt8`  cells: 4 listed, 4 benched, comparator: MKL

control cell: 4 readings, 1.058..1.092 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        64  2^6              2p       replayed          36         31    0.83  1.5e-16  
       512  2^9              2p       replayed         303        293    0.92  2.3e-16  
      4096  2^12             ztt      replayed        3560       3855    1.07  3.9e-16  
      8192  2^13             ztt      replayed        6409       9132    1.31  3.4e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt          2      0      0   1.07   1.19   1.31    1.18
 2p           2      0      2   0.83   0.87   0.92    0.87
 ALL          4      0      2   0.83   0.99   1.31    1.02
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                1   0.83      1      0
 512..2047              1   0.92      1      0
 2048..8191             1   1.07      0      0
 8192..8192             1   1.31      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             4   0.99      2   1.02
```


flip agreement: our two readings more than 25% apart at 0 of 4 cells.

worst 10: 64 (2p 0.83), 512 (2p 0.92), 4096 (ztt 1.07), 8192 (ztt 1.31)
best 5: 8192 (ztt 1.31), 4096 (ztt 1.07), 512 (2p 0.92), 64 (2p 0.83)
