# gauntlet report

run: `head_t8_vs_sept`  contract file suffix: `_mt8`  cells: 4 listed, 4 benched, comparator: MKL

control cell: 4 readings, 1.026..1.053 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      4096  2^12             ztt      replayed        3556       3751    1.02  3.9e-16  
      8192  2^13             ztt      replayed        6023       8288    1.32  3.4e-16  
     16384  2^14             ztt      replayed       11170      14298    1.03  3.6e-16  
     65536  2^16             ztt      replayed       27673      40370    1.38  4.0e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt          4      0      0   1.02   1.18   1.38    1.18
 ALL          4      0      0   1.02   1.18   1.38    1.18
```


## by size
```
 band               cells median   <1.0   <0.8
 2048..8191             1   1.02      0      0
 8192..32767            2   1.18      0      0
 32768..65536           1   1.38      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             4   1.18      0   1.18
```


flip agreement: our two readings more than 25% apart at 0 of 4 cells.

worst 10: 4096 (ztt 1.02), 16384 (ztt 1.03), 8192 (ztt 1.32), 65536 (ztt 1.38)
best 5: 65536 (ztt 1.38), 8192 (ztt 1.32), 16384 (ztt 1.03), 4096 (ztt 1.02)
