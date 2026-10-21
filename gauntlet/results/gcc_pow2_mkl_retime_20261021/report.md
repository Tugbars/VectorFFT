# gauntlet report

run: `gcc_pow2_mkl_retime_20261021`  contract file suffix: `(oop, T=1)`  cells: 3 listed, 3 benched, comparator: MKL

control cell: 4 readings, 1.076..1.079 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
       128  2^7              2p       replayed          64         68    1.07  3.2e-16  
      2048  2^11             ztt      replayed        1605       2060    1.26  3.1e-16  
     32768  2^15             ztt      replayed       39313      36953    0.58  3.4e-16  flips differ 1.62x
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt          2      1      1   0.58   0.92   1.26    0.85
 2p           1      0      0   1.07   1.07   1.07    1.07
 ALL          3      1      1   0.58   1.07   1.26    0.92
```


## by size
```
 band               cells median   <1.0   <0.8
 128..511               1   1.07      0      0
 2048..8191             1   1.26      0      0
 32768..32768           1   0.58      1      1
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             3   1.07      1   0.92
```


flip agreement: our two readings more than 25% apart at 1 of 3 cells.

worst 10: 32768 (ztt 0.58), 128 (2p 1.07), 2048 (ztt 1.26)
best 5: 2048 (ztt 1.26), 128 (2p 1.07), 32768 (ztt 0.58)
