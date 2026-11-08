# gauntlet report

run: `gnr_pow2_retime2_20261108`  contract file suffix: `(oop, T=1)`  cells: 6 listed, 6 benched, comparator: MKL

control cell: 4 readings, 1.201..1.215 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
       128  2^7              2p       replayed          81        107    1.31  1.6e-16  
       256  2^8              2p       replayed         160        283    1.36  2.5e-16  
       512  2^9              2p       replayed         366        478    1.24  3.5e-16  
      1024  2^10             ztt      replayed         914       1264    1.36  2.2e-16  
      2048  2^11             ztt      replayed        2020       2993    1.45  3.2e-16  
      4096  2^12             ztt      replayed        5093       6480    1.25  3.5e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 2p           3      0      0   1.24   1.31   1.36    1.30
 ztt          3      0      0   1.25   1.36   1.45    1.35
 ALL          6      0      0   1.24   1.34   1.45    1.33
```


## by size
```
 band               cells median   <1.0   <0.8
 128..511               2   1.34      0      0
 512..2047              2   1.30      0      0
 2048..4096             2   1.35      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             6   1.34      0   1.33
```


flip agreement: our two readings more than 25% apart at 0 of 6 cells.

worst 10: 512 (2p 1.24), 4096 (ztt 1.25), 128 (2p 1.31), 256 (2p 1.36), 1024 (ztt 1.36), 2048 (ztt 1.45)
best 5: 2048 (ztt 1.45), 1024 (ztt 1.36), 256 (2p 1.36), 128 (2p 1.31), 4096 (ztt 1.25)
