# gauntlet report

run: `gnr_pow2_retime3_20261108`  contract file suffix: `(oop, T=1)`  cells: 6 listed, 6 benched, comparator: MKL

control cell: 4 readings, 1.184..1.216 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
       128  2^7              2p       replayed          83        107    1.28  1.6e-16  
       256  2^8              2p       replayed         162        243    1.31  2.5e-16  
       512  2^9              2p       replayed         370        510    1.27  3.5e-16  
      1024  2^10             ztt      replayed         870       1245    1.39  2.2e-16  
      2048  2^11             ztt      replayed        2058       3120    1.51  3.2e-16  
      4096  2^12             ztt      replayed        5223       6318    1.19  3.5e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 2p           3      0      0   1.27   1.28   1.31    1.29
 ztt          3      0      0   1.19   1.39   1.51    1.35
 ALL          6      0      0   1.19   1.29   1.51    1.32
```


## by size
```
 band               cells median   <1.0   <0.8
 128..511               2   1.29      0      0
 512..2047              2   1.33      0      0
 2048..4096             2   1.35      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             6   1.29      0   1.32
```


flip agreement: our two readings more than 25% apart at 0 of 6 cells.

worst 10: 4096 (ztt 1.19), 512 (2p 1.27), 128 (2p 1.28), 256 (2p 1.31), 1024 (ztt 1.39), 2048 (ztt 1.51)
best 5: 2048 (ztt 1.51), 1024 (ztt 1.39), 256 (2p 1.31), 128 (2p 1.28), 512 (2p 1.27)
