# gauntlet report

run: `gnr_pow2_retime1_20261108`  contract file suffix: `(oop, T=1)`  cells: 6 listed, 6 benched, comparator: MKL

control cell: 4 readings, 1.188..1.278 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
       128  2^7              2p       replayed          82        127    1.28  1.6e-16  
       256  2^8              2p       replayed         163        217    1.32  2.5e-16  
       512  2^9              2p       replayed         378        485    1.27  3.5e-16  
      1024  2^10             ztt      replayed         925       1254    1.34  2.2e-16  
      2048  2^11             ztt      replayed        2116       4261    1.90  3.2e-16  
      4096  2^12             ztt      replayed        5128       6311    1.22  3.5e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 2p           3      0      0   1.27   1.28   1.32    1.29
 ztt          3      0      0   1.22   1.34   1.90    1.46
 ALL          6      0      0   1.22   1.30   1.90    1.37
```


## by size
```
 band               cells median   <1.0   <0.8
 128..511               2   1.30      0      0
 512..2047              2   1.30      0      0
 2048..4096             2   1.56      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             6   1.30      0   1.37
```


flip agreement: our two readings more than 25% apart at 0 of 6 cells.

worst 10: 4096 (ztt 1.22), 512 (2p 1.27), 128 (2p 1.28), 256 (2p 1.32), 1024 (ztt 1.34), 2048 (ztt 1.90)
best 5: 2048 (ztt 1.90), 1024 (ztt 1.34), 256 (2p 1.32), 128 (2p 1.28), 512 (2p 1.27)
