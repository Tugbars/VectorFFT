# gauntlet report

run: `gnr_pow2_128_4096_mkl_20261108`  contract file suffix: `(oop, T=1)`  cells: 6 listed, 6 benched, comparator: MKL

control cell: 4 readings, 1.195..1.238 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
       128  2^7              2p       raced             83        106    0.96  1.6e-16  flips differ 1.31x
       256  2^8              2p       raced            166        258    1.55  2.5e-16  
       512  2^9              2p       raced            374        483    1.28  3.5e-16  
      1024  2^10             ztt      raced            906       1262    1.37  2.2e-16  
      2048  2^11             ztt      raced           2063       3019    1.40  3.2e-16  
      4096  2^12             ztt      raced           5188       6365    1.20  3.5e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt          3      0      0   1.20   1.37   1.40    1.32
 2p           3      0      1   0.96   1.28   1.55    1.24
 ALL          6      0      1   0.96   1.33   1.55    1.28
```


## by size
```
 band               cells median   <1.0   <0.8
 128..511               2   1.25      1      0
 512..2047              2   1.33      0      0
 2048..4096             2   1.30      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             6   1.33      1   1.28
```


flip agreement: our two readings more than 25% apart at 1 of 6 cells.

worst 10: 128 (2p 0.96), 4096 (ztt 1.20), 512 (2p 1.28), 1024 (ztt 1.37), 2048 (ztt 1.40), 256 (2p 1.55)
best 5: 256 (2p 1.55), 2048 (ztt 1.40), 1024 (ztt 1.37), 512 (2p 1.28), 4096 (ztt 1.20)
