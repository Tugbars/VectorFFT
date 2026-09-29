# gauntlet report

run: `realk_base_2026-09-29`  contract file suffix: `_r2c_k32`  cells: 12 listed, 12 benched, comparator: MKL

control cell: 4 readings, 1.182..1.211 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        16  2^4              -        replayed         515        188    0.36  1.6e-16  
        32  2^5              -        replayed         696        552    0.77  2.0e-16  
        64  2^6              -        replayed        1054        883    0.84  1.6e-16  
        96  2^5.3            -        replayed        1559       1959    1.25  3.8e-16  
       128  2^7              -        replayed        2075       1882    0.72  3.1e-16  flips differ 1.26x
       256  2^8              -        replayed        3690       4275    1.08  3.6e-16  
       512  2^9              -        replayed        8129       9698    1.13  4.1e-16  
      1000  2^3.5^3          -        replayed       23813      29203    1.22  4.4e-16  
      1024  2^10             -        replayed       17118      19669    1.13  3.5e-16  
      1215  3^5.5            -        replayed       64476      59870    0.91  5.7e-16  
      2048  2^11             -        replayed       37913      45146    1.18  3.3e-16  
      4096  2^12             -        replayed       87027     104512    1.18  4.4e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           12      3      5   0.72   1.10   1.22    0.94
 ALL         12      3      5   0.72   1.10   1.22    0.94
```


## by size
```
 band               cells median   <1.0   <0.8
 8..31                  1   0.36      1      1
 32..127                3   0.84      2      1
 128..511               2   0.90      1      1
 512..2047              4   1.13      1      0
 2048..4096             2   1.18      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             9   1.08      4   0.88
 -                                                3   1.22      1   1.12
```


flip agreement: our two readings more than 25% apart at 1 of 12 cells.

worst 10: 16 (- 0.36), 128 (- 0.72), 32 (- 0.77), 64 (- 0.84), 1215 (- 0.91), 256 (- 1.08), 1024 (- 1.13), 512 (- 1.13), 4096 (- 1.18), 2048 (- 1.18)
best 5: 96 (- 1.25), 1000 (- 1.22), 2048 (- 1.18), 4096 (- 1.18), 512 (- 1.13)
