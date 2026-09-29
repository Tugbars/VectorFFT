# gauntlet report

run: `realk_base_2026-09-29`  contract file suffix: `_c2r_k8`  cells: 12 listed, 12 benched, comparator: MKL

control cell: 4 readings, 1.078..1.086 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        16  2^4              -        replayed         139         65    0.47  3.4e-16  
        32  2^5              -        replayed         180        146    0.79  3.3e-16  
        64  2^6              -        replayed         260        250    0.96  5.0e-16  
        96  2^5.3            -        replayed         382        481    1.26  6.3e-16  
       128  2^7              -        replayed         449        448    0.99  4.4e-16  
       256  2^8              -        replayed        1003       1005    0.99  6.1e-16  
       512  2^9              -        replayed        1844       2068    1.12  7.8e-16  
      1000  2^3.5^3          -        replayed        6265       6663    1.04  9.4e-16  
      1024  2^10             -        replayed        4751       4300    0.85  7.8e-16  
      1215  3^5.5            -        replayed       19358      14706    0.74  1.4e-15  
      2048  2^11             -        replayed        9751       9809    0.99  7.8e-16  
      4096  2^12             -        replayed       19645      21533    1.09  7.8e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           12      3      8   0.74   0.99   1.12    0.92
 ALL         12      3      8   0.74   0.99   1.12    0.92
```


## by size
```
 band               cells median   <1.0   <0.8
 8..31                  1   0.47      1      1
 32..127                3   0.96      2      1
 128..511               2   0.99      2      0
 512..2047              4   0.95      2      1
 2048..4096             2   1.04      1      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             9   0.99      7   0.89
 -                                                3   1.04      1   0.99
```


flip agreement: our two readings more than 25% apart at 0 of 12 cells.

worst 10: 16 (- 0.47), 1215 (- 0.74), 32 (- 0.79), 1024 (- 0.85), 64 (- 0.96), 128 (- 0.99), 256 (- 0.99), 2048 (- 0.99), 1000 (- 1.04), 4096 (- 1.09)
best 5: 96 (- 1.26), 512 (- 1.12), 4096 (- 1.09), 1000 (- 1.04), 256 (- 0.99)
