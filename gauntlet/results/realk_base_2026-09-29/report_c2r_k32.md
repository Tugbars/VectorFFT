# gauntlet report

run: `realk_base_2026-09-29`  contract file suffix: `_c2r_k32`  cells: 12 listed, 12 benched, comparator: MKL

control cell: 4 readings, 1.046..1.119 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        16  2^4              -        replayed         517        245    0.47  3.4e-16  
        32  2^5              -        replayed         685        570    0.83  4.4e-16  
        64  2^6              -        replayed        1008        980    0.95  5.0e-16  
        96  2^5.3            -        replayed        1672       1914    1.14  7.4e-16  
       128  2^7              -        replayed        2018       1833    0.91  6.7e-16  
       256  2^8              -        replayed        4612       4046    0.88  7.8e-16  
       512  2^9              -        replayed        7362       8226    1.11  7.8e-16  
      1000  2^3.5^3          -        replayed       25265      26564    1.00  1.1e-15  
      1024  2^10             -        replayed       18202      17143    0.89  7.8e-16  
      1215  3^5.5            -        replayed       77750      58067    0.74  1.5e-15  
      2048  2^11             -        replayed       39141      38743    0.91  8.3e-16  
      4096  2^12             -        replayed       82483      85367    1.01  7.8e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           12      2      8   0.74   0.91   1.11    0.88
 ALL         12      2      8   0.74   0.91   1.11    0.88
```


## by size
```
 band               cells median   <1.0   <0.8
 8..31                  1   0.47      1      1
 32..127                3   0.95      2      0
 128..511               2   0.89      2      0
 512..2047              4   0.94      2      1
 2048..4096             2   0.96      1      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             9   0.91      7   0.87
 -                                                3   1.00      1   0.94
```


flip agreement: our two readings more than 25% apart at 0 of 12 cells.

worst 10: 16 (- 0.47), 1215 (- 0.74), 32 (- 0.83), 256 (- 0.88), 1024 (- 0.89), 128 (- 0.91), 2048 (- 0.91), 64 (- 0.95), 1000 (- 1.00), 4096 (- 1.01)
best 5: 96 (- 1.14), 512 (- 1.11), 4096 (- 1.01), 1000 (- 1.00), 64 (- 0.95)
