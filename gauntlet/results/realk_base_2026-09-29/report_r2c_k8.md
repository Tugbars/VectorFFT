# gauntlet report

run: `realk_base_2026-09-29`  contract file suffix: `_r2c_k8`  cells: 12 listed, 12 benched, comparator: MKL

control cell: 4 readings, 1.182..1.197 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        16  2^4              -        replayed         138         50    0.36  1.8e-16  
        32  2^5              -        replayed         178        135    0.76  1.2e-16  
        64  2^6              -        replayed         266        217    0.75  1.6e-16  
        96  2^5.3            -        replayed         387        476    1.23  3.2e-16  
       128  2^7              -        replayed         481        430    0.89  2.6e-16  
       256  2^8              -        replayed         831       1020    1.18  3.8e-16  
       512  2^9              -        replayed        2070       2516    1.21  4.1e-16  
      1000  2^3.5^3          -        replayed        5920       7251    1.22  4.6e-16  
      1024  2^10             -        replayed        4356       4907    1.00  4.2e-16  
      1215  3^5.5            -        replayed       15520      15027    0.92  5.7e-16  
      2048  2^11             -        replayed        9483      11230    1.17  4.0e-16  
      4096  2^12             -        replayed       20956      24886    1.18  4.4e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           12      3      6   0.75   1.08   1.22    0.94
 ALL         12      3      6   0.75   1.08   1.22    0.94
```


## by size
```
 band               cells median   <1.0   <0.8
 8..31                  1   0.36      1      1
 32..127                3   0.76      2      2
 128..511               2   1.03      1      0
 512..2047              4   1.10      2      0
 2048..4096             2   1.18      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             9   1.00      5   0.89
 -                                                3   1.22      1   1.11
```


flip agreement: our two readings more than 25% apart at 0 of 12 cells.

worst 10: 16 (- 0.36), 64 (- 0.75), 32 (- 0.76), 128 (- 0.89), 1215 (- 0.92), 1024 (- 1.00), 2048 (- 1.17), 256 (- 1.18), 4096 (- 1.18), 512 (- 1.21)
best 5: 96 (- 1.23), 1000 (- 1.22), 512 (- 1.21), 4096 (- 1.18), 256 (- 1.18)
