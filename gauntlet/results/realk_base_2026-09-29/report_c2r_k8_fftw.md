# gauntlet report

run: `realk_base_2026-09-29`  contract file suffix: `_c2r_k8_fftw`  cells: 12 listed, 12 benched, comparator: FFTW

control cell: 4 readings, 0.955..0.968 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        16  2^4              -        replayed         139         50    0.36  3.4e-16  
        32  2^5              -        raced            180        135    0.75  3.3e-16  
        64  2^6              -        replayed         261        305    0.95  4.4e-16  
        96  2^5.3            -        raced            382        365    0.95  6.7e-16  
       128  2^7              -        replayed         453        458    1.00  5.6e-16  
       256  2^8              -        replayed         993        877    0.87  5.6e-16  
       512  2^9              -        replayed        1840       1898    0.92  7.8e-16  
      1000  2^3.5^3          -        raced           6325       5029    0.78  1.1e-15  
      1024  2^10             -        replayed        4719       3839    0.76  1.0e-15  
      1215  3^5.5            -        replayed       19198      19469    0.97  1.3e-15  
      2048  2^11             -        replayed        9782       8583    0.84  9.7e-16  
      4096  2^12             -        replayed       19667      19120    0.96  7.8e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           12      4     11   0.75   0.90   0.97    0.82
 ALL         12      4     11   0.75   0.90   0.97    0.82
```


## by size
```
 band               cells median   <1.0   <0.8
 8..31                  1   0.36      1      1
 32..127                3   0.95      3      1
 128..511               2   0.94      1      0
 512..2047              4   0.85      4      2
 2048..4096             2   0.90      2      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             9   0.87      8   0.79
 -                                                3   0.95      3   0.90
```


flip agreement: our two readings more than 25% apart at 0 of 12 cells.

worst 10: 16 (- 0.36), 32 (- 0.75), 1024 (- 0.76), 1000 (- 0.78), 2048 (- 0.84), 256 (- 0.87), 512 (- 0.92), 64 (- 0.95), 96 (- 0.95), 4096 (- 0.96)
best 5: 128 (- 1.00), 1215 (- 0.97), 4096 (- 0.96), 64 (- 0.95), 96 (- 0.95)
