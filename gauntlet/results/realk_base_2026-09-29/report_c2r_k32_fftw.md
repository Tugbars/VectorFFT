# gauntlet report

run: `realk_base_2026-09-29`  contract file suffix: `_c2r_k32_fftw`  cells: 12 listed, 12 benched, comparator: FFTW

control cell: 4 readings, 0.999..1.083 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        16  2^4              -        replayed         515        193    0.37  3.4e-16  
        32  2^5              -        replayed         684        531    0.77  4.4e-16  
        64  2^6              -        replayed        1007       1076    1.06  5.6e-16  
        96  2^5.3            -        replayed        1637       1547    0.94  8.9e-16  
       128  2^7              -        replayed        2014       1914    0.94  5.6e-16  
       256  2^8              -        replayed        4653       3817    0.78  7.8e-16  
       512  2^9              -        replayed        7381       7785    1.05  7.8e-16  
      1000  2^3.5^3          -        replayed       25229      20241    0.80  1.1e-15  
      1024  2^10             -        replayed       18901      15463    0.81  1.0e-15  
      1215  3^5.5            -        replayed       77998      78016    1.00  1.4e-15  
      2048  2^11             -        replayed       38464      33658    0.87  9.7e-16  
      4096  2^12             -        replayed       84243      86135    0.86  8.9e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           12      4     10   0.77   0.86   1.05    0.83
 ALL         12      4     10   0.77   0.86   1.05    0.83
```


## by size
```
 band               cells median   <1.0   <0.8
 8..31                  1   0.37      1      1
 32..127                3   0.94      2      1
 128..511               2   0.86      2      1
 512..2047              4   0.90      3      1
 2048..4096             2   0.86      2      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             9   0.86      7   0.80
 -                                                3   0.94      3   0.91
```


flip agreement: our two readings more than 25% apart at 0 of 12 cells.

worst 10: 16 (- 0.37), 32 (- 0.77), 256 (- 0.78), 1000 (- 0.80), 1024 (- 0.81), 4096 (- 0.86), 2048 (- 0.87), 96 (- 0.94), 128 (- 0.94), 1215 (- 1.00)
best 5: 64 (- 1.06), 512 (- 1.05), 1215 (- 1.00), 128 (- 0.94), 96 (- 0.94)
