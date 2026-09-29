# gauntlet report

run: `realk_base_2026-09-29`  contract file suffix: `_r2c_k32_fftw`  cells: 12 listed, 12 benched, comparator: FFTW

control cell: 4 readings, 0.987..1.049 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        16  2^4              -        replayed         516        197    0.38  1.6e-16  
        32  2^5              -        replayed         694        513    0.72  2.0e-16  
        64  2^6              -        replayed        1042       1304    1.25  1.9e-16  
        96  2^5.3            -        replayed        1556       1712    1.08  3.8e-16  
       128  2^7              -        replayed        2022       1866    0.92  2.6e-16  
       256  2^8              -        replayed        3751       3756    0.93  2.9e-16  
       512  2^9              -        replayed        8275       7883    0.81  3.1e-16  
      1000  2^3.5^3          -        replayed       23169      21077    0.87  4.6e-16  
      1024  2^10             -        replayed       17153      16457    0.73  3.5e-16  flips differ 1.31x
      1215  3^5.5            -        replayed       63914      74950    1.10  6.3e-16  
      2048  2^11             -        replayed       38023      36786    0.95  3.3e-16  
      4096  2^12             -        replayed       87803      91486    1.02  4.4e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           12      3      8   0.72   0.93   1.10    0.86
 ALL         12      3      8   0.72   0.93   1.10    0.86
```


## by size
```
 band               cells median   <1.0   <0.8
 8..31                  1   0.38      1      1
 32..127                3   1.08      1      1
 128..511               2   0.93      2      0
 512..2047              4   0.84      3      1
 2048..4096             2   0.99      1      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             9   0.92      7   0.82
 -                                                3   1.08      1   1.01
```


flip agreement: our two readings more than 25% apart at 1 of 12 cells.

worst 10: 16 (- 0.38), 32 (- 0.72), 1024 (- 0.73), 512 (- 0.81), 1000 (- 0.87), 128 (- 0.92), 256 (- 0.93), 2048 (- 0.95), 4096 (- 1.02), 96 (- 1.08)
best 5: 64 (- 1.25), 1215 (- 1.10), 96 (- 1.08), 4096 (- 1.02), 2048 (- 0.95)
