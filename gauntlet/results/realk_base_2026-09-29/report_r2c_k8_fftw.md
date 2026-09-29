# gauntlet report

run: `realk_base_2026-09-29`  contract file suffix: `_r2c_k8_fftw`  cells: 12 listed, 12 benched, comparator: FFTW

control cell: 4 readings, 0.730..0.975 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        16  2^4              -        replayed         138         52    0.38  1.8e-16  
        32  2^5              -        raced            179        127    0.71  1.2e-16  
        64  2^6              -        replayed         269        336    1.21  1.6e-16  
        96  2^5.3            -        raced            386        384    0.99  3.8e-16  
       128  2^7              -        replayed         478        448    0.94  3.2e-16  
       256  2^8              -        replayed         853        855    0.97  3.1e-16  
       512  2^9              -        replayed        2045       1952    0.94  2.6e-16  
      1000  2^3.5^3          -        raced           5963       5250    0.87  4.6e-16  
      1024  2^10             -        replayed        4230       4149    0.98  4.2e-16  
      1215  3^5.5            -        raced          15949      18794    1.16  5.4e-16  
      2048  2^11             -        replayed        9449       9131    0.96  4.0e-16  
      4096  2^12             -        replayed       20875      20494    0.98  4.4e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           12      2     10   0.71   0.97   1.16    0.89
 ALL         12      2     10   0.71   0.97   1.16    0.89
```


## by size
```
 band               cells median   <1.0   <0.8
 8..31                  1   0.38      1      1
 32..127                3   0.99      2      1
 128..511               2   0.96      2      0
 512..2047              4   0.96      3      0
 2048..4096             2   0.97      2      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             9   0.96      8   0.86
 -                                                3   0.99      2   1.00
```


flip agreement: our two readings more than 25% apart at 0 of 12 cells.

worst 10: 16 (- 0.38), 32 (- 0.71), 1000 (- 0.87), 128 (- 0.94), 512 (- 0.94), 2048 (- 0.96), 256 (- 0.97), 4096 (- 0.98), 1024 (- 0.98), 96 (- 0.99)
best 5: 64 (- 1.21), 1215 (- 1.16), 96 (- 0.99), 1024 (- 0.98), 4096 (- 0.98)
