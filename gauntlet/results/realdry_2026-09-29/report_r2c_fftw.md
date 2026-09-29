# gauntlet report

run: `realdry_2026-09-29`  contract file suffix: `_r2c_fftw`  cells: 3 listed, 3 benched, comparator: FFTW

control cell: 4 readings, 1.136..1.232 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      1000  2^3.5^3          -        raced            587        545    0.92  4.8e-16  
      1024  2^10             -        replayed         421        427    1.01  3.4e-16  
      1215  3^5.5            -        raced           2140       2371    1.10  5.8e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            3      0      1   0.92   1.01   1.10    1.01
 ALL          3      0      1   0.92   1.01   1.10    1.01
```


## by size
```
 band               cells median   <1.0   <0.8
 512..1215              3   1.01      1      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                                2   1.01      1   1.01
 pow2                                             1   1.01      0   1.01
```


flip agreement: our two readings more than 25% apart at 0 of 3 cells.

worst 10: 1000 (- 0.92), 1024 (- 1.01), 1215 (- 1.10)
best 5: 1215 (- 1.10), 1024 (- 1.01), 1000 (- 0.92)
