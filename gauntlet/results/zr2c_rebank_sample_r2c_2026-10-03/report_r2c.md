# gauntlet report

run: `zr2c_rebank_sample_r2c_2026-10-03`  contract file suffix: `_r2c`  cells: 5 listed, 5 benched, comparator: MKL

control cell: 4 readings, 1.187..1.199 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
       240  2^4.3.5          -        replayed         112        175    1.56  3.4e-16  
      1000  2^3.5^3          -        replayed         570        833    1.44  4.8e-16  
      4096  2^12             -        raced           2091       2513    1.20  4.1e-16  
     30000  2^4.3.5^4        -        replayed       32304      40434    1.23  5.8e-16  
    131072  2^17             -        raced         107303     151145    1.37  3.8e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            5      0      0   1.20   1.37   1.56    1.35
 ALL          5      0      0   1.20   1.37   1.56    1.35
```


## by size
```
 band               cells median   <1.0   <0.8
 128..511               1   1.56      0      0
 512..2047              1   1.44      0      0
 2048..8191             1   1.20      0      0
 8192..32767            1   1.23      0      0
 131072..131072         1   1.37      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                                3   1.44      0   1.40
 pow2                                             2   1.28      0   1.28
```


flip agreement: our two readings more than 25% apart at 0 of 5 cells.

worst 10: 4096 (- 1.20), 30000 (- 1.23), 131072 (- 1.37), 1000 (- 1.44), 240 (- 1.56)
best 5: 240 (- 1.56), 1000 (- 1.44), 131072 (- 1.37), 30000 (- 1.23), 4096 (- 1.20)
