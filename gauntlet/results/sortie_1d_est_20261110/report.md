# gauntlet report

run: `sortie_1d_est_20261110`  contract file suffix: `(oop, T=1)`  cells: 8 listed, 8 benched, comparator: MKL

control cell: 4 readings, 1.068..1.083 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      1920  2^7.3.5          chain3   replayed        2209       3133    1.41  4.7e-16  
      2048  2^11             ztt      replayed        1624       2094    1.27  3.1e-16  
      2560  2^9.5            ztt      replayed        2234       4477    1.99  3.3e-16  
      3072  2^10.3           ztt      replayed        2951       5229    1.72  3.0e-16  
      3840  2^8.3.5          ztt      replayed        4013       6732    1.67  4.1e-16  
     65536  2^16             ztt      replayed       90280     114905    1.06  4.8e-16  
   1048576  2^20             fs       replayed     3857637    5015119    1.30  4.6e-16  
   4194304  2^22             fs       replayed    21092225   28112831    1.31  9.4e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt          5      0      0   1.06   1.67   1.99    1.50
 fs           2      0      0   1.30   1.31   1.31    1.31
 chain3       1      0      0   1.41   1.41   1.41    1.41
 ALL          8      0      0   1.06   1.36   1.99    1.44
```


## by size
```
 band               cells median   <1.0   <0.8
 512..2047              1   1.41      0      0
 2048..8191             4   1.69      0      0
 32768..131071          1   1.06      0      0
 524288..2097151        1   1.30      0      0
 2097152..4194304       1   1.31      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             4   1.28      0   1.23
 ztt                                              3   1.72      0   1.79
 chain3                                           1   1.41      0   1.41
```


flip agreement: our two readings more than 25% apart at 0 of 8 cells.

worst 10: 65536 (ztt 1.06), 2048 (ztt 1.27), 1048576 (fs 1.30), 4194304 (fs 1.31), 1920 (chain3 1.41), 3840 (ztt 1.67), 3072 (ztt 1.72), 2560 (ztt 1.99)
best 5: 2560 (ztt 1.99), 3072 (ztt 1.72), 3840 (ztt 1.67), 1920 (chain3 1.41), 4194304 (fs 1.31)
