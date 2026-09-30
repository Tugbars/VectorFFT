# gauntlet report

run: `real_batch_r2c_k256_t1_2026-09-30`  contract file suffix: `_r2c_k256`  cells: 6 listed, 6 benched, comparator: MKL

control cell: 4 readings, 1.083..1.205 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        16  2^4              -        replayed        2620       1591    0.60  1.1e-16  
        64  2^6              -        replayed        7900       7195    0.90  2.2e-16  
       256  2^8              -        replayed       29805      34027    1.09  4.0e-16  
      1024  2^10             -        replayed      178600     189533    1.06  4.6e-16  
      4096  2^12             -        replayed      752275     909037    1.14  4.3e-16  
     16384  2^14             -        replayed     4361150    5121262    1.11  3.9e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            6      1      2   0.60   1.08   1.14    0.96
 ALL          6      1      2   0.60   1.08   1.14    0.96
```


## by size
```
 band               cells median   <1.0   <0.8
 8..31                  1   0.60      1      1
 32..127                1   0.90      1      0
 128..511               1   1.09      0      0
 512..2047              1   1.06      0      0
 2048..8191             1   1.14      0      0
 8192..16384            1   1.11      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             6   1.08      2   0.96
```


flip agreement: our two readings more than 25% apart at 0 of 6 cells.

worst 10: 16 (- 0.60), 64 (- 0.90), 1024 (- 1.06), 256 (- 1.09), 16384 (- 1.11), 4096 (- 1.14)
best 5: 4096 (- 1.14), 16384 (- 1.11), 256 (- 1.09), 1024 (- 1.06), 64 (- 0.90)
