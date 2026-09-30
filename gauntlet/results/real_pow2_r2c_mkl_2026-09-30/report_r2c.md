# gauntlet report

run: `real_pow2_r2c_mkl_2026-09-30`  contract file suffix: `_r2c`  cells: 23 listed, 23 benched, comparator: MKL

control cell: 4 readings, 1.197..1.206 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                -        replayed           9         12    1.29  0.0e+00  
         4  2^2              -        replayed           4         12    3.02  0.0e+00  
         8  2^3              -        replayed           6         12    2.24  7.4e-17  
        16  2^4              -        replayed          10         14    1.32  4.6e-17  
        32  2^5              -        replayed          18         20    1.12  1.2e-16  
        64  2^6              -        replayed          26         30    1.15  1.6e-16  
       128  2^7              -        replayed          51         57    1.11  2.0e-16  
       256  2^8              -        replayed         100        128    1.26  3.3e-16  
       512  2^9              -        replayed         200        261    1.30  4.1e-16  
      1024  2^10             -        replayed         421        533    1.26  3.4e-16  
      2048  2^11             -        replayed         965       1194    1.24  3.0e-16  
      4096  2^12             -        replayed        2084       2498    1.19  4.1e-16  
      8192  2^13             -        replayed        4519       6027    1.31  3.2e-16  
     16384  2^14             -        replayed        9726      12830    1.30  4.0e-16  
     32768  2^15             -        replayed       21464      27418    1.27  3.5e-16  
     65536  2^16             -        replayed       45856      55260    1.20  4.0e-16  
    131072  2^17             -        raced         103370     148393    1.41  4.4e-16  
    262144  2^18             -        raced         259780     342683    1.23  4.7e-16  
    524288  2^19             -        raced         640650     859475    1.32  4.4e-16  
   1048576  2^20             -        raced        1891225    2130868    1.02  5.6e-16  
   2097152  2^21             -        raced        4401500    5336075    1.20  6.1e-16  
   4194304  2^22             -        raced       11150738   13654893    1.20  1.0e-15  
   8388608  2^23             -        raced       25497462   30467456    1.16  1.1e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           23      0      0   1.12   1.26   1.41    1.31
 ALL         23      0      0   1.12   1.26   1.41    1.31
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   2   2.15      0      0
 8..31                  2   1.78      0      0
 32..127                2   1.14      0      0
 128..511               2   1.19      0      0
 512..2047              2   1.28      0      0
 2048..8191             2   1.21      0      0
 8192..32767            2   1.31      0      0
 32768..131071          2   1.24      0      0
 131072..524287         2   1.32      0      0
 524288..2097151        2   1.17      0      0
 2097152..8388607       2   1.20      0      0
 8388608..8388608       1   1.16      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            23   1.26      0   1.31
```


flip agreement: our two readings more than 25% apart at 0 of 23 cells.

worst 10: 1048576 (- 1.02), 128 (- 1.11), 32 (- 1.12), 64 (- 1.15), 8388608 (- 1.16), 4096 (- 1.19), 2097152 (- 1.20), 65536 (- 1.20), 4194304 (- 1.20), 262144 (- 1.23)
best 5: 4 (- 3.02), 8 (- 2.24), 131072 (- 1.41), 16 (- 1.32), 524288 (- 1.32)
