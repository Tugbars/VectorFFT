# gauntlet report

run: `dst_1d_2026-11-14`  contract file suffix: `_dst3`  cells: 26 listed, 26 benched, comparator: MKL

control cell: 4 readings, 2.036..2.067 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        64  2^6              -        replayed          42        106    2.50  2.2e-15  
       100  2^2.5^2          -        replayed          70        168    2.38  2.0e-15  
       125  5^3              -        replayed         135        254    1.88  8.5e-16  
       128  2^7              -        replayed          77        184    2.37  2.3e-15  
       256  2^8              -        replayed         161        353    2.16  1.9e-14  
       512  2^9              -        replayed         335        688    2.03  2.2e-14  
      1000  2^3.5^3          -        replayed        1008       1645    1.62  8.7e-14  
      1001  7.11.13          -        replayed        1057       2589    2.41  2.8e-14  
      1024  2^10             -        replayed         771       1371    1.76  7.9e-14  
      1536  2^9.3            -        replayed        1225       2459    2.00  2.4e-13  
      2048  2^11             -        replayed        1673       2979    1.77  1.6e-13  
      3125  5^5              -        replayed        3915       9211    2.14  6.2e-13  
      3840  2^8.3.5          -        replayed        3255       7025    2.14  1.2e-12  
      4096  2^12             -        replayed        3329       6800    2.04  1.3e-12  
      8192  2^13             -        replayed        6772      14178    2.08  4.7e-12  
     10000  2^4.5^4          -        replayed       13001      20276    1.55  7.2e-12  
     10007  10007            -        replayed       51591     121461    2.31  1.7e-11  
     16384  2^14             -        replayed       13943      29340    2.10  7.5e-11  
     32768  2^15             -        replayed       35490      64648    1.48  4.8e-11  flips differ 1.30x
     65536  2^16             -        replayed       78598     123376    1.48  3.0e-10  
     65537  65537            -        replayed      569054    1793054    3.02  1.3e-10  
    100000  2^5.5^5          -        replayed      170462     227215    1.32  1.4e-09  
    131072  2^17             -        replayed      196973     273040    1.34  5.0e-10  
    262144  2^18             -        replayed      514393     617730    1.15  1.2e-08  
    524288  2^19             -        replayed     1194687    1523275    1.19  8.9e-09  
   1048576  2^20             -        replayed     3122300    3273712    0.98  1.6e-08  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           26      0      1   1.19   2.02   2.41    1.83
 ALL         26      0      1   1.19   2.02   2.41    1.83
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                3   2.38      0      0
 128..511               2   2.26      0      0
 512..2047              5   2.00      0      0
 2048..8191             4   2.09      0      0
 8192..32767            4   2.09      0      0
 32768..131071          4   1.48      0      0
 131072..524287         2   1.24      0      0
 524288..1048576        2   1.08      1      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            15   1.77      1   1.70
 -                                               11   2.14      0   2.02
```


flip agreement: our two readings more than 25% apart at 1 of 26 cells.

worst 10: 1048576 (- 0.98), 262144 (- 1.15), 524288 (- 1.19), 100000 (- 1.32), 131072 (- 1.34), 32768 (- 1.48), 65536 (- 1.48), 10000 (- 1.55), 1000 (- 1.62), 1024 (- 1.76)
best 5: 65537 (- 3.02), 64 (- 2.50), 1001 (- 2.41), 100 (- 2.38), 128 (- 2.37)
