# gauntlet report

run: `icx_pow2_mkl_20261021`  contract file suffix: `(oop, T=1)`  cells: 23 listed, 23 benched, comparator: MKL

control cell: 4 readings, 1.062..1.078 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     raced              3         11    3.62  0.0e+00  
         4  2^2              mono     raced              3         11    3.40  0.0e+00  
         8  2^3              mono     raced              5         11    2.47  1.5e-16  
        16  2^4              ztt      raced              8         12    1.55  7.5e-17  
        32  2^5              2p       raced             16         16    0.95  8.7e-17  
        64  2^6              2p       raced             30         31    0.99  3.1e-16  
       128  2^7              2p       raced             66         69    1.05  2.7e-16  
       256  2^8              2p       raced            138        137    0.99  2.5e-16  
       512  2^9              2p       raced            300        298    0.99  2.3e-16  
      1024  2^10             ztt      raced            727        852    1.15  2.5e-16  
      2048  2^11             ztt      raced           1623       2033    1.25  3.4e-16  
      4096  2^12             ztt      raced           3607       3827    1.06  3.9e-16  
      8192  2^13             ztt      raced           7719       8478    1.10  4.1e-16  
     16384  2^14             ztt      raced          17482      18852    1.08  4.5e-16  
     32768  2^15             ztt      raced          38805      37046    0.95  4.5e-16  
     65536  2^16             ztt      raced          86230     108921    1.25  3.6e-16  
    131072  2^17             ztt      raced         221487     276670    1.18  5.2e-16  
    262144  2^18             ztt      raced         543913     706793    1.29  4.0e-16  
    524288  2^19             fs       raced        1431338    1480994    1.02  5.6e-16  
   1048576  2^20             fs       raced        3311700    4140294    1.22  5.3e-16  
   2097152  2^21             fs       raced        8303887   10747512    1.27  1.5e-15  
   4194304  2^22             fs       raced       19169338   25766612    1.24  1.1e-15  
   8388608  2^23             fs       raced       41803887   60211981    1.43  2.5e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt         10      0      1   1.06   1.16   1.55    1.18
 fs           5      0      0   1.02   1.24   1.43    1.23
 2p           5      0      4   0.95   0.99   1.05    0.99
 mono         3      0      0   2.47   3.40   3.62    3.12
 ALL         23      0      5   0.99   1.18   2.47    1.30
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   2   3.51      0      0
 8..31                  2   2.01      0      0
 32..127                2   0.97      2      0
 128..511               2   1.02      1      0
 512..2047              2   1.07      1      0
 2048..8191             2   1.15      0      0
 8192..32767            2   1.09      0      0
 32768..131071          2   1.10      1      0
 131072..524287         2   1.24      0      0
 524288..2097151        2   1.12      0      0
 2097152..8388607       2   1.26      0      0
 8388608..8388608       1   1.43      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            23   1.18      5   1.30
```


flip agreement: our two readings more than 25% apart at 0 of 23 cells.

worst 10: 32 (2p 0.95), 32768 (ztt 0.95), 512 (2p 0.99), 64 (2p 0.99), 256 (2p 0.99), 524288 (fs 1.02), 128 (2p 1.05), 4096 (ztt 1.06), 16384 (ztt 1.08), 8192 (ztt 1.10)
best 5: 2 (mono 3.62), 4 (mono 3.40), 8 (mono 2.47), 16 (ztt 1.55), 8388608 (fs 1.43)
