# gauntlet report

run: `pow2rec_t1_mkl_2026-10-20`  contract file suffix: `(oop, T=1)`  cells: 23 listed, 23 benched, comparator: MKL

control cell: 4 readings, 1.074..1.088 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     replayed           2         11    4.36  0.0e+00  
         4  2^2              mono     replayed           3         11    3.63  0.0e+00  
         8  2^3              mono     replayed           5         11    2.12  7.4e-17  
        16  2^4              ztt      replayed           9         12    1.37  7.5e-17  
        32  2^5              2p       replayed          15         16    1.04  1.7e-16  
        64  2^6              2p       replayed          30         31    1.03  1.5e-16  
       128  2^7              2p       replayed          64         68    0.98  3.2e-16  
       256  2^8              2p       replayed         134        137    1.02  0.0e+00  
       512  2^9              2p       replayed         296        295    0.99  2.3e-16  
      1024  2^10             ztt      replayed         718        876    0.77  4.0e-16  flips differ 1.58x
      2048  2^11             ztt      replayed        1598       2078    1.28  3.1e-16  
      4096  2^12             ztt      replayed        3564       3830    1.07  3.9e-16  
      8192  2^13             ztt      replayed        7966       8505    1.06  4.1e-16  
     16384  2^14             ztt      replayed       17592      18955    1.06  4.5e-16  
     32768  2^15             ztt      replayed       39449      38952    0.97  4.5e-16  
     65536  2^16             ztt      replayed       98013     108338    1.10  4.8e-16  
    131072  2^17             ztt      replayed      262133     287046    1.06  4.6e-16  
    262144  2^18             ztt      replayed      540812     723750    1.25  3.3e-16  
    524288  2^19             fs       raced        1374000    1493375    1.05  5.6e-16  
   1048576  2^20             fs       raced        3310425    4216850    1.26  5.3e-16  
   2097152  2^21             fs       raced        8148288   11072068    1.31  7.4e-16  
   4194304  2^22             fs       raced       18937675   26115262    1.35  9.7e-16  
   8388608  2^23             fs       raced       42781088   59604500    1.37  1.2e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt         10      1      2   0.97   1.07   1.37    1.09
 2p           5      0      2   0.98   1.02   1.04    1.01
 fs           5      0      0   1.05   1.31   1.37    1.26
 mono         3      0      0   2.12   3.63   4.36    3.22
 ALL         23      1      4   0.98   1.07   2.12    1.27
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   2   3.99      0      0
 8..31                  2   1.75      0      0
 32..127                2   1.04      0      0
 128..511               2   1.00      1      0
 512..2047              2   0.88      2      1
 2048..8191             2   1.18      0      0
 8192..32767            2   1.06      0      0
 32768..131071          2   1.03      1      0
 131072..524287         2   1.16      0      0
 524288..2097151        2   1.16      0      0
 2097152..8388607       2   1.33      0      0
 8388608..8388608       1   1.37      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            23   1.07      4   1.27
```


flip agreement: our two readings more than 25% apart at 1 of 23 cells.

worst 10: 1024 (ztt 0.77), 32768 (ztt 0.97), 128 (2p 0.98), 512 (2p 0.99), 256 (2p 1.02), 64 (2p 1.03), 32 (2p 1.04), 524288 (fs 1.05), 16384 (ztt 1.06), 8192 (ztt 1.06)
best 5: 2 (mono 4.36), 4 (mono 3.63), 8 (mono 2.12), 16 (ztt 1.37), 8388608 (fs 1.37)
