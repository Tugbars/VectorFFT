# gauntlet report

run: `gcc_pow2_mkl_20261021`  contract file suffix: `(oop, T=1)`  cells: 23 listed, 23 benched, comparator: MKL

control cell: 4 readings, 1.076..1.082 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     raced              2         11    4.35  0.0e+00  
         4  2^2              mono     raced              3         11    3.63  0.0e+00  
         8  2^3              mono     raced              5         11    2.12  7.4e-17  
        16  2^4              ztt      raced              8         12    1.44  7.5e-17  
        32  2^5              2p       raced             15         16    1.05  1.7e-16  
        64  2^6              2p       raced             30         31    1.04  1.5e-16  
       128  2^7              2p       raced             64         68    0.72  3.2e-16  flips differ 1.48x
       256  2^8              2p       raced            134        137    1.02  0.0e+00  
       512  2^9              2p       raced            296        294    0.88  2.3e-16  
      1024  2^10             ztt      raced            714        845    1.18  4.0e-16  
      2048  2^11             ztt      raced           1608       2036    0.87  3.1e-16  flips differ 1.46x
      4096  2^12             ztt      raced           3543       3826    1.07  3.9e-16  
      8192  2^13             ztt      raced           7739       8463    1.09  4.1e-16  
     16384  2^14             ztt      raced          17298      19006    0.97  3.6e-16  
     32768  2^15             ztt      raced          39048      36996    0.58  3.4e-16  flips differ 1.64x
     65536  2^16             ztt      raced          86933     108923    1.25  4.8e-16  
    131072  2^17             ztt      raced         222800     261610    1.17  4.6e-16  
    262144  2^18             ztt      raced         542750     695187    1.23  3.3e-16  
    524288  2^19             fs       raced        1384938    1522831    1.08  4.5e-16  
   1048576  2^20             fs       raced        3144950    4158025    1.29  5.3e-16  
   2097152  2^21             fs       raced        7915738   10717219    1.35  7.4e-16  
   4194304  2^22             fs       raced       18441563   25759112    1.39  9.7e-16  
   8388608  2^23             fs       raced       40587675   59761525    1.46  1.2e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt         10      1      3   0.87   1.13   1.44    1.06
 fs           5      0      0   1.08   1.35   1.46    1.31
 2p           5      1      2   0.72   1.02   1.05    0.93
 mono         3      0      0   2.12   3.63   4.35    3.22
 ALL         23      2      5   0.87   1.17   2.12    1.25
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   2   3.99      0      0
 8..31                  2   1.78      0      0
 32..127                2   1.04      0      0
 128..511               2   0.87      1      1
 512..2047              2   1.03      1      0
 2048..8191             2   0.97      1      0
 8192..32767            2   1.03      1      0
 32768..131071          2   0.91      1      1
 131072..524287         2   1.20      0      0
 524288..2097151        2   1.18      0      0
 2097152..8388607       2   1.37      0      0
 8388608..8388608       1   1.46      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            23   1.17      5   1.25
```


flip agreement: our two readings more than 25% apart at 3 of 23 cells.

worst 10: 32768 (ztt 0.58), 128 (2p 0.72), 2048 (ztt 0.87), 512 (2p 0.88), 16384 (ztt 0.97), 256 (2p 1.02), 64 (2p 1.04), 32 (2p 1.05), 4096 (ztt 1.07), 524288 (fs 1.08)
best 5: 2 (mono 4.35), 4 (mono 3.63), 8 (mono 2.12), 8388608 (fs 1.46), 16 (ztt 1.44)
