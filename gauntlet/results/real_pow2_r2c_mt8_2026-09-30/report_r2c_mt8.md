# gauntlet report

run: `real_pow2_r2c_mt8_2026-09-30`  contract file suffix: `_r2c_mt8`  cells: 23 listed, 23 benched, comparator: MKL

control cell: 2 readings, 1.200..1.252 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                -        replayed          11         12    1.16  0.0e+00  
         4  2^2              -        raced              4         12    3.24  0.0e+00  
         8  2^3              -        raced              5         13    2.41  7.4e-17  
        16  2^4              -        raced             10         14    1.35  4.6e-17  
        32  2^5              -        raced             18         20    1.13  1.2e-16  
        64  2^6              -        raced             29         29    1.01  1.6e-16  
       128  2^7              -        raced             54         57    1.06  2.0e-16  
       256  2^8              -        raced            102        127    1.24  3.3e-16  
       512  2^9              -        raced            205        260    1.27  4.1e-16  
      1024  2^10             -        raced            423        531    1.26  3.4e-16  
      2048  2^11             -        raced            974       1195    1.23  3.0e-16  
      4096  2^12             -        raced           2093       2525    1.21  4.1e-16  
      8192  2^13             -        raced           4732       6061    1.28  3.2e-16  
     16384  2^14             -        raced           6329      10301    1.34  3.9e-16  
     32768  2^15             -        raced          12925      18273    1.22  3.5e-16  
     65536  2^16             -        raced          18169      24869    1.37  3.7e-16  
    131072  2^17             -        raced          28923      51227    1.77  3.8e-16  
    262144  2^18             -        raced          54533     104573    1.86  4.7e-16  
    524288  2^19             -        raced         107087     200450    1.74  4.4e-16  
   1048576  2^20             -        raced         271950     378556    1.33  4.6e-16  
   2097152  2^21             -        raced         622600    1306988    2.10  1.5e-15  
   4194304  2^22             -        raced        1718088    3837188    2.18  2.9e-15  
   8388608  2^23             -        raced        6810300    9636831    1.39  1.4e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           23      0      0   1.13   1.33   2.18    1.46
 ALL         23      0      0   1.13   1.33   2.18    1.46
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   2   2.20      0      0
 8..31                  2   1.88      0      0
 32..127                2   1.07      0      0
 128..511               2   1.15      0      0
 512..2047              2   1.26      0      0
 2048..8191             2   1.22      0      0
 8192..32767            2   1.31      0      0
 32768..131071          2   1.29      0      0
 131072..524287         2   1.81      0      0
 524288..2097151        2   1.54      0      0
 2097152..8388607       2   2.14      0      0
 8388608..8388608       1   1.39      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            23   1.33      0   1.46
```


flip agreement: our two readings more than 25% apart at 0 of 23 cells.

worst 10: 64 (- 1.01), 128 (- 1.06), 32 (- 1.13), 2 (- 1.16), 4096 (- 1.21), 32768 (- 1.22), 2048 (- 1.23), 256 (- 1.24), 1024 (- 1.26), 512 (- 1.27)
best 5: 4 (- 3.24), 8 (- 2.41), 4194304 (- 2.18), 2097152 (- 2.10), 262144 (- 1.86)
