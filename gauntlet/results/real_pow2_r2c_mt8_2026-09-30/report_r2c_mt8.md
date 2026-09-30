# gauntlet report

run: `real_pow2_r2c_mt8_2026-09-30`  contract file suffix: `_r2c_mt8`  cells: 23 listed, 23 benched, comparator: MKL

control cell: 2 readings, 1.183..1.236 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                -        replayed          10         12    1.15  0.0e+00  
         4  2^2              -        raced              4         12    3.26  0.0e+00  
         8  2^3              -        raced              5         13    2.33  7.4e-17  
        16  2^4              -        raced             10         14    1.37  4.6e-17  
        32  2^5              -        raced             18         20    1.14  1.2e-16  
        64  2^6              -        raced             26         30    1.15  1.6e-16  
       128  2^7              -        raced             53         57    1.07  2.0e-16  
       256  2^8              -        raced            115        127    1.10  3.3e-16  
       512  2^9              -        raced            218        260    1.19  4.1e-16  
      1024  2^10             -        raced            433        530    1.23  3.4e-16  
      2048  2^11             -        raced            960       1195    1.25  3.0e-16  
      4096  2^12             -        raced           2097       2620    1.25  4.1e-16  
      8192  2^13             -        raced           4629       6042    1.30  3.6e-16  
     16384  2^14             -        raced           7247      11559    1.51  3.9e-16  
     32768  2^15             -        raced          15925      18404    1.14  4.4e-16  
     65536  2^16             -        raced          24630      26177    1.02  3.5e-16  
    131072  2^17             -        raced          33120      52773    1.59  4.0e-16  
    262144  2^18             -        raced          63347     107187    1.69  4.7e-16  
    524288  2^19             -        raced         127762     203931    1.43  4.4e-16  
   1048576  2^20             -        raced         285637     400931    1.29  4.6e-16  
   2097152  2^21             -        raced         655387    1160950    1.77  1.5e-15  
   4194304  2^22             -        raced        1770313    3812081    2.05  2.9e-15  
   8388608  2^23             -        raced        7042912    9612131    1.25  1.4e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           23      0      0   1.10   1.25   2.05    1.40
 ALL         23      0      0   1.10   1.25   2.05    1.40
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   2   2.20      0      0
 8..31                  2   1.85      0      0
 32..127                2   1.14      0      0
 128..511               2   1.09      0      0
 512..2047              2   1.21      0      0
 2048..8191             2   1.25      0      0
 8192..32767            2   1.41      0      0
 32768..131071          2   1.08      0      0
 131072..524287         2   1.64      0      0
 524288..2097151        2   1.36      0      0
 2097152..8388607       2   1.91      0      0
 8388608..8388608       1   1.25      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            23   1.25      0   1.40
```


flip agreement: our two readings more than 25% apart at 0 of 23 cells.

worst 10: 65536 (- 1.02), 128 (- 1.07), 256 (- 1.10), 32768 (- 1.14), 32 (- 1.14), 2 (- 1.15), 64 (- 1.15), 512 (- 1.19), 1024 (- 1.23), 2048 (- 1.25)
best 5: 4 (- 3.26), 8 (- 2.33), 4194304 (- 2.05), 2097152 (- 1.77), 262144 (- 1.69)
