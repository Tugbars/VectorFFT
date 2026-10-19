# gauntlet report

run: `pow2rec_t8_mkl_2026-10-20`  contract file suffix: `_mt8`  cells: 23 listed, 23 benched, comparator: MKL

control cell: 4 readings, 1.020..1.055 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     raced            114         11    0.10  0.0e+00  
         4  2^2              mono     raced            115         11    0.09  0.0e+00  
         8  2^3              mono     raced            125         12    0.10  7.4e-17  
        16  2^4              ztt      replayed         130         12    0.09  7.5e-17  
        32  2^5              2p       raced            142         16    0.11  1.7e-16  
        64  2^6              2p       raced            152         31    0.20  1.5e-16  
       128  2^7              2p       raced            177         68    0.38  3.2e-16  
       256  2^8              2p       raced            248        136    0.55  0.0e+00  
       512  2^9              2p       raced            412        293    0.70  2.3e-16  
      1024  2^10             ztt      replayed         846        845    0.97  4.0e-16  
      2048  2^11             ztt      replayed        1744       2114    0.81  3.1e-16  flips differ 1.53x
      4096  2^12             ztt      replayed        3715       3875    1.03  3.9e-16  
      8192  2^13             ztt      replayed        6246      10608    1.21  3.4e-16  
     16384  2^14             ztt      replayed       11427      15242    1.19  3.6e-16  
     32768  2^15             ztt      replayed       21985      22333    0.97  3.5e-16  
     65536  2^16             ztt      replayed       27880      40458    1.17  4.0e-16  
    131072  2^17             ztt      replayed       55327      84433    1.49  4.6e-16  
    262144  2^18             ztt      replayed      108825     172588    1.50  4.0e-16  
    524288  2^19             fs       raced         285950     367969    1.12  4.3e-16  
   1048576  2^20             fs       raced         650313    1117243    1.71  1.1e-15  
   2097152  2^21             fs       raced        1806875    2848687    1.44  2.7e-15  
   4194304  2^22             fs       raced        7144887    7500550    1.01  9.2e-16  
   8388608  2^23             fs       raced       15787488   17070962    0.95  9.1e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt         10      1      4   0.81   1.10   1.50    0.88
 fs           5      0      1   0.95   1.12   1.71    1.22
 2p           5      5      5   0.11   0.38   0.70    0.32
 mono         3      3      3   0.09   0.10   0.10    0.09
 ALL         23      9     13   0.10   0.97   1.49    0.57
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   2   0.09      2      2
 8..31                  2   0.09      2      2
 32..127                2   0.15      2      2
 128..511               2   0.47      2      2
 512..2047              2   0.84      2      1
 2048..8191             2   0.92      1      0
 8192..32767            2   1.20      0      0
 32768..131071          2   1.07      1      0
 131072..524287         2   1.49      0      0
 524288..2097151        2   1.42      0      0
 2097152..8388607       2   1.22      0      0
 8388608..8388608       1   0.95      1      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            23   0.97     13   0.57
```


flip agreement: our two readings more than 25% apart at 1 of 23 cells.

worst 10: 16 (ztt 0.09), 4 (mono 0.09), 2 (mono 0.10), 8 (mono 0.10), 32 (2p 0.11), 64 (2p 0.20), 128 (2p 0.38), 256 (2p 0.55), 512 (2p 0.70), 2048 (ztt 0.81)
best 5: 1048576 (fs 1.71), 262144 (ztt 1.50), 131072 (ztt 1.49), 2097152 (fs 1.44), 8192 (ztt 1.21)
