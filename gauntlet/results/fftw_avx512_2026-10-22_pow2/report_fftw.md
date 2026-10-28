# gauntlet report

run: `fftw_avx512_2026-10-22_pow2`  contract file suffix: `_fftw`  cells: 23 listed, 23 benched, comparator: FFTW

control cell: 4 readings, 1.315..1.359 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     raced             13         15    1.12  0.0e+00  
         4  2^2              mono     raced             19         17    0.91  0.0e+00  
         8  2^3              mono     raced             28         25    0.91  5.6e-17  
        16  2^4              2p       raced             45         40    0.88  9.4e-17  
        32  2^5              2p       raced             61         70    1.13  1.3e-16  
        64  2^6              mono     raced             91        112    1.23  3.8e-16  
       128  2^7              2p       raced            187        207    1.09  2.1e-16  
       256  2^8              2p       raced            341        432    1.19  2.5e-16  
       512  2^9              2p       raced            771        901    1.16  3.1e-16  
      1024  2^10             ztt      raced           1924       2140    1.05  2.5e-16  
      2048  2^11             ztt      raced           4289       5067    1.17  3.1e-16  
      4096  2^12             ztt      raced           9593      12796    1.33  3.4e-16  
      8192  2^13             ztt      raced          20653      29362    1.41  3.4e-16  
     16384  2^14             ztt      raced          44291      61359    1.37  4.5e-16  
     32768  2^15             ztt      raced         101392     157705    1.54  4.5e-16  
     65536  2^16             ztt      raced         226423     546506    1.97  3.2e-16  
    131072  2^17             ztt      raced         501453    1347413    2.56  4.6e-16  
    262144  2^18             ztt      raced        1091937    2228743    1.99  4.0e-16  
    524288  2^19             fs       raced        3621488    9985112    2.75  4.5e-16  
   1048576  2^20             fs       raced       10202637   24004106    2.30  4.6e-16  
   2097152  2^21             fs       raced       20910475   48229037    2.29  1.0e-15  
   4194304  2^22             fs       raced       43225600  125749774    2.82  7.1e-16  
   8388608  2^23             fs       raced       93316263  252380743    2.61  6.4e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt          9      0      0   1.05   1.41   2.56    1.54
 2p           5      0      1   0.88   1.13   1.19    1.08
 fs           5      0      0   2.29   2.61   2.82    2.54
 mono         4      0      2   0.91   1.02   1.23    1.04
 ALL         23      0      3   0.91   1.33   2.61    1.48
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   2   1.02      1      0
 8..31                  2   0.89      2      0
 32..127                2   1.18      0      0
 128..511               2   1.14      0      0
 512..2047              2   1.10      0      0
 2048..8191             2   1.25      0      0
 8192..32767            2   1.39      0      0
 32768..131071          2   1.75      0      0
 131072..524287         2   2.27      0      0
 524288..2097151        2   2.52      0      0
 2097152..8388607       2   2.55      0      0
 8388608..8388608       1   2.61      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            23   1.33      3   1.48
```


flip agreement: our two readings more than 25% apart at 0 of 23 cells.

worst 10: 16 (2p 0.88), 4 (mono 0.91), 8 (mono 0.91), 1024 (ztt 1.05), 128 (2p 1.09), 2 (mono 1.12), 32 (2p 1.13), 512 (2p 1.16), 2048 (ztt 1.17), 256 (2p 1.19)
best 5: 4194304 (fs 2.82), 524288 (fs 2.75), 8388608 (fs 2.61), 131072 (ztt 2.56), 1048576 (fs 2.30)
