# gauntlet report

run: `zttr_c2r_2026-09-29`  contract file suffix: `_c2r_fftw`  cells: 16 listed, 15 benched, comparator: FFTW

control cell: 4 readings, 0.949..0.962 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     refused            -          -       -        -  not benched
         4  2^2              -        raced             10          4    0.41  0.0e+00  
         8  2^3              -        raced             12          5    0.47  2.2e-16  
        16  2^4              -        raced             12         10    0.80  1.1e-16  
        32  2^5              -        raced             16         20    1.19  2.3e-16  
        64  2^6              -        raced             25         33    1.30  2.2e-16  
       128  2^7              -        raced             52         56    1.08  3.4e-16  
       256  2^8              -        raced            109        101    0.91  5.6e-16  
       512  2^9              -        raced            227        204    0.90  5.6e-16  
      1024  2^10             -        raced            488        458    0.89  6.7e-16  
      2048  2^11             -        raced           1119       1054    0.94  7.2e-16  
      4096  2^12             -        raced           2447       2351    0.96  7.8e-16  
      8192  2^13             -        raced           5220       5922    1.11  7.8e-16  
     16384  2^14             -        raced          10988      12835    1.16  7.8e-16  
     32768  2^15             -        raced          23768      26855    1.12  1.0e-15  
     65536  2^16             -        raced          54202      59735    1.09  1.0e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           15      3      8   0.47   0.96   1.19    0.92
 ALL         15      3      8   0.47   0.96   1.19    0.92
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   1   0.41      1      1
 8..31                  2   0.63      2      2
 32..127                2   1.24      0      0
 128..511               2   1.00      1      0
 512..2047              2   0.89      2      0
 2048..8191             2   0.95      2      0
 8192..32767            2   1.14      0      0
 32768..65536           2   1.11      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            15   0.96      8   0.92
```


flip agreement: our two readings more than 25% apart at 0 of 15 cells.

worst 10: 4 (- 0.41), 8 (- 0.47), 16 (- 0.80), 1024 (- 0.89), 512 (- 0.90), 256 (- 0.91), 2048 (- 0.94), 4096 (- 0.96), 128 (- 1.08), 65536 (- 1.09)
best 5: 64 (- 1.30), 32 (- 1.19), 16384 (- 1.16), 32768 (- 1.12), 8192 (- 1.11)
