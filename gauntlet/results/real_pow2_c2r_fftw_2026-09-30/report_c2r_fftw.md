# gauntlet report

run: `real_pow2_c2r_fftw_2026-09-30`  contract file suffix: `_c2r_fftw`  cells: 23 listed, 22 benched, comparator: FFTW

control cell: 4 readings, 1.106..1.181 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     refused            -          -       -        -  not benched
         4  2^2              -        replayed           4          4    0.98  0.0e+00  
         8  2^3              -        replayed           6          6    0.96  1.7e-16  
        16  2^4              -        replayed          12         10    0.83  1.1e-16  
        32  2^5              -        replayed          16         20    1.22  2.3e-16  
        64  2^6              -        replayed          24         37    1.33  2.2e-16  
       128  2^7              -        replayed          54         56    1.02  3.7e-16  
       256  2^8              -        replayed         107        101    0.94  5.6e-16  
       512  2^9              -        replayed         235        204    0.86  6.1e-16  
      1024  2^10             -        replayed         482        434    0.90  5.6e-16  
      2048  2^11             -        replayed        1020       1047    1.02  7.2e-16  
      4096  2^12             -        replayed        2159       2395    1.11  6.7e-16  
      8192  2^13             -        replayed        4614       5915    0.95  8.9e-16  flips differ 1.35x
     16384  2^14             -        replayed        9747      12929    1.32  8.9e-16  
     32768  2^15             -        replayed       20261      27503    1.34  9.7e-16  
     65536  2^16             -        replayed       46539      59816    1.27  1.0e-15  
    131072  2^17             -        replayed      108177     144708    1.24  1.1e-15  
    262144  2^18             -        replayed      257413     414607    1.61  1.2e-15  
    524288  2^19             -        replayed      613762     896549    1.40  1.2e-15  
   1048576  2^20             -        replayed     2312200    2180574    0.93  1.6e-15  
   2097152  2^21             -        replayed     5634925    5898487    1.04  1.8e-15  
   4194304  2^22             -        replayed    13334988   14741850    1.10  1.7e-15  
   8388608  2^23             -        replayed    30871987   33267644    1.02  1.7e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           22      0      8   0.90   1.03   1.34    1.09
 ALL         22      0      8   0.90   1.03   1.34    1.09
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   1   0.98      1      0
 8..31                  2   0.89      2      0
 32..127                2   1.28      0      0
 128..511               2   0.98      1      0
 512..2047              2   0.88      2      0
 2048..8191             2   1.06      0      0
 8192..32767            2   1.14      1      0
 32768..131071          2   1.31      0      0
 131072..524287         2   1.42      0      0
 524288..2097151        2   1.17      1      0
 2097152..8388607       2   1.07      0      0
 8388608..8388608       1   1.02      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            22   1.03      8   1.09
```


flip agreement: our two readings more than 25% apart at 1 of 22 cells.

worst 10: 16 (- 0.83), 512 (- 0.86), 1024 (- 0.90), 1048576 (- 0.93), 256 (- 0.94), 8192 (- 0.95), 8 (- 0.96), 4 (- 0.98), 128 (- 1.02), 2048 (- 1.02)
best 5: 262144 (- 1.61), 524288 (- 1.40), 32768 (- 1.34), 64 (- 1.33), 16384 (- 1.32)
