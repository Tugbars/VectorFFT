# gauntlet report

run: `real_pow2_r2c_fftw_2026-09-30`  contract file suffix: `_r2c_fftw`  cells: 23 listed, 23 benched, comparator: FFTW

control cell: 4 readings, 1.149..1.194 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                -        replayed           9          4    0.43  0.0e+00  
         4  2^2              -        replayed           4          4    1.12  0.0e+00  
         8  2^3              -        replayed           6          6    1.12  7.4e-17  
        16  2^4              -        replayed          10         10    0.95  4.6e-17  
        32  2^5              -        replayed          18         18    1.02  1.2e-16  
        64  2^6              -        replayed          26         34    1.31  1.6e-16  
       128  2^7              -        replayed          51         56    1.01  2.3e-16  
       256  2^8              -        replayed         100        102    1.01  2.9e-16  
       512  2^9              -        replayed         200        208    1.04  2.1e-16  
      1024  2^10             -        replayed         422        428    1.00  3.4e-16  
      2048  2^11             -        replayed         964       1004    1.04  3.0e-16  
      4096  2^12             -        replayed        2087       2400    1.15  3.1e-16  
      8192  2^13             -        replayed        4508       5559    1.23  3.6e-16  
     16384  2^14             -        replayed        9773      12836    1.31  3.9e-16  
     32768  2^15             -        replayed       21497      26993    1.25  3.5e-16  
     65536  2^16             -        replayed       46030      59350    1.28  3.0e-16  
    131072  2^17             -        replayed      104467     153995    1.44  4.0e-16  
    262144  2^18             -        replayed      259393     374960    1.41  4.7e-16  
    524288  2^19             -        replayed      595425     850700    1.42  4.4e-16  
   1048576  2^20             -        replayed     1843900    2068981    1.08  5.6e-16  
   2097152  2^21             -        replayed     4415225    5902262    1.32  1.4e-15  
   4194304  2^22             -        replayed    10450850   14683425    1.40  3.0e-15  
   8388608  2^23             -        replayed    24266750   35290612    1.44  1.2e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           23      1      2   1.00   1.15   1.42    1.14
 ALL         23      1      2   1.00   1.15   1.42    1.14
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   2   0.78      1      1
 8..31                  2   1.03      1      0
 32..127                2   1.17      0      0
 128..511               2   1.01      0      0
 512..2047              2   1.02      0      0
 2048..8191             2   1.09      0      0
 8192..32767            2   1.27      0      0
 32768..131071          2   1.26      0      0
 131072..524287         2   1.43      0      0
 524288..2097151        2   1.25      0      0
 2097152..8388607       2   1.36      0      0
 8388608..8388608       1   1.44      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            23   1.15      2   1.14
```


flip agreement: our two readings more than 25% apart at 0 of 23 cells.

worst 10: 2 (- 0.43), 16 (- 0.95), 1024 (- 1.00), 128 (- 1.01), 256 (- 1.01), 32 (- 1.02), 512 (- 1.04), 2048 (- 1.04), 1048576 (- 1.08), 8 (- 1.12)
best 5: 131072 (- 1.44), 8388608 (- 1.44), 524288 (- 1.42), 262144 (- 1.41), 4194304 (- 1.40)
