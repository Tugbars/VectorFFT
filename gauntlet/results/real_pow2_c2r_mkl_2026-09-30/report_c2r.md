# gauntlet report

run: `real_pow2_c2r_mkl_2026-09-30`  contract file suffix: `_c2r`  cells: 23 listed, 22 benched, comparator: MKL

control cell: 4 readings, 1.256..1.267 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     refused            -          -       -        -  not benched
         4  2^2              -        replayed           4         12    3.08  0.0e+00  
         8  2^3              -        replayed           6         13    2.27  1.7e-16  
        16  2^4              -        replayed          12         16    1.35  1.1e-16  
        32  2^5              -        replayed          16         22    1.35  2.3e-16  
        64  2^6              -        replayed          25         34    1.35  3.3e-16  
       128  2^7              -        replayed          54         62    1.15  4.5e-16  
       256  2^8              -        replayed         107        128    1.16  5.6e-16  
       512  2^9              -        replayed         235        260    1.11  5.6e-16  
      1024  2^10             -        replayed         481        535    1.11  7.8e-16  
      2048  2^11             -        replayed        1017       1248    1.22  8.9e-16  
      4096  2^12             -        replayed        2158       2712    1.25  6.7e-16  
      8192  2^13             -        replayed        4623       6268    1.35  8.0e-16  
     16384  2^14             -        replayed        9697      13224    1.35  8.9e-16  
     32768  2^15             -        replayed       20277      28341    1.38  8.9e-16  
     65536  2^16             -        replayed       46608      59042    1.24  1.0e-15  
    131072  2^17             -        raced         107167     165475    1.44  1.1e-15  
    262144  2^18             -        raced         256173     361026    1.41  1.1e-15  
    524288  2^19             -        raced         685337     792962    1.14  1.2e-15  
   1048576  2^20             -        raced        2317675    1806913    0.77  1.2e-15  
   2097152  2^21             -        raced        5581813    4754113    0.84  1.3e-15  
   4194304  2^22             -        raced       13207287   12000475    0.89  1.3e-15  
   8388608  2^23             -        raced       30786450   27711587    0.89  1.4e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           22      1      4   0.89   1.25   1.44    1.26
 ALL         22      1      4   0.89   1.25   1.44    1.26
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   1   3.08      0      0
 8..31                  2   1.81      0      0
 32..127                2   1.35      0      0
 128..511               2   1.16      0      0
 512..2047              2   1.11      0      0
 2048..8191             2   1.24      0      0
 8192..32767            2   1.35      0      0
 32768..131071          2   1.31      0      0
 131072..524287         2   1.42      0      0
 524288..2097151        2   0.96      1      1
 2097152..8388607       2   0.86      2      0
 8388608..8388608       1   0.89      1      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            22   1.25      4   1.26
```


flip agreement: our two readings more than 25% apart at 0 of 22 cells.

worst 10: 1048576 (- 0.77), 2097152 (- 0.84), 4194304 (- 0.89), 8388608 (- 0.89), 512 (- 1.11), 1024 (- 1.11), 524288 (- 1.14), 128 (- 1.15), 256 (- 1.16), 2048 (- 1.22)
best 5: 4 (- 3.08), 8 (- 2.27), 131072 (- 1.44), 262144 (- 1.41), 32768 (- 1.38)
