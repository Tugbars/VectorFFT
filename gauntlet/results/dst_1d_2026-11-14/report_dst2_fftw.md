# gauntlet report

run: `dst_1d_2026-11-14`  contract file suffix: `_dst2_fftw`  cells: 26 listed, 26 benched, comparator: FFTW

control cell: 4 readings, 2.117..2.144 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        64  2^6              -        replayed          41         79    1.79  2.0e-16  
       100  2^2.5^2          -        replayed          66        161    2.42  4.2e-16  
       125  5^3              -        replayed         136        260    1.90  3.1e-16  
       128  2^7              -        replayed          75        159    2.13  2.4e-16  
       256  2^8              -        replayed         145        369    2.53  2.6e-16  
       512  2^9              -        replayed         294        790    2.63  2.3e-16  
      1000  2^3.5^3          -        replayed         809       1913    2.28  4.4e-16  
      1001  7.11.13          -        replayed         962       2630    2.72  5.0e-16  
      1024  2^10             -        replayed         688       1810    2.62  3.2e-16  
      1536  2^9.3            -        replayed        1245       2594    2.07  4.0e-16  
      2048  2^11             -        replayed        1747       3698    2.09  4.8e-16  
      3125  5^5              -        replayed        4371       8097    1.80  4.6e-16  
      3840  2^8.3.5          -        replayed        3701       8347    2.22  4.1e-16  
      4096  2^12             -        replayed        3812       8175    2.08  3.3e-16  
      8192  2^13             -        replayed        8011      18094    2.25  3.6e-16  
     10000  2^4.5^4          -        replayed       12277      25063    1.91  4.6e-16  
     10007  10007            -        replayed       51615     175751    3.34  8.4e-16  
     16384  2^14             -        replayed       16906      39690    2.34  3.7e-16  
     32768  2^15             -        replayed       34882      83248    2.37  4.1e-16  
     65536  2^16             -        replayed       78020     186860    2.32  4.1e-16  
     65537  65537            -        replayed      590059     883669    1.49  6.7e-16  
    100000  2^5.5^5          -        replayed      152149     346968    2.26  3.9e-16  
    131072  2^17             -        replayed      179750     563575    2.91  3.9e-16  
    262144  2^18             -        replayed      417753    1183290    2.53  4.7e-16  
    524288  2^19             -        replayed     1012450    2858031    2.52  4.7e-16  
   1048576  2^20             -        replayed     2671925    6801562    2.53  5.8e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           26      0      0   1.80   2.30   2.72    2.28
 ALL         26      0      0   1.80   2.30   2.72    2.28
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                3   1.90      0      0
 128..511               2   2.33      0      0
 512..2047              5   2.62      0      0
 2048..8191             4   2.08      0      0
 8192..32767            4   2.29      0      0
 32768..131071          4   2.29      0      0
 131072..524287         2   2.72      0      0
 524288..1048576        2   2.53      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            15   2.37      0   2.36
 -                                               11   2.22      0   2.17
```


flip agreement: our two readings more than 25% apart at 0 of 26 cells.

worst 10: 65537 (- 1.49), 64 (- 1.79), 3125 (- 1.80), 125 (- 1.90), 10000 (- 1.91), 1536 (- 2.07), 4096 (- 2.08), 2048 (- 2.09), 128 (- 2.13), 3840 (- 2.22)
best 5: 10007 (- 3.34), 131072 (- 2.91), 1001 (- 2.72), 512 (- 2.63), 1024 (- 2.62)
