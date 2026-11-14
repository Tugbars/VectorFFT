# gauntlet report

run: `dst_1d_2026-11-14`  contract file suffix: `_dst3_fftw`  cells: 26 listed, 26 benched, comparator: FFTW

control cell: 4 readings, 2.313..2.348 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        64  2^6              -        replayed          42         81    1.89  2.0e-16  
       100  2^2.5^2          -        replayed          70        160    2.21  3.4e-16  
       125  5^3              -        replayed         135        265    1.95  6.0e-16  
       128  2^7              -        replayed          77        154    1.98  1.9e-16  
       256  2^8              -        replayed         161        370    2.25  2.7e-16  
       512  2^9              -        replayed         335        764    2.24  2.6e-16  
      1000  2^3.5^3          -        replayed        1010       1847    1.81  3.8e-16  
      1001  7.11.13          -        replayed        1056       2590    2.44  5.0e-16  
      1024  2^10             -        replayed         768       1767    2.25  3.2e-16  
      1536  2^9.3            -        replayed        1221       2483    2.03  3.4e-16  
      2048  2^11             -        replayed        1680       3530    2.09  3.1e-16  
      3125  5^5              -        replayed        3879       7624    1.96  4.5e-16  
      3840  2^8.3.5          -        replayed        3210       7837    2.43  4.4e-16  
      4096  2^12             -        replayed        3307       7816    2.33  3.5e-16  
      8192  2^13             -        replayed        6762      17039    2.50  3.9e-16  
     10000  2^4.5^4          -        replayed       13026      23906    1.82  4.3e-16  
     10007  10007            -        replayed       52273     174426    3.31  6.8e-16  
     16384  2^14             -        replayed       13897      38077    2.72  4.1e-16  
     32768  2^15             -        replayed       29046      79082    2.71  3.6e-16  
     65536  2^16             -        replayed       84048     179837    2.12  4.5e-16  
     65537  65537            -        replayed      584462     873895    1.44  6.3e-16  
    100000  2^5.5^5          -        replayed      167421     334400    1.96  4.2e-16  
    131072  2^17             -        replayed      201400     578278    2.83  3.5e-16  
    262144  2^18             -        replayed      472560    1222773    2.40  4.7e-16  
    524288  2^19             -        replayed     1232113    2884394    2.25  4.7e-16  
   1048576  2^20             -        replayed     3130100    6700869    2.11  4.3e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           26      0      0   1.82   2.23   2.72    2.20
 ALL         26      0      0   1.82   2.23   2.72    2.20
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                3   1.95      0      0
 128..511               2   2.11      0      0
 512..2047              5   2.24      0      0
 2048..8191             4   2.21      0      0
 8192..32767            4   2.61      0      0
 32768..131071          4   2.04      0      0
 131072..524287         2   2.62      0      0
 524288..1048576        2   2.18      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            15   2.25      0   2.30
 -                                               11   1.96      0   2.08
```


flip agreement: our two readings more than 25% apart at 0 of 26 cells.

worst 10: 65537 (- 1.44), 1000 (- 1.81), 10000 (- 1.82), 64 (- 1.89), 125 (- 1.95), 100000 (- 1.96), 3125 (- 1.96), 128 (- 1.98), 1536 (- 2.03), 2048 (- 2.09)
best 5: 10007 (- 3.31), 131072 (- 2.83), 16384 (- 2.72), 32768 (- 2.71), 8192 (- 2.50)
