# gauntlet report

run: `zttr2_c2r_2026-09-29`  contract file suffix: `_c2r_fftw`  cells: 16 listed, 15 benched, comparator: FFTW

control cell: 4 readings, 1.088..1.096 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     refused            -          -       -        -  not benched
         4  2^2              -        raced             11          4    0.38  0.0e+00  
         8  2^3              -        raced             12          6    0.47  2.2e-16  
        16  2^4              -        raced             13         10    0.75  1.1e-16  
        32  2^5              -        raced             16         20    1.19  2.3e-16  
        64  2^6              -        raced             25         33    1.28  2.2e-16  
       128  2^7              -        raced             55         55    1.01  4.5e-16  
       256  2^8              -        raced            108        101    0.93  5.6e-16  
       512  2^9              -        raced            226        207    0.92  5.6e-16  
      1024  2^10             -        raced            482        435    0.90  5.6e-16  
      2048  2^11             -        raced           1018       1049    1.02  7.2e-16  
      4096  2^12             -        raced           2150       2379    1.10  7.8e-16  
      8192  2^13             -        raced           4614       5918    1.28  7.2e-16  
     16384  2^14             -        raced           9753      12819    1.00  7.8e-16  flips differ 1.33x
     32768  2^15             -        raced          20334      27516    1.22  1.0e-15  
     65536  2^16             -        raced          45695      60513    1.28  1.0e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           15      3      7   0.47   1.01   1.28    0.93
 ALL         15      3      7   0.47   1.01   1.28    0.93
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   1   0.38      1      1
 8..31                  2   0.61      2      2
 32..127                2   1.23      0      0
 128..511               2   0.97      1      0
 512..2047              2   0.91      2      0
 2048..8191             2   1.06      0      0
 8192..32767            2   1.14      1      0
 32768..65536           2   1.25      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            15   1.01      7   0.93
```


flip agreement: our two readings more than 25% apart at 1 of 15 cells.

worst 10: 4 (- 0.38), 8 (- 0.47), 16 (- 0.75), 1024 (- 0.90), 512 (- 0.92), 256 (- 0.93), 16384 (- 1.00), 128 (- 1.01), 2048 (- 1.02), 4096 (- 1.10)
best 5: 65536 (- 1.28), 8192 (- 1.28), 64 (- 1.28), 32768 (- 1.22), 32 (- 1.19)
