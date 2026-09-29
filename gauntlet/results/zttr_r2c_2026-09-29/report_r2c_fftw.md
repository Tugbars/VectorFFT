# gauntlet report

run: `zttr_r2c_2026-09-29`  contract file suffix: `_r2c_fftw`  cells: 16 listed, 16 benched, comparator: FFTW

control cell: 4 readings, 1.140..1.161 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                -        replayed          10          4    0.38  0.0e+00  
         4  2^2              -        raced             11          4    0.38  0.0e+00  
         8  2^3              -        raced             12          6    0.51  7.4e-17  
        16  2^4              -        raced             12         10    0.84  1.2e-17  
        32  2^5              -        raced             18         18    0.99  1.2e-16  
        64  2^6              -        raced             31         34    1.12  1.6e-16  
       128  2^7              -        raced             52         56    1.08  2.3e-16  
       256  2^8              -        raced            101        102    1.00  2.9e-16  
       512  2^9              -        raced            202        207    1.02  2.1e-16  
      1024  2^10             -        raced            421        428    1.01  3.4e-16  
      2048  2^11             -        raced            964       1023    1.05  2.8e-16  
      4096  2^12             -        raced           2087       2389    1.14  3.4e-16  
      8192  2^13             -        raced           4515       5530    1.22  3.6e-16  
     16384  2^14             -        raced           9700      12791    1.31  3.9e-16  
     32768  2^15             -        raced          21217      27058    1.27  3.5e-16  
     65536  2^16             -        raced          47126      58653    1.24  3.0e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           16      3      5   0.38   1.03   1.27    0.91
 ALL         16      3      5   0.38   1.03   1.27    0.91
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   2   0.38      2      2
 8..31                  2   0.67      2      1
 32..127                2   1.06      1      0
 128..511               2   1.04      0      0
 512..2047              2   1.02      0      0
 2048..8191             2   1.10      0      0
 8192..32767            2   1.27      0      0
 32768..65536           2   1.25      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            16   1.03      5   0.91
```


flip agreement: our two readings more than 25% apart at 0 of 16 cells.

worst 10: 2 (- 0.38), 4 (- 0.38), 8 (- 0.51), 16 (- 0.84), 32 (- 0.99), 256 (- 1.00), 1024 (- 1.01), 512 (- 1.02), 2048 (- 1.05), 128 (- 1.08)
best 5: 16384 (- 1.31), 32768 (- 1.27), 65536 (- 1.24), 8192 (- 1.22), 4096 (- 1.14)
