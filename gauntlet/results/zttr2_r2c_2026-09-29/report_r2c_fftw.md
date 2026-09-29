# gauntlet report

run: `zttr2_r2c_2026-09-29`  contract file suffix: `_r2c_fftw`  cells: 16 listed, 16 benched, comparator: FFTW

control cell: 4 readings, 0.824..1.157 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                -        replayed          10          4    0.38  0.0e+00  
         4  2^2              -        raced             11          4    0.39  0.0e+00  
         8  2^3              -        raced             12          6    0.51  7.4e-17  
        16  2^4              -        raced             12         10    0.84  1.2e-17  
        32  2^5              -        raced             18         18    0.99  1.2e-16  
        64  2^6              -        raced             27         34    1.27  1.6e-16  
       128  2^7              -        raced             52         56    1.06  2.3e-16  
       256  2^8              -        raced            102        102    0.77  2.9e-16  flips differ 1.29x
       512  2^9              -        raced            221        209    0.88  2.1e-16  
      1024  2^10             -        raced            422        427    0.76  3.4e-16  flips differ 1.33x
      2048  2^11             -        raced            946       1029    1.08  3.0e-16  
      4096  2^12             -        raced           2088       2403    1.15  3.4e-16  
      8192  2^13             -        raced           4512       5554    1.23  3.6e-16  
     16384  2^14             -        raced           9648      12942    1.33  3.9e-16  
     32768  2^15             -        raced          21139      27083    1.27  3.5e-16  
     65536  2^16             -        raced          45890      58728    1.27  4.0e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           16      5      8   0.39   1.03   1.27    0.88
 ALL         16      5      8   0.39   1.03   1.27    0.88
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   2   0.38      2      2
 8..31                  2   0.68      2      1
 32..127                2   1.13      1      0
 128..511               2   0.92      1      1
 512..2047              2   0.82      2      1
 2048..8191             2   1.11      0      0
 8192..32767            2   1.28      0      0
 32768..65536           2   1.27      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            16   1.03      8   0.88
```


flip agreement: our two readings more than 25% apart at 2 of 16 cells.

worst 10: 2 (- 0.38), 4 (- 0.39), 8 (- 0.51), 1024 (- 0.76), 256 (- 0.77), 16 (- 0.84), 512 (- 0.88), 32 (- 0.99), 128 (- 1.06), 2048 (- 1.08)
best 5: 16384 (- 1.33), 65536 (- 1.27), 32768 (- 1.27), 64 (- 1.27), 8192 (- 1.23)
