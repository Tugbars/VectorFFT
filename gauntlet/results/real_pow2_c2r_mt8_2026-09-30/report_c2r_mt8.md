# gauntlet report

run: `real_pow2_c2r_mt8_2026-09-30`  contract file suffix: `_c2r_mt8`  cells: 23 listed, 22 benched, comparator: MKL

control cell: 2 readings, 1.260..1.326 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     refused            -          -       -        -  not benched
         4  2^2              -        raced              4         13    3.43  0.0e+00  
         8  2^3              -        raced              6         14    2.49  1.7e-16  
        16  2^4              -        raced             12         17    1.40  1.1e-16  
        32  2^5              -        raced             16         22    1.36  2.3e-16  
        64  2^6              -        raced             24         34    1.41  3.3e-16  
       128  2^7              -        raced             51         60    1.17  4.5e-16  
       256  2^8              -        raced            111        127    1.15  5.6e-16  
       512  2^9              -        raced            225        259    1.15  5.6e-16  
      1024  2^10             -        raced            488        535    1.10  7.2e-16  
      2048  2^11             -        raced           1022       1242    1.22  8.9e-16  
      4096  2^12             -        raced           2137       2702    1.26  8.3e-16  
      8192  2^13             -        raced           3823       6261    1.52  8.9e-16  
     16384  2^14             -        raced           4995      12274    1.97  8.9e-16  
     32768  2^15             -        raced          10942      19460    1.51  1.0e-15  
     65536  2^16             -        raced          16797      31823    1.69  1.0e-15  
    131072  2^17             -        raced          27043      54658    1.79  1.1e-15  
    262144  2^18             -        raced          56033     109013    1.75  1.1e-15  
    524288  2^19             -        raced         116750     185912    1.59  1.2e-15  
   1048576  2^20             -        raced         370625     378693    0.96  1.2e-15  
   2097152  2^21             -        raced        1005887    1208400    1.13  1.3e-15  
   4194304  2^22             -        raced        2713675    3650100    1.34  1.3e-15  
   8388608  2^23             -        raced        7720925    9699225    1.26  1.7e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           22      0      1   1.13   1.38   1.97    1.46
 ALL         22      0      1   1.13   1.38   1.97    1.46
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   1   3.43      0      0
 8..31                  2   1.95      0      0
 32..127                2   1.39      0      0
 128..511               2   1.16      0      0
 512..2047              2   1.12      0      0
 2048..8191             2   1.24      0      0
 8192..32767            2   1.75      0      0
 32768..131071          2   1.60      0      0
 131072..524287         2   1.77      0      0
 524288..2097151        2   1.28      1      0
 2097152..8388607       2   1.24      0      0
 8388608..8388608       1   1.26      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            22   1.38      1   1.46
```


flip agreement: our two readings more than 25% apart at 0 of 22 cells.

worst 10: 1048576 (- 0.96), 1024 (- 1.10), 2097152 (- 1.13), 256 (- 1.15), 512 (- 1.15), 128 (- 1.17), 2048 (- 1.22), 8388608 (- 1.26), 4096 (- 1.26), 4194304 (- 1.34)
best 5: 4 (- 3.43), 8 (- 2.49), 16384 (- 1.97), 131072 (- 1.79), 262144 (- 1.75)
