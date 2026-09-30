# gauntlet report

run: `real_pow2_c2r_mt8_2026-09-30`  contract file suffix: `_c2r_mt8`  cells: 23 listed, 22 benched, comparator: MKL

control cell: 2 readings, 1.130..1.209 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     refused            -          -       -        -  not benched
         4  2^2              -        raced              4         14    3.37  0.0e+00  
         8  2^3              -        raced              6         14    2.38  1.7e-16  
        16  2^4              -        raced             12         16    1.38  1.1e-16  
        32  2^5              -        raced             16         25    1.61  2.3e-16  
        64  2^6              -        raced             34         34    1.01  3.3e-16  
       128  2^7              -        raced             51         60    1.16  4.5e-16  
       256  2^8              -        raced            110        127    1.16  5.6e-16  
       512  2^9              -        raced            228        259    1.14  5.6e-16  
      1024  2^10             -        raced            481        533    1.11  7.2e-16  
      2048  2^11             -        raced           1028       1271    1.24  8.9e-16  
      4096  2^12             -        raced           2191       2894    1.32  6.7e-16  
      8192  2^13             -        raced           4740       6293    1.33  7.2e-16  
     16384  2^14             -        raced           7398      13242    1.20  8.9e-16  flips differ 1.35x
     32768  2^15             -        raced          14535      19437    1.11  8.9e-16  
     65536  2^16             -        raced          25920      31672    1.04  9.4e-16  
    131072  2^17             -        raced          38160      58698    1.52  1.1e-15  
    262144  2^18             -        raced          70727     112833    1.54  1.2e-15  
    524288  2^19             -        raced         145925     212325    1.36  1.3e-15  
   1048576  2^20             -        raced         468262     404900    0.86  1.3e-15  
   2097152  2^21             -        raced        1290750    1104425    0.86  1.6e-15  
   4194304  2^22             -        raced        3010912    4514100    1.50  1.3e-15  
   8388608  2^23             -        raced        7620450    9849575    1.29  1.7e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           22      0      2   1.01   1.27   1.61    1.32
 ALL         22      0      2   1.01   1.27   1.61    1.32
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   1   3.37      0      0
 8..31                  2   1.88      0      0
 32..127                2   1.31      0      0
 128..511               2   1.16      0      0
 512..2047              2   1.12      0      0
 2048..8191             2   1.28      0      0
 8192..32767            2   1.27      0      0
 32768..131071          2   1.07      0      0
 131072..524287         2   1.53      0      0
 524288..2097151        2   1.11      1      0
 2097152..8388607       2   1.18      1      0
 8388608..8388608       1   1.29      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            22   1.27      2   1.32
```


flip agreement: our two readings more than 25% apart at 1 of 22 cells.

worst 10: 2097152 (- 0.86), 1048576 (- 0.86), 64 (- 1.01), 65536 (- 1.04), 32768 (- 1.11), 1024 (- 1.11), 512 (- 1.14), 256 (- 1.16), 128 (- 1.16), 16384 (- 1.20)
best 5: 4 (- 3.37), 8 (- 2.38), 32 (- 1.61), 262144 (- 1.54), 131072 (- 1.52)
