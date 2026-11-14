# gauntlet report

run: `dct1_dst1_1d_2026-11-14`  contract file suffix: `_dst1_fftw`  cells: 24 listed, 24 benched, comparator: FFTW

control cell: 8 readings, 6.771..6.874 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        63  3^2.7            -        replayed          65        121    1.85  1.9e-16  
        64  2^6              -        replayed          85        202    2.37  2.7e-16  
        65  5.13             -        replayed          82        187    2.28  3.0e-16  
       100  2^2.5^2          -        replayed         290       3255   11.20  5.1e-16  
       255  3.5.17           -        replayed         238        385    1.61  2.0e-16  
       256  2^8              -        replayed         752       5558    7.39  3.6e-16  
       257  257              -        replayed         479       2915    6.08  3.8e-16  
      1000  2^3.5^3          -        replayed        2013       5119    2.52  3.9e-16  
      1023  3.11.31          -        replayed        1307       1991    1.51  3.1e-16  
      1024  2^10             -        replayed        2314      15454    6.67  3.1e-16  
      1025  5^2.41           -        replayed        2082       6868    3.24  3.9e-16  
      4095  3^2.5.7.13       -        replayed        6081       8551    1.38  3.2e-16  
      4096  2^12             -        replayed       16882     114651    6.78  5.9e-16  
      4097  17.241           -        replayed       28389      90446    3.18  5.5e-16  
     10000  2^4.5^4          -        replayed       80560     671111    8.30  5.0e-16  
     16383  3.43.127         -        replayed       27118      38947    1.42  3.4e-16  
     16385  5.29.113         -        replayed      130652     320898    2.43  6.1e-16  
     65535  3.5.17.257       -        replayed      155587     181223    1.12  3.7e-16  
     65536  2^16             -        replayed      429700    2014869    4.64  5.0e-16  
     65537  65537            -        replayed      371815    1703668    4.57  6.0e-16  
    100000  2^5.5^5          -        replayed     1249297    4153886    3.24  5.6e-16  
    262143  3^3.7.19.73      -        replayed      885033    1107573    1.22  3.5e-16  
    262145  5.13.37.109      -        replayed     4713313    9453636    1.95  1.3e-15  
   1048575  3.5^2.11.31.41   -        replayed     5521950    6064450    1.07  4.3e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           24      0      0   1.22   2.48   7.39    2.89
 ALL         24      0      0   1.22   2.48   7.39    2.89
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                4   2.32      0      0
 128..511               3   6.08      0      0
 512..2047              4   2.88      0      0
 2048..8191             3   3.18      0      0
 8192..32767            3   2.43      0      0
 32768..131071          4   3.90      0      0
 131072..524287         2   1.58      0      0
 524288..1048575        1   1.07      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                               19   2.28      0   2.48
 pow2                                             5   6.67      0   5.16
```


flip agreement: our two readings more than 25% apart at 0 of 24 cells.

worst 10: 1048575 (- 1.07), 65535 (- 1.12), 262143 (- 1.22), 4095 (- 1.38), 16383 (- 1.42), 1023 (- 1.51), 255 (- 1.61), 63 (- 1.85), 262145 (- 1.95), 65 (- 2.28)
best 5: 100 (- 11.20), 10000 (- 8.30), 256 (- 7.39), 4096 (- 6.78), 1024 (- 6.67)
