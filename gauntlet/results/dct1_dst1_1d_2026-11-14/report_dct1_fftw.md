# gauntlet report

run: `dct1_dst1_1d_2026-11-14`  contract file suffix: `_dct1_fftw`  cells: 24 listed, 24 benched, comparator: FFTW

control cell: 8 readings, 2.013..2.044 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        63  3^2.7            -        replayed         139        506    3.51  3.7e-16  
        64  2^6              -        replayed          78        178    2.26  2.3e-16  
        65  5.13             -        replayed          64        109    1.72  1.7e-16  
       100  2^2.5^2          -        replayed         123        309    2.50  3.3e-16  
       255  3.5.17           -        replayed        1071       7538    6.98  4.5e-16  
       256  2^8              -        replayed         373       1866    4.94  4.0e-16  
       257  257              -        replayed         250        431    1.72  1.7e-16  
      1000  2^3.5^3          -        replayed        2084      12744    6.10  3.2e-16  
      1023  3.11.31          -        replayed        3894      19154    4.89  4.6e-16  
      1024  2^10             -        replayed        2472      11305    4.51  4.6e-16  
      1025  5^2.41           -        replayed        1303       2005    1.53  2.2e-16  
      4095  3^2.5.7.13       -        replayed       16991     109081    6.41  5.3e-16  
      4096  2^12             -        replayed        9177      18770    2.02  4.2e-16  
      4097  17.241           -        replayed        6030       8484    1.39  2.9e-16  
     10000  2^4.5^4          -        replayed       44102     338988    7.66  5.2e-16  
     16383  3.43.127         -        replayed      126860     324384    2.55  6.8e-16  
     16385  5.29.113         -        replayed       27696      38648    1.39  3.5e-16  
     65535  3.5.17.257       -        replayed      423900    2876900    6.74  7.5e-16  
     65536  2^16             -        replayed      320533    2259587    6.99  5.4e-16  
     65537  65537            -        replayed      149979     182038    1.17  3.6e-16  
    100000  2^5.5^5          -        replayed      705777    4915907    6.92  6.5e-16  
    262143  3^3.7.19.73      -        replayed     4175707   15049953    3.56  5.8e-16  
    262145  5.13.37.109      -        replayed      882840    1104733    1.23  3.4e-16  
   1048577  17.61681         -        replayed     5448037    6016962    1.10  4.2e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           24      0      0   1.23   3.03   6.98    3.03
 ALL         24      0      0   1.23   3.03   6.98    3.03
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                4   2.38      0      0
 128..511               3   4.94      0      0
 512..2047              4   4.70      0      0
 2048..8191             3   2.02      0      0
 8192..32767            3   2.55      0      0
 32768..131071          4   6.83      0      0
 131072..524287         2   2.39      0      0
 524288..1048577        1   1.10      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                               19   2.55      0   2.88
 pow2                                             5   4.51      0   3.72
```


flip agreement: our two readings more than 25% apart at 0 of 24 cells.

worst 10: 1048577 (- 1.10), 65537 (- 1.17), 262145 (- 1.23), 16385 (- 1.39), 4097 (- 1.39), 1025 (- 1.53), 65 (- 1.72), 257 (- 1.72), 4096 (- 2.02), 64 (- 2.26)
best 5: 10000 (- 7.66), 65536 (- 6.99), 255 (- 6.98), 100000 (- 6.92), 65535 (- 6.74)
