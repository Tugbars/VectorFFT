# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D r2c interleaved, natural, out of place, K=1_r2c_fftw_est  cells: 17 listed, 17 benched, comparator: FFTW 6D (out of place, MEASURE)

control cell 8x8x8x8x8x8: 4 readings, 1.777..1.819


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            child  -               1012        703    0.69     11.4  0.0e+00
     2x4x6x8x10x12  2            child  -              40979      54641    1.33     43.6  5.4e-16
       3x3x3x3x3x3  3            child  -               2145       1032    0.47      8.1  5.9e-16
       3x5x7x7x5x3  3            child  -              24763      15682    0.63     14.9  4.4e-16
       4x4x4x4x4x4  2^2          child  -               4810       6010    1.23     25.5  0.0e+00
      4x4x4x4x4x53  2^2          child  -             171701     456587    2.58     12.4  7.3e-16
       4x4x4x8x8x8  2^2          child  -              25695      40115    1.56     47.8  3.7e-16
       5x5x5x5x5x5  5            child  -              31799      20069    0.63     17.1  4.6e-16
       6x6x6x6x6x6  2.3          child  -              50148      57401    1.14     36.1  6.9e-16
       8x8x8x4x4x4  2^3          child  -              40705      50642    1.20     30.2  2.0e-16
       8x8x8x8x8x8  2^3          child  -             227433     409373    1.77     51.9  3.5e-16
       9x9x9x9x9x9  3^2          child  -             914675     843656    0.92     27.6  8.3e-16
 10x10x10x10x10x10  2.5          child  -            1180437    1732406    1.43     42.2  5.1e-16
     12x10x8x6x4x2  2^2.3        child  -             198736      95838    0.48      9.0  5.8e-16
     16x4x4x4x4x16  2^4          child  -              61562      86994    1.41     42.6  3.1e-16
 16x16x16x16x16x16  2^4          child  -           55015937   64838831    1.17     18.3  3.9e-16
      53x4x4x4x4x4  53           child  -             126444     304542    2.40     16.9  7.8e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           17      5      6   0.48   1.20   2.40    1.10
 ALL                             17      5      6   0.48   1.20   2.40    1.10
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      9      1      1   0.69   1.33   2.58    1.36
 prime column                     4      3      3   0.47   0.63   2.40    0.82
 even column                      3      1      1   0.48   1.14   1.43    0.92
 odd column                       1      0      1   0.92   0.92   0.92    0.92
 ALL                             17      5      6   0.48   1.20   2.40    1.10
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      3      3   0.63   1.26   2.58    1.17
 > 65536 points                   4      0      1   0.92   1.30   1.77    1.29
 257..1024                        2      2      2   0.47   0.58   0.69    0.57
 1025..4096                       1      0      0   1.23   1.23   1.23    1.23
 ALL                             17      5      6   0.48   1.20   2.40    1.10
```


worst 10: 3x3x3x3x3x3 (child 0.47), 12x10x8x6x4x2 (child 0.48), 5x5x5x5x5x5 (child 0.63), 3x5x7x7x5x3 (child 0.63), 2x4x2x4x2x4 (child 0.69), 9x9x9x9x9x9 (child 0.92), 6x6x6x6x6x6 (child 1.14), 16x16x16x16x16x16 (child 1.17), 8x8x8x4x4x4 (child 1.20), 4x4x4x4x4x4 (child 1.23)
best 5: 4x4x4x4x4x53 (child 2.58), 53x4x4x4x4x4 (child 2.40), 8x8x8x8x8x8 (child 1.77), 4x4x4x8x8x8 (child 1.56), 10x10x10x10x10x10 (child 1.43)
