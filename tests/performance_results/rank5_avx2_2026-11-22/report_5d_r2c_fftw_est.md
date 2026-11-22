# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D r2c interleaved, natural, out of place, K=1_r2c_fftw_est  cells: 21 listed, 21 benched, comparator: FFTW 5D (out of place, MEASURE)

control cell 8x8x8x8x8: 4 readings, 1.583..1.598


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            child  -              27097      32192    1.13     45.3  2.8e-16
     3x5x7x9x11  3            child  -              14819      11576    0.78     23.4  6.1e-16
      4x4x4x4x4  2^2          child  -               1089       1092    0.96     23.5  0.0e+00
    4x4x4x4x256  2^2          child  -              50185      59324    1.15     52.2  3.4e-16
      5x5x5x5x5  5            child  -               5962       3348    0.54     15.2  3.5e-16
      6x6x6x6x6  2.3          child  -               7390       8363    1.13     34.0  5.8e-16
      8x8x8x8x8  2^3          child  -              23284      37101    1.59     52.8  2.7e-16
     8x8x8x8x53  2^3          child  -             600328    1814736    3.00     16.0  6.8e-16
     8x8x53x8x8  2^3          child  -             375017     976650    2.60     25.7  5.9e-16
      9x9x9x9x9  3^2          child  -              90997      73000    0.80     25.7  6.8e-16
 10x10x10x10x10  2.5          child  -              87762     124182    1.28     47.3  5.3e-16
 12x12x12x12x12  2^2.3        child  -             221044     315175    1.42     50.4  8.5e-16
     15x9x7x5x3  3.5          child  -              33021      20264    0.59     14.8  5.7e-16
 15x15x15x15x15  3.5          child  -            1536825    1178375    0.72     24.1  6.0e-16
   16x16x16x8x8  2^4          child  -             354073     466223    1.30     33.3  3.6e-16
 16x16x16x16x16  2^4          child  -            1450850    2221574    1.51     36.1  4.0e-16
   20x16x12x8x4  2^2.5        child  -             200125     231304    1.14     26.0  4.0e-16
  24x12x6x12x24  2^3.3        child  -             669837     907700    1.35     35.2  7.4e-16
    32x16x8x4x2  2^5          child  -             139218      75310    0.53      8.8  2.7e-16
 32x16x16x16x16  2^5          child  -            3808700    5438894    1.40     28.9  3.3e-16
     53x8x8x8x8  53           child  -             363994    1050389    2.53     26.4  6.7e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           21      6      7   0.59   1.15   2.53    1.17
 ALL                             21      6      7   0.59   1.15   2.53    1.17
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      1      2   0.96   1.35   3.00    1.37
 even column                      5      0      0   1.13   1.28   1.42    1.26
 prime column                     3      2      2   0.54   0.78   2.53    1.02
 odd column                       3      3      3   0.59   0.72   0.80    0.70
 ALL                             21      6      7   0.59   1.15   2.53    1.17
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  11      1      1   1.14   1.40   2.60    1.53
 4097..65536                      8      4      4   0.53   0.96   1.59    0.91
 257..1024                        1      0      1   0.96   0.96   0.96    0.96
 1025..4096                       1      1      1   0.54   0.54   0.54    0.54
 ALL                             21      6      7   0.59   1.15   2.53    1.17
```


worst 10: 32x16x8x4x2 (child 0.53), 5x5x5x5x5 (child 0.54), 15x9x7x5x3 (child 0.59), 15x15x15x15x15 (child 0.72), 3x5x7x9x11 (child 0.78), 9x9x9x9x9 (child 0.80), 4x4x4x4x4 (child 0.96), 6x6x6x6x6 (child 1.13), 2x4x8x16x32 (child 1.13), 20x16x12x8x4 (child 1.14)
best 5: 8x8x8x8x53 (child 3.00), 8x8x53x8x8 (child 2.60), 53x8x8x8x8 (child 2.53), 8x8x8x8x8 (child 1.59), 16x16x16x16x16 (child 1.51)
