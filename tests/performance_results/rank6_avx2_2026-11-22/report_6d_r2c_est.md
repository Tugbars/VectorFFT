# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D r2c interleaved, natural, out of place, K=1_r2c_est  cells: 17 listed, 17 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 11.945..12.159


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            child  -                981      28943   29.25     11.7  0.0e+00
     2x4x6x8x10x12  2            child  -              40762     608559   14.89     43.8  4.4e-16
       3x3x3x3x3x3  3            child  -               2056      38661   18.80      8.4  5.9e-16
       3x5x7x7x5x3  3            child  -              24569     374254   15.19     15.1  5.2e-16
       4x4x4x4x4x4  2^2          child  -               4757     144208   30.23     25.8  0.0e+00
      4x4x4x4x4x53  2^2          child  -             172170     722602    4.09     12.4  7.7e-16
       4x4x4x8x8x8  2^2          child  -              25585     488961   19.04     48.0  2.8e-16
       5x5x5x5x5x5  5            child  -              31715     372839   11.75     17.2  4.1e-16
       6x6x6x6x6x6  2.3          child  -              49796     706620   14.18     36.3  6.6e-16
       8x8x8x4x4x4  2^3          child  -              40024     882120   22.04     30.7  2.7e-16
       8x8x8x8x8x8  2^3          child  -             223993    2721680   12.13     52.7  3.5e-16
       9x9x9x9x9x9  3^2          child  -             921763    5015825    5.43     27.4  7.3e-16
 10x10x10x10x10x10  2.5          child  -            1076387    7119306    6.44     46.3  6.2e-16
     12x10x8x6x4x2  2^2.3        child  -             191064    1874264    9.81      9.3  5.3e-16
     16x4x4x4x4x16  2^4          child  -              61313     634274   10.34     42.8  3.7e-16
 16x16x16x16x16x16  2^4          child  -           47708500   93118137    1.94     21.1  4.2e-16
      53x4x4x4x4x4  53           child  -             125696    2014324   15.93     17.0  7.8e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           17      0      0   4.09  14.18  29.25   11.69
 ALL                             17      0      0   4.09  14.18  29.25   11.69
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      9      0      0   1.94  14.89  30.23   12.09
 prime column                     4      0      0  11.75  15.56  18.80   15.21
 even column                      3      0      0   6.44   9.81  14.18    9.64
 odd column                       1      0      0   5.43   5.43   5.43    5.43
 ALL                             17      0      0   4.09  14.18  29.25   11.69
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   9.81  14.53  22.04   12.64
 > 65536 points                   4      0      0   1.94   5.94  12.13    5.36
 257..1024                        2      0      0  18.80  24.03  29.25   23.45
 1025..4096                       1      0      0  30.23  30.23  30.23   30.23
 ALL                             17      0      0   4.09  14.18  29.25   11.69
```


worst 10: 16x16x16x16x16x16 (child 1.94), 4x4x4x4x4x53 (child 4.09), 9x9x9x9x9x9 (child 5.43), 10x10x10x10x10x10 (child 6.44), 12x10x8x6x4x2 (child 9.81), 16x4x4x4x4x16 (child 10.34), 5x5x5x5x5x5 (child 11.75), 8x8x8x8x8x8 (child 12.13), 6x6x6x6x6x6 (child 14.18), 2x4x6x8x10x12 (child 14.89)
best 5: 4x4x4x4x4x4 (child 30.23), 2x4x2x4x2x4 (child 29.25), 8x8x8x4x4x4 (child 22.04), 4x4x4x8x8x8 (child 19.04), 3x3x3x3x3x3 (child 18.80)
