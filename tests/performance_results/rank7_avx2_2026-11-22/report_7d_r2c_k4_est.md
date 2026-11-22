# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D r2c interleaved, natural, out of place, K=4 contiguous_r2c_k4_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 31.815..31.946


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -               4788      67690   14.03      1.9  0.0e+00
  2x2x2x2x2x2x53  2            -      -              52962     289472    5.36      7.5  5.1e-16
   2x3x4x5x6x7x8  2            -      -             179817    3956179   21.95     34.3  5.4e-16
   2x4x8x2x4x8x2  2            -      -             107425    2472140   22.96      9.9  2.2e-16
   3x3x3x3x3x3x3  3            -      -              27405     544147   19.80      8.9  5.3e-16
   3x5x7x3x5x7x3  3            -      -             291673    5668157   19.40     17.0  5.0e-16
   4x4x4x4x4x4x4  2^2          -      -              85067    2714548   31.77     27.0  0.0e+00
   4x4x4x4x8x8x8  2^2          -      -             472825    9847056   20.69     47.1  2.4e-16
   5x5x5x5x5x5x5  5            -      -             682462    8780018   12.80     18.6  3.6e-16
   6x6x6x6x6x6x6  2.3          -      -            1403475   19979062   14.05     36.1  8.4e-16
   8x8x8x8x8x8x4  2^3          -      -            7412062   94428975   12.55     28.3  3.1e-16
   8x8x8x8x8x8x8  2^3          -      -           14132837  108811231    7.65     31.2  4.3e-16
 10x8x6x4x6x8x10  2.5          -      -            5172850   44099812    8.32     35.3  5.8e-16
  53x2x2x2x2x2x2  53           -      -             145906    1695999   11.14      2.7  8.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   7.65  14.04  22.96   14.32
 ALL                             14      0      0   7.65  14.04  22.96   14.32
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   5.36  17.36  31.77   14.87
 prime column                     4      0      0  11.14  16.10  19.80   15.30
 even column                      2      0      0   8.32  11.18  14.05   10.81
 ALL                             14      0      0   7.65  14.04  22.96   14.32
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   7.65  12.67  20.69   11.99
 4097..65536                      4      0      0  19.40  22.46  31.77   23.61
 1025..4096                       3      0      0   5.36  11.14  19.80   10.57
 <= 256 points                    1      0      0  14.03  14.03  14.03   14.03
 ALL                             14      0      0   7.65  14.04  22.96   14.32
```


worst 10: 2x2x2x2x2x2x53 (- 5.36), 8x8x8x8x8x8x8 (- 7.65), 10x8x6x4x6x8x10 (- 8.32), 53x2x2x2x2x2x2 (- 11.14), 8x8x8x8x8x8x4 (- 12.55), 5x5x5x5x5x5x5 (- 12.80), 2x2x2x2x2x2x2 (- 14.03), 6x6x6x6x6x6x6 (- 14.05), 3x5x7x3x5x7x3 (- 19.40), 3x3x3x3x3x3x3 (- 19.80)
best 5: 4x4x4x4x4x4x4 (- 31.77), 2x4x8x2x4x8x2 (- 22.96), 2x3x4x5x6x7x8 (- 21.95), 4x4x4x4x8x8x8 (- 20.69), 3x3x3x3x3x3x3 (- 19.80)
