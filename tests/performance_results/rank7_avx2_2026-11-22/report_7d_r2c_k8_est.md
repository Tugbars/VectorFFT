# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D r2c interleaved, natural, out of place, K=8 contiguous_r2c_k8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 31.738..32.441


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -               9468     137860   14.36      1.9  0.0e+00
  2x2x2x2x2x2x53  2            -      -             108871     590563    5.39      7.3  4.9e-16
   2x3x4x5x6x7x8  2            -      -             361438    8088700   22.21     34.1  5.1e-16
   2x4x8x2x4x8x2  2            -      -             213373    5080203   23.70     10.0  2.2e-16
   3x3x3x3x3x3x3  3            -      -              53580    1117711   20.84      9.1  4.9e-16
   3x5x7x3x5x7x3  3            -      -             587725   11481894   19.34     16.9  5.5e-16
   4x4x4x4x4x4x4  2^2          -      -             170447    5434463   31.88     26.9  0.0e+00
   4x4x4x4x8x8x8  2^2          -      -             947513   19738581   20.82     47.0  2.9e-16
   5x5x5x5x5x5x5  5            -      -            1354738   17601512   12.98     18.7  4.3e-16
   6x6x6x6x6x6x6  2.3          -      -            3265825   40313575   12.32     31.0  7.7e-16
   8x8x8x8x8x8x4  2^3          -      -           15667825  189946394   11.56     26.8  3.9e-16
   8x8x8x8x8x8x8  2^3          -      -           29137138  220216088    7.40     30.2  3.3e-16
 10x8x6x4x6x8x10  2.5          -      -           11882300   88822306    7.24     30.7  5.5e-16
  53x2x2x2x2x2x2  53           -      -             290684    3385075   11.62      2.7  6.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   7.24  13.67  23.70   14.12
 ALL                             14      0      0   7.24  13.67  23.70   14.12
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   5.39  17.59  31.88   14.81
 prime column                     4      0      0  11.62  16.16  20.84   15.70
 even column                      2      0      0   7.24   9.78  12.32    9.44
 ALL                             14      0      0   7.24  13.67  23.70   14.12
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   7.24  11.94  20.82   11.28
 4097..65536                      4      0      0  19.34  22.96  31.88   23.87
 1025..4096                       3      0      0   5.39  11.62  20.84   10.93
 <= 256 points                    1      0      0  14.36  14.36  14.36   14.36
 ALL                             14      0      0   7.24  13.67  23.70   14.12
```


worst 10: 2x2x2x2x2x2x53 (- 5.39), 10x8x6x4x6x8x10 (- 7.24), 8x8x8x8x8x8x8 (- 7.40), 8x8x8x8x8x8x4 (- 11.56), 53x2x2x2x2x2x2 (- 11.62), 6x6x6x6x6x6x6 (- 12.32), 5x5x5x5x5x5x5 (- 12.98), 2x2x2x2x2x2x2 (- 14.36), 3x5x7x3x5x7x3 (- 19.34), 4x4x4x4x8x8x8 (- 20.82)
best 5: 4x4x4x4x4x4x4 (- 31.88), 2x4x8x2x4x8x2 (- 23.70), 2x3x4x5x6x7x8 (- 22.21), 3x3x3x3x3x3x3 (- 20.84), 4x4x4x4x8x8x8 (- 20.82)
