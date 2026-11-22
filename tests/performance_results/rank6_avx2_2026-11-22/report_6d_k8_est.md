# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2c interleaved, natural, out of place, K=8 contiguous_k8_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 3.005..3.049


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -              11452     123457   10.77     16.1  3.0e-16
     2x4x6x8x10x12  2            -      -             587425    2800337    4.75     48.6  5.5e-16
       3x3x3x3x3x3  3            -      -              19451     149278    7.67     14.3  4.7e-16
       3x5x7x7x5x3  3            -      -             228559    1341527    5.87     25.9  5.0e-16
       4x4x4x4x4x4  2^2          -      -              59656     508658    8.51     33.0  3.3e-16
      4x4x4x4x4x53  2^2          -      -            1921650    3763225    1.87     17.8  1.1e-15
       4x4x4x8x8x8  2^2          -      -             387250    1928875    4.98     50.8  3.4e-16
       5x5x5x5x5x5  5            -      -             272527    1365750    5.01     31.9  4.4e-16
       6x6x6x6x6x6  2.3          -      -             693213    3031400    4.35     41.8  8.4e-16
       8x8x8x4x4x4  2^3          -      -             507900    3644794    7.17     38.7  3.3e-16
       8x8x8x8x8x8  2^3          -      -            4880200   14724869    3.00     38.7  5.9e-16
       9x9x9x9x9x9  3^2          -      -           13090187   33687244    2.57     30.9  8.4e-16
 10x10x10x10x10x10  2.5          -      -           24841050   65162925    2.25     32.1  5.5e-16
     12x10x8x6x4x2  2^2.3        -      -            1146900    9269156    7.97     24.9  5.7e-16
     16x4x4x4x4x16  2^4          -      -             945775    4138537    4.25     44.3  3.0e-16
      53x4x4x4x4x4  53           -      -            1470963    7101619    4.81     23.2  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               16      0      0   2.25   4.89   8.51    4.80
 ALL                             16      0      0   2.25   4.89   8.51    4.80
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   1.87   4.87  10.77    4.97
 prime column                     4      0      0   4.81   5.44   7.67    5.74
 even column                      3      0      0   2.25   4.35   7.97    4.27
 odd column                       1      0      0   2.57   2.57   2.57    2.57
 ALL                             16      0      0   2.25   4.89   8.51    4.80
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   4.25   4.89   7.97    4.81
 > 65536 points                   3      0      0   2.25   2.57   3.00    2.59
 257..1024                        2      0      0   7.67   9.22  10.77    9.09
 1025..4096                       1      0      0   8.51   8.51   8.51    8.51
 ALL                             16      0      0   2.25   4.89   8.51    4.80
```


worst 10: 4x4x4x4x4x53 (- 1.87), 10x10x10x10x10x10 (- 2.25), 9x9x9x9x9x9 (- 2.57), 8x8x8x8x8x8 (- 3.00), 16x4x4x4x4x16 (- 4.25), 6x6x6x6x6x6 (- 4.35), 2x4x6x8x10x12 (- 4.75), 53x4x4x4x4x4 (- 4.81), 4x4x4x8x8x8 (- 4.98), 5x5x5x5x5x5 (- 5.01)
best 5: 2x4x2x4x2x4 (- 10.77), 4x4x4x4x4x4 (- 8.51), 12x10x8x6x4x2 (- 7.97), 3x3x3x3x3x3 (- 7.67), 8x8x8x4x4x4 (- 7.17)
