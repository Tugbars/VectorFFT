# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2r interleaved, natural, out of place, K=4 contiguous_c2r_k4_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 23.097..23.553


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -               5125      66757   12.81      1.7  2.2e-16
  2x2x2x2x2x2x53  2            -      -              61292     283300    4.51      6.5  1.3e-15
   2x3x4x5x6x7x8  2            -      -             269483    3953979   14.66     22.9  1.2e-15
   2x4x8x2x4x8x2  2            -      -             129448    2458731   18.84      8.2  5.6e-16
   3x3x3x3x3x3x3  3            -      -              44694     517675   11.47      5.4  1.2e-15
   3x5x7x3x5x7x3  3            -      -             589507    5540250    9.36      8.4  1.3e-15
   4x4x4x4x4x4x4  2^2          -      -             109093    2542285   22.60     21.0  5.6e-16
   4x4x4x4x8x8x8  2^2          -      -             731712    9674106   13.04     30.5  7.8e-16
   5x5x5x5x5x5x5  5            -      -            1405250    8035619    5.71      9.0  1.3e-15
   6x6x6x6x6x6x6  2.3          -      -            1951175   19105706    9.65     26.0  1.9e-15
   8x8x8x8x8x8x4  2^3          -      -            9995425   88085812    8.42     21.0  1.0e-15
   8x8x8x8x8x8x8  2^3          -      -           18339175   98136393    5.31     24.0  1.0e-15
 10x8x6x4x6x8x10  2.5          -      -            6766938   42788875    6.30     27.0  1.5e-15
  53x2x2x2x2x2x2  53           -      -             156329    1592491   10.13      2.5  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   5.31   9.89  18.84    9.84
 ALL                             14      0      0   5.31   9.89  18.84    9.84
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   4.51  12.93  22.60   10.97
 prime column                     4      0      0   5.71   9.75  11.47    8.88
 even column                      2      0      0   6.30   7.98   9.65    7.80
 ALL                             14      0      0   5.31   9.89  18.84    9.84
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   5.31   7.36  13.04    7.66
 4097..65536                      4      0      0   9.36  16.75  22.60   15.55
 1025..4096                       3      0      0   4.51  10.13  11.47    8.07
 <= 256 points                    1      0      0  12.81  12.81  12.81   12.81
 ALL                             14      0      0   5.31   9.89  18.84    9.84
```


worst 10: 2x2x2x2x2x2x53 (- 4.51), 8x8x8x8x8x8x8 (- 5.31), 5x5x5x5x5x5x5 (- 5.71), 10x8x6x4x6x8x10 (- 6.30), 8x8x8x8x8x8x4 (- 8.42), 3x5x7x3x5x7x3 (- 9.36), 6x6x6x6x6x6x6 (- 9.65), 53x2x2x2x2x2x2 (- 10.13), 3x3x3x3x3x3x3 (- 11.47), 2x2x2x2x2x2x2 (- 12.81)
best 5: 4x4x4x4x4x4x4 (- 22.60), 2x4x8x2x4x8x2 (- 18.84), 2x3x4x5x6x7x8 (- 14.66), 4x4x4x4x8x8x8 (- 13.04), 2x2x2x2x2x2x2 (- 12.81)
