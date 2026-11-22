# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D r2c interleaved, natural, out of place, K=4 contiguous_r2c_k4_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 29.795..31.841

threaded plans that ran serial (engaged = 0 at both flips): 1: 2x2x2x2x2x2x2


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -               4836      19257    3.85      1.9  0.0e+00
  2x2x2x2x2x2x53  2            -      -              16456      75215    4.24     24.2  5.1e-16
   2x3x4x5x6x7x8  2            -      -              48258    1030779   15.54    127.8  5.4e-16
   2x4x8x2x4x8x2  2            -      -              29169     647837   21.28     36.5  2.2e-16
   3x3x3x3x3x3x3  3            -      -               9625     151265   14.93     25.2  5.3e-16
   3x5x7x3x5x7x3  3            -      -              76400    1455693   18.02     65.0  5.0e-16
   4x4x4x4x4x4x4  2^2          -      -              22590     717660   22.89    101.5  0.0e+00
   4x4x4x4x8x8x8  2^2          -      -             169775    2505562   14.11    131.2  2.4e-16
   5x5x5x5x5x5x5  5            -      -             165787    2300618   13.70     76.6  3.6e-16
   6x6x6x6x6x6x6  2.3          -      -             317537    5146481   15.82    159.5  8.4e-16
   8x8x8x8x8x8x4  2^3          -      -            1969362   24987606    9.15    106.5  3.1e-16
   8x8x8x8x8x8x8  2^3          -      -            5623875   31685637    4.44     78.3  4.3e-16
 10x8x6x4x6x8x10  2.5          -      -            1529712   12104843    7.79    119.4  5.8e-16
  53x2x2x2x2x2x2  53           -      -              42422     468442   10.04      9.4  8.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   4.24  13.90  21.28   10.87
 ALL                             14      0      0   4.24  13.90  21.28   10.87
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   3.85  11.63  22.89    9.58
 prime column                     4      0      0  10.04  14.31  18.02   13.87
 even column                      2      0      0   7.79  11.81  15.82   11.10
 ALL                             14      0      0   4.24  13.90  21.28   10.87
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   4.44  11.42  15.82    9.94
 4097..65536                      4      0      0  15.54  19.65  22.89   19.22
 1025..4096                       3      0      0   4.24  10.04  14.93    8.60
 <= 256 points                    1      0      0   3.85   3.85   3.85    3.86
 ALL                             14      0      0   4.24  13.90  21.28   10.87
```


worst 10: 2x2x2x2x2x2x2 (- 3.85), 2x2x2x2x2x2x53 (- 4.24), 8x8x8x8x8x8x8 (- 4.44), 10x8x6x4x6x8x10 (- 7.79), 8x8x8x8x8x8x4 (- 9.15), 53x2x2x2x2x2x2 (- 10.04), 5x5x5x5x5x5x5 (- 13.70), 4x4x4x4x8x8x8 (- 14.11), 3x3x3x3x3x3x3 (- 14.93), 2x3x4x5x6x7x8 (- 15.54)
best 5: 4x4x4x4x4x4x4 (- 22.89), 2x4x8x2x4x8x2 (- 21.28), 3x5x7x3x5x7x3 (- 18.02), 6x6x6x6x6x6x6 (- 15.82), 2x3x4x5x6x7x8 (- 15.54)
