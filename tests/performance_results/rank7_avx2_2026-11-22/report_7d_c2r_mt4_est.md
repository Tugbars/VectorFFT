# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2r interleaved, natural, out of place, K=1_c2r_mt4_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 14.445..17.665

threaded plans that ran serial (engaged = 0 at both flips): 5: 2x2x2x2x2x2x2, 2x2x2x2x2x2x53, 2x3x4x5x6x7x8, 2x4x8x2x4x8x2, 3x3x3x3x3x3x3


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            child  -               1199      11566    9.44      1.9  3.4e-16
  2x2x2x2x2x2x53  2            child  -              14254      39784    2.71      7.0  1.2e-15
   2x3x4x5x6x7x8  2            child  -              63638     456623    7.13     24.2  1.2e-15
   2x4x8x2x4x8x2  2            child  -              33066     277141    8.25      8.1  6.7e-16
   3x3x3x3x3x3x3  3            child  -              10949      51588    4.67      5.5  1.2e-15
   3x5x7x3x5x7x3  3            child  -              57006     532117    9.19     21.8  1.5e-15
   4x4x4x4x4x4x4  2^2          child  -              10643     187227   17.17     53.9  4.4e-16
   4x4x4x4x8x8x8  2^2          child  -              52063     648486   11.92    107.0  7.8e-16
   5x5x5x5x5x5x5  5            child  -             139992     835804    5.70     22.7  1.3e-15
   6x6x6x6x6x6x6  2.3          child  -             180971    1633471    8.98     70.0  1.8e-15
   8x8x8x8x8x8x4  2^3          child  -             735837    5990318    7.88     71.3  1.1e-15
   8x8x8x8x8x8x8  2^3          child  -            1800013    6536469    3.52     61.2  1.2e-15
 10x8x6x4x6x8x10  2.5          child  -             531013    3241700    5.84     86.0  1.4e-15
  53x2x2x2x2x2x2  53           child  -              12157     195881   14.47      8.2  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           14      0      0   3.52   8.07  14.47    7.44
 ALL                             14      0      0   3.52   8.07  14.47    7.44
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   2.71   8.07  17.17    7.35
 prime column                     4      0      0   4.67   7.44  14.47    7.71
 even column                      2      0      0   5.84   7.41   8.98    7.24
 ALL                             14      0      0   3.52   8.07  14.47    7.44
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   3.52   6.86  11.92    6.80
 4097..65536                      4      0      0   7.13   8.72  17.17    9.81
 1025..4096                       3      0      0   2.71   4.67  14.47    5.68
 <= 256 points                    1      0      0   9.44   9.44   9.44    9.44
 ALL                             14      0      0   3.52   8.07  14.47    7.44
```


worst 10: 2x2x2x2x2x2x53 (child 2.71), 8x8x8x8x8x8x8 (child 3.52), 3x3x3x3x3x3x3 (child 4.67), 5x5x5x5x5x5x5 (child 5.70), 10x8x6x4x6x8x10 (child 5.84), 2x3x4x5x6x7x8 (child 7.13), 8x8x8x8x8x8x4 (child 7.88), 2x4x8x2x4x8x2 (child 8.25), 6x6x6x6x6x6x6 (child 8.98), 3x5x7x3x5x7x3 (child 9.19)
best 5: 4x4x4x4x4x4x4 (child 17.17), 53x2x2x2x2x2x2 (child 14.47), 4x4x4x4x8x8x8 (child 11.92), 2x2x2x2x2x2x2 (child 9.44), 3x5x7x3x5x7x3 (child 9.19)
