# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2r interleaved, natural, out of place, K=1_c2r_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 15.635..17.259

threaded plans that ran serial (engaged = 0 at both flips): 5: 2x2x2x2x2x2x2, 2x2x2x2x2x2x53, 2x3x4x5x6x7x8, 2x4x8x2x4x8x2, 3x3x3x3x3x3x3


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            child  -               1209      11589    9.27      1.9  3.4e-16
  2x2x2x2x2x2x53  2            child  -              14521      40048    2.68      6.8  1.2e-15
   2x3x4x5x6x7x8  2            child  -              62573     456151    6.38     24.6  1.2e-15
   2x4x8x2x4x8x2  2            child  -              32843     274106    8.21      8.1  6.7e-16
   3x3x3x3x3x3x3  3            child  -              10950      51485    4.62      5.5  1.2e-15
   3x5x7x3x5x7x3  3            child  -              51320     454482    8.64     24.2  1.5e-15
   4x4x4x4x4x4x4  2^2          child  -               9987     175333   14.60     57.4  4.4e-16
   4x4x4x4x8x8x8  2^2          child  -              49813     626138   11.57    111.8  7.8e-16
   5x5x5x5x5x5x5  5            child  -              81135     437153    5.30     39.1  1.3e-15
   6x6x6x6x6x6x6  2.3          child  -              87457     876343    8.07    144.8  1.8e-15
   8x8x8x8x8x8x4  2^3          child  -             446638    3517656    7.84    117.4  1.1e-15
   8x8x8x8x8x8x8  2^3          child  -            1336225    3699856    2.67     82.4  1.2e-15
 10x8x6x4x6x8x10  2.5          child  -             345600    2594425    7.25    132.1  1.4e-15
  53x2x2x2x2x2x2  53           child  -               7038     193958   18.80     14.1  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           14      0      0   2.68   7.96  14.60    7.23
 ALL                             14      0      0   2.68   7.96  14.60    7.23
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   2.67   8.03  14.60    6.80
 prime column                     4      0      0   4.62   6.97  18.80    7.94
 even column                      2      0      0   7.25   7.66   8.07    7.65
 ALL                             14      0      0   2.68   7.96  14.60    7.23
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   2.67   7.55  11.57    6.49
 4097..65536                      4      0      0   6.38   8.43  14.60    9.02
 1025..4096                       3      0      0   2.68   4.62  18.80    6.15
 <= 256 points                    1      0      0   9.27   9.27   9.27    9.27
 ALL                             14      0      0   2.68   7.96  14.60    7.23
```


worst 10: 8x8x8x8x8x8x8 (child 2.67), 2x2x2x2x2x2x53 (child 2.68), 3x3x3x3x3x3x3 (child 4.62), 5x5x5x5x5x5x5 (child 5.30), 2x3x4x5x6x7x8 (child 6.38), 10x8x6x4x6x8x10 (child 7.25), 8x8x8x8x8x8x4 (child 7.84), 6x6x6x6x6x6x6 (child 8.07), 2x4x8x2x4x8x2 (child 8.21), 3x5x7x3x5x7x3 (child 8.64)
best 5: 53x2x2x2x2x2x2 (child 18.80), 4x4x4x4x4x4x4 (child 14.60), 4x4x4x4x8x8x8 (child 11.57), 2x2x2x2x2x2x2 (child 9.27), 3x5x7x3x5x7x3 (child 8.64)
