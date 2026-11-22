# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D r2c interleaved, natural, out of place, K=1_r2c_fftw_est  cells: 14 listed, 14 benched, comparator: FFTW 7D (out of place, MEASURE)

control cell 4x4x4x4x4x4x4: 4 readings, 1.238..1.272


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            child  -               1186        332    0.28      1.9  0.0e+00
  2x2x2x2x2x2x53  2            child  -              12812      28709    2.08      7.8  5.8e-16
   2x3x4x5x6x7x8  2            child  -              44647      57939    1.29     34.5  5.1e-16
   2x4x8x2x4x8x2  2            child  -              26735      15718    0.58     10.0  2.0e-16
   3x3x3x3x3x3x3  3            child  -               6610       3285    0.49      9.2  5.8e-16
   3x5x7x3x5x7x3  3            child  -              73272      49502    0.67     16.9  5.5e-16
   4x4x4x4x4x4x4  2^2          child  -              21289      27002    1.27     26.9  0.0e+00
   4x4x4x4x8x8x8  2^2          child  -             117560     181266    1.54     47.4  2.4e-16
   5x5x5x5x5x5x5  5            child  -             167455     111442    0.66     19.0  5.6e-16
   6x6x6x6x6x6x6  2.3          child  -             349600     493003    1.41     36.2  7.9e-16
   8x8x8x8x8x8x4  2^3          child  -            1514238    2597325    1.67     34.6  3.4e-16
   8x8x8x8x8x8x8  2^3          child  -            3209325    5622406    1.67     34.3  4.5e-16
 10x8x6x4x6x8x10  2.5          child  -             945550    1746600    1.83     48.3  6.4e-16
  53x2x2x2x2x2x2  53           child  -              36227      27166    0.75      2.7  1.1e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           14      6      6   0.49   1.28   1.83    1.00
 ALL                             14      6      6   0.49   1.28   1.83    1.00
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      2      2   0.28   1.41   2.08    1.11
 prime column                     4      4      4   0.49   0.67   0.75    0.64
 even column                      2      0      0   1.41   1.62   1.83    1.61
 ALL                             14      6      6   0.49   1.28   1.83    1.00
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      1      1   0.66   1.60   1.83    1.39
 4097..65536                      4      2      2   0.58   0.97   1.29    0.90
 1025..4096                       3      2      2   0.49   0.75   2.08    0.91
 <= 256 points                    1      1      1   0.28   0.28   0.28    0.28
 ALL                             14      6      6   0.49   1.28   1.83    1.00
```


worst 10: 2x2x2x2x2x2x2 (child 0.28), 3x3x3x3x3x3x3 (child 0.49), 2x4x8x2x4x8x2 (child 0.58), 5x5x5x5x5x5x5 (child 0.66), 3x5x7x3x5x7x3 (child 0.67), 53x2x2x2x2x2x2 (child 0.75), 4x4x4x4x4x4x4 (child 1.27), 2x3x4x5x6x7x8 (child 1.29), 6x6x6x6x6x6x6 (child 1.41), 4x4x4x4x8x8x8 (child 1.54)
best 5: 2x2x2x2x2x2x53 (child 2.08), 10x8x6x4x6x8x10 (child 1.83), 8x8x8x8x8x8x4 (child 1.67), 8x8x8x8x8x8x8 (child 1.67), 4x4x4x4x8x8x8 (child 1.54)
