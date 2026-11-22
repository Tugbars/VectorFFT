# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D r2c interleaved, natural, out of place, K=8 contiguous_r2c_k8_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 23.003..32.820

threaded plans that ran serial (engaged = 0 at both flips): 0


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -               2162      23017    8.15      8.3  0.0e+00
  2x2x2x2x2x2x53  2            -      -              16832      84091    3.05     47.3  4.9e-16
   2x3x4x5x6x7x8  2            -      -              52737    1068862   15.95    233.9  5.1e-16
   2x4x8x2x4x8x2  2            -      -              29273     674578   17.13     72.8  2.2e-16
   3x3x3x3x3x3x3  3            -      -              13096     173983   11.62     37.1  4.9e-16
   3x5x7x3x5x7x3  3            -      -              75762    1500550   13.05    131.1  5.5e-16
   4x4x4x4x4x4x4  2^2          -      -              22700     749466   22.79    202.1  0.0e+00
   4x4x4x4x8x8x8  2^2          -      -             121900    2641162   17.55    365.6  2.9e-16
   5x5x5x5x5x5x5  5            -      -             170538    2435119    9.70    148.9  4.3e-16
   6x6x6x6x6x6x6  2.3          -      -             712687    5635012    7.24    142.1  7.7e-16
   8x8x8x8x8x8x4  2^3          -      -            4708250   31293262    6.13     89.1  3.9e-16
   8x8x8x8x8x8x8  2^3          -      -           10593225   35439631    2.68     83.1  3.3e-16
 10x8x6x4x6x8x10  2.5          -      -            3510525   13677212    3.74    104.0  5.5e-16
  53x2x2x2x2x2x2  53           -      -              43253     499312    9.27     18.4  6.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   3.05   9.48  17.55    8.79
 ALL                             14      0      0   3.05   9.48  17.55    8.79
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   2.68  12.05  22.79    9.04
 prime column                     4      0      0   9.27  10.66  13.05   10.81
 even column                      2      0      0   3.74   5.49   7.24    5.21
 ALL                             14      0      0   3.05   9.48  17.55    8.79
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   2.68   6.69  17.55    6.50
 4097..65536                      4      0      0  13.05  16.54  22.79   16.88
 1025..4096                       3      0      0   3.05   9.27  11.62    6.90
 <= 256 points                    1      0      0   8.15   8.15   8.15    8.15
 ALL                             14      0      0   3.05   9.48  17.55    8.79
```


worst 10: 8x8x8x8x8x8x8 (- 2.68), 2x2x2x2x2x2x53 (- 3.05), 10x8x6x4x6x8x10 (- 3.74), 8x8x8x8x8x8x4 (- 6.13), 6x6x6x6x6x6x6 (- 7.24), 2x2x2x2x2x2x2 (- 8.15), 53x2x2x2x2x2x2 (- 9.27), 5x5x5x5x5x5x5 (- 9.70), 3x3x3x3x3x3x3 (- 11.62), 3x5x7x3x5x7x3 (- 13.05)
best 5: 4x4x4x4x4x4x4 (- 22.79), 4x4x4x4x8x8x8 (- 17.55), 2x4x8x2x4x8x2 (- 17.13), 2x3x4x5x6x7x8 (- 15.95), 3x5x7x3x5x7x3 (- 13.05)
