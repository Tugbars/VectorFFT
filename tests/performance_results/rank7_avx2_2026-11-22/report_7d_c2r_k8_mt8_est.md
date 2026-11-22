# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2r interleaved, natural, out of place, K=8 contiguous_c2r_k8_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 14.597..21.388

threaded plans that ran serial (engaged = 0 at both flips): 0


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -               2016      18381    6.66      8.9  3.3e-16
  2x2x2x2x2x2x53  2            -      -              18452     101787    2.92     43.1  1.2e-15
   2x3x4x5x6x7x8  2            -      -              88025     996206   10.45    140.2  1.2e-15
   2x4x8x2x4x8x2  2            -      -              37830     620292   12.29     56.3  6.7e-16
   3x3x3x3x3x3x3  3            -      -              16918     138315    6.43     28.7  1.5e-15
   3x5x7x3x5x7x3  3            -      -             153300    1409962    7.38     64.8  1.3e-15
   4x4x4x4x4x4x4  2^2          -      -              32967     666993   14.76    139.2  5.6e-16
   4x4x4x4x8x8x8  2^2          -      -             197875    2464787   10.62    225.2  8.6e-16
   5x5x5x5x5x5x5  5            -      -             358350    2072950    4.64     70.9  1.1e-15
   6x6x6x6x6x6x6  2.3          -      -             762788    5164569    5.59    132.8  1.9e-15
   8x8x8x8x8x8x4  2^3          -      -            4467713   27199249    5.77     93.9  8.9e-16
   8x8x8x8x8x8x8  2^3          -      -           10903500   34334981    3.00     80.8  1.1e-15
 10x8x6x4x6x8x10  2.5          -      -            3548775   13590212    3.57    102.9  1.6e-15
  53x2x2x2x2x2x2  53           -      -              45434     406634    6.34     17.5  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   3.00   6.39  12.29    6.39
 ALL                             14      0      0   3.00   6.39  12.29    6.39
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   2.92   8.56  14.76    7.14
 prime column                     4      0      0   4.64   6.39   7.38    6.11
 even column                      2      0      0   3.57   4.58   5.59    4.47
 ALL                             14      0      0   3.00   6.39  12.29    6.39
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   3.00   5.12  10.62    5.07
 4097..65536                      4      0      0   7.38  11.37  14.76   10.88
 1025..4096                       3      0      0   2.92   6.34   6.43    4.92
 <= 256 points                    1      0      0   6.66   6.66   6.66    6.66
 ALL                             14      0      0   3.00   6.39  12.29    6.39
```


worst 10: 2x2x2x2x2x2x53 (- 2.92), 8x8x8x8x8x8x8 (- 3.00), 10x8x6x4x6x8x10 (- 3.57), 5x5x5x5x5x5x5 (- 4.64), 6x6x6x6x6x6x6 (- 5.59), 8x8x8x8x8x8x4 (- 5.77), 53x2x2x2x2x2x2 (- 6.34), 3x3x3x3x3x3x3 (- 6.43), 2x2x2x2x2x2x2 (- 6.66), 3x5x7x3x5x7x3 (- 7.38)
best 5: 4x4x4x4x4x4x4 (- 14.76), 2x4x8x2x4x8x2 (- 12.29), 4x4x4x4x8x8x8 (- 10.62), 2x3x4x5x6x7x8 (- 10.45), 3x5x7x3x5x7x3 (- 7.38)
