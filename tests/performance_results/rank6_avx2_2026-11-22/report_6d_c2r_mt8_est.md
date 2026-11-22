# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2r interleaved, natural, out of place, K=1_c2r_mt8_est  cells: 17 listed, 17 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 6.429..11.939

threaded plans that ran serial (engaged = 0 at both flips): 3: 2x4x2x4x2x4, 2x4x6x8x10x12, 3x3x3x3x3x3


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            child  -                941      17442   18.05     12.2  3.3e-16
     2x4x6x8x10x12  2            child  -              50009     264675    4.33     35.7  1.1e-15
       3x3x3x3x3x3  3            child  -               3208      16712    5.03      5.4  1.0e-15
       3x5x7x7x5x3  3            child  -              17719     121100    6.61     20.9  1.1e-15
       4x4x4x4x4x4  2^2          child  -               2892      41225   12.71     42.5  4.4e-16
      4x4x4x4x4x53  2^2          child  -              73058     204759    2.69     29.2  1.4e-15
       4x4x4x8x8x8  2^2          child  -              12946     144772   10.53     94.9  6.7e-16
       5x5x5x5x5x5  5            child  -              15529      82169    4.89     35.0  1.1e-15
       6x6x6x6x6x6  2.3          child  -              18296     133700    6.81     98.9  1.6e-15
       8x8x8x4x4x4  2^3          child  -              10856     130166    9.55    113.2  7.8e-16
       8x8x8x8x8x8  2^3          child  -              55320     493620    7.08    213.2  1.0e-15
       9x9x9x9x9x9  3^2          child  -             333125    1356925    2.77     75.9  2.2e-15
 10x10x10x10x10x10  2.5          child  -             318325    1690331    4.62    156.5  1.6e-15
     12x10x8x6x4x2  2^2.3        child  -              47194     362670    6.23     37.8  1.0e-15
     16x4x4x4x4x16  2^4          child  -              17759     177798    9.02    147.6  7.8e-16
 16x16x16x16x16x16  2^4          child  -           23602125   34026419    1.28     42.6  1.3e-15
      53x4x4x4x4x4  53           child  -              24623     518179   15.70     86.7  1.9e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           17      0      0   2.69   6.61  15.70    6.21
 ALL                             17      0      0   2.69   6.61  15.70    6.21
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      9      0      0   1.28   9.02  18.05    6.54
 prime column                     4      0      0   4.89   5.82  15.70    7.11
 even column                      3      0      0   4.62   6.23   6.81    5.81
 odd column                       1      0      0   2.77   2.77   2.77    2.77
 ALL                             17      0      0   2.69   6.61  15.70    6.21
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   4.33   6.71  15.70    6.85
 > 65536 points                   4      0      0   1.28   3.69   7.08    3.28
 257..1024                        2      0      0   5.03  11.54  18.05    9.53
 1025..4096                       1      0      0  12.71  12.71  12.71   12.71
 ALL                             17      0      0   2.69   6.61  15.70    6.21
```


worst 10: 16x16x16x16x16x16 (child 1.28), 4x4x4x4x4x53 (child 2.69), 9x9x9x9x9x9 (child 2.77), 2x4x6x8x10x12 (child 4.33), 10x10x10x10x10x10 (child 4.62), 5x5x5x5x5x5 (child 4.89), 3x3x3x3x3x3 (child 5.03), 12x10x8x6x4x2 (child 6.23), 3x5x7x7x5x3 (child 6.61), 6x6x6x6x6x6 (child 6.81)
best 5: 2x4x2x4x2x4 (child 18.05), 53x4x4x4x4x4 (child 15.70), 4x4x4x4x4x4 (child 12.71), 4x4x4x8x8x8 (child 10.53), 8x8x8x4x4x4 (child 9.55)
