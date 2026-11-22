# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D r2c interleaved, natural, out of place, K=1_r2c_mt4_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 16.813..18.186

threaded plans that ran serial (engaged = 0 at both flips): 5: 2x2x2x2x2x2x2, 2x2x2x2x2x2x53, 2x3x4x5x6x7x8, 2x4x8x2x4x8x2, 3x3x3x3x3x3x3


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            child  -               1210      12017    9.71      1.9  0.0e+00
  2x2x2x2x2x2x53  2            child  -              12948      41007    3.01      7.7  4.7e-16
   2x3x4x5x6x7x8  2            child  -              45110     477830   10.43     34.2  5.1e-16
   2x4x8x2x4x8x2  2            child  -              26613     284527   10.49     10.0  2.2e-16
   3x3x3x3x3x3x3  3            child  -               6751      52002    7.67      9.0  5.8e-16
   3x5x7x3x5x7x3  3            child  -              37361     512824    8.05     33.2  6.0e-16
   4x4x4x4x4x4x4  2^2          child  -              10477     202302   17.93     54.7  0.0e+00
   4x4x4x4x8x8x8  2^2          child  -              43480     643616   14.56    128.1  2.4e-16
   5x5x5x5x5x5x5  5            child  -              75882     904887   11.90     41.8  4.5e-16
   6x6x6x6x6x6x6  2.3          child  -             140100    1749721   12.20     90.4  7.9e-16
   8x8x8x8x8x8x4  2^3          child  -             478387    6198512   12.49    109.6  3.4e-16
   8x8x8x8x8x8x8  2^3          child  -             842237    6967800    7.21    130.7  3.7e-16
 10x8x6x4x6x8x10  2.5          child  -             350663    3253487    9.20    130.2  5.8e-16
  53x2x2x2x2x2x2  53           child  -              12515     138658    9.84      7.9  1.1e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           14      0      0   7.21  10.13  14.56    9.65
 ALL                             14      0      0   7.21  10.13  14.56    9.65
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   3.01  10.46  17.93    9.65
 prime column                     4      0      0   7.67   8.94  11.90    9.22
 even column                      2      0      0   9.20  10.70  12.20   10.59
 ALL                             14      0      0   7.21  10.13  14.56    9.65
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   7.21  12.05  14.56   10.98
 4097..65536                      4      0      0   8.05  10.46  17.93   11.21
 1025..4096                       3      0      0   3.01   7.67   9.84    6.10
 <= 256 points                    1      0      0   9.71   9.71   9.71    9.71
 ALL                             14      0      0   7.21  10.13  14.56    9.65
```


worst 10: 2x2x2x2x2x2x53 (child 3.01), 8x8x8x8x8x8x8 (child 7.21), 3x3x3x3x3x3x3 (child 7.67), 3x5x7x3x5x7x3 (child 8.05), 10x8x6x4x6x8x10 (child 9.20), 2x2x2x2x2x2x2 (child 9.71), 53x2x2x2x2x2x2 (child 9.84), 2x3x4x5x6x7x8 (child 10.43), 2x4x8x2x4x8x2 (child 10.49), 5x5x5x5x5x5x5 (child 11.90)
best 5: 4x4x4x4x4x4x4 (child 17.93), 4x4x4x4x8x8x8 (child 14.56), 8x8x8x8x8x8x4 (child 12.49), 6x6x6x6x6x6x6 (child 12.20), 5x5x5x5x5x5x5 (child 11.90)
