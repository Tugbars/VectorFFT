# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D r2c interleaved, natural, out of place, K=1_r2c_mt8_est  cells: 17 listed, 17 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 6.847..13.815

threaded plans that ran serial (engaged = 0 at both flips): 3: 2x4x2x4x2x4, 2x4x6x8x10x12, 3x3x3x3x3x3


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            child  -                988      18491   18.47     11.7  0.0e+00
     2x4x6x8x10x12  2            child  -              39624     270277    6.58     45.0  4.4e-16
       3x3x3x3x3x3  3            child  -               2092      16356    7.52      8.3  5.9e-16
       3x5x7x7x5x3  3            child  -              13940     124342    8.81     26.6  5.2e-16
       4x4x4x4x4x4  2^2          child  -               3907      41851    9.41     31.4  0.0e+00
      4x4x4x4x4x53  2^2          child  -              58237     190591    3.23     36.6  7.7e-16
       4x4x4x8x8x8  2^2          child  -              16270     140953    8.57     75.5  2.8e-16
       5x5x5x5x5x5  5            child  -              12784      88925    4.51     42.6  4.1e-16
       6x6x6x6x6x6  2.3          child  -              21209     136139    5.89     85.3  6.6e-16
       8x8x8x4x4x4  2^3          child  -              12116     143501   10.14    101.4  2.7e-16
       8x8x8x8x8x8  2^3          child  -              61647     516460    6.52    191.4  3.5e-16
       9x9x9x9x9x9  3^2          child  -             227863    1188662    4.03    110.9  7.3e-16
 10x10x10x10x10x10  2.5          child  -             280287    1485125    4.37    177.8  6.2e-16
     12x10x8x6x4x2  2^2.3        child  -              43792     351851    6.29     40.8  5.3e-16
     16x4x4x4x4x16  2^4          child  -              16105     160007    6.81    162.8  3.7e-16
 16x16x16x16x16x16  2^4          child  -           20755563   23523393    0.99     48.5  4.2e-16
      53x4x4x4x4x4  53           child  -              27319     350692   10.94     78.1  7.8e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           17      0      1   3.23   6.58  10.94    6.21
 ALL                             17      0      1   3.23   6.58  10.94    6.21
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      9      0      1   0.99   6.81  18.47    6.23
 prime column                     4      0      0   4.51   8.16  10.94    7.56
 even column                      3      0      0   4.37   5.89   6.29    5.45
 odd column                       1      0      0   4.03   4.03   4.03    4.03
 ALL                             17      0      1   3.23   6.58  10.94    6.21
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   4.51   6.70  10.94    6.77
 > 65536 points                   4      0      1   0.99   4.20   6.52    3.27
 257..1024                        2      0      0   7.52  12.99  18.47   11.78
 1025..4096                       1      0      0   9.41   9.41   9.41    9.41
 ALL                             17      0      1   3.23   6.58  10.94    6.21
```


worst 10: 16x16x16x16x16x16 (child 0.99), 4x4x4x4x4x53 (child 3.23), 9x9x9x9x9x9 (child 4.03), 10x10x10x10x10x10 (child 4.37), 5x5x5x5x5x5 (child 4.51), 6x6x6x6x6x6 (child 5.89), 12x10x8x6x4x2 (child 6.29), 8x8x8x8x8x8 (child 6.52), 2x4x6x8x10x12 (child 6.58), 16x4x4x4x4x16 (child 6.81)
best 5: 2x4x2x4x2x4 (child 18.47), 53x4x4x4x4x4 (child 10.94), 8x8x8x4x4x4 (child 10.14), 4x4x4x4x4x4 (child 9.41), 3x5x7x7x5x3 (child 8.81)
