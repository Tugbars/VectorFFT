# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D r2c interleaved, natural, out of place, K=1_r2c_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 14.296..14.906

threaded plans that ran serial (engaged = 0 at both flips): 5: 2x2x2x2x2x2x2, 2x2x2x2x2x2x53, 2x3x4x5x6x7x8, 2x4x8x2x4x8x2, 3x3x3x3x3x3x3


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            child  -               1235      11850    9.50      1.8  0.0e+00
  2x2x2x2x2x2x53  2            child  -              12751      40652    3.03      7.8  4.7e-16
   2x3x4x5x6x7x8  2            child  -              46829     474273   10.06     32.9  5.1e-16
   2x4x8x2x4x8x2  2            child  -              26685     286716   10.48     10.0  2.2e-16
   3x3x3x3x3x3x3  3            child  -               6749      52738    7.42      9.0  5.8e-16
   3x5x7x3x5x7x3  3            child  -              40519     463724   10.59     30.6  6.0e-16
   4x4x4x4x4x4x4  2^2          child  -              12949     191942   14.05     44.3  0.0e+00
   4x4x4x4x8x8x8  2^2          child  -              45290     648516   14.30    123.0  2.4e-16
   5x5x5x5x5x5x5  5            child  -              53614     488469    8.54     59.2  4.5e-16
   6x6x6x6x6x6x6  2.3          child  -              82700     938442    9.39    153.1  7.9e-16
   8x8x8x8x8x8x4  2^3          child  -             252550    3839362   13.95    207.6  3.4e-16
   8x8x8x8x8x8x8  2^3          child  -             445287    4212775    8.87    247.3  3.7e-16
 10x8x6x4x6x8x10  2.5          child  -             248487    2478494    8.88    183.7  5.8e-16
  53x2x2x2x2x2x2  53           child  -               8237      98505    9.41     12.1  1.1e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           14      0      0   7.42   9.45  14.05    9.36
 ALL                             14      0      0   7.42   9.45  14.05    9.36
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   3.03  10.27  14.30    9.65
 prime column                     4      0      0   7.42   8.98  10.59    8.92
 even column                      2      0      0   8.88   9.13   9.39    9.13
 ALL                             14      0      0   7.42   9.45  14.05    9.36
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   8.54   9.13  14.30   10.39
 4097..65536                      4      0      0  10.06  10.54  14.05   11.19
 1025..4096                       3      0      0   3.03   7.42   9.41    5.96
 <= 256 points                    1      0      0   9.50   9.50   9.50    9.50
 ALL                             14      0      0   7.42   9.45  14.05    9.36
```


worst 10: 2x2x2x2x2x2x53 (child 3.03), 3x3x3x3x3x3x3 (child 7.42), 5x5x5x5x5x5x5 (child 8.54), 8x8x8x8x8x8x8 (child 8.87), 10x8x6x4x6x8x10 (child 8.88), 6x6x6x6x6x6x6 (child 9.39), 53x2x2x2x2x2x2 (child 9.41), 2x2x2x2x2x2x2 (child 9.50), 2x3x4x5x6x7x8 (child 10.06), 2x4x8x2x4x8x2 (child 10.48)
best 5: 4x4x4x4x8x8x8 (child 14.30), 4x4x4x4x4x4x4 (child 14.05), 8x8x8x8x8x8x4 (child 13.95), 3x5x7x3x5x7x3 (child 10.59), 2x4x8x2x4x8x2 (child 10.48)
