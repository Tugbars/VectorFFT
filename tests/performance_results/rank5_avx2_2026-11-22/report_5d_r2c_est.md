# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D r2c interleaved, natural, out of place, K=1_r2c_est  cells: 21 listed, 21 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 11.992..12.022


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            child  -              26592     186081    6.89     46.2  3.1e-16
     3x5x7x9x11  3            child  -              14706     100296    6.79     23.6  6.1e-16
      4x4x4x4x4  2^2          child  -               1075      29884   27.58     23.8  0.0e+00
    4x4x4x4x256  2^2          child  -              50125     213316    4.23     52.3  3.7e-16
      5x5x5x5x5  5            child  -               5935      61937   10.41     15.3  2.9e-16
      6x6x6x6x6  2.3          child  -               7389      97914   13.15     34.0  5.8e-16
      8x8x8x8x8  2^3          child  -              23218     280627   12.03     52.9  2.7e-16
     8x8x8x8x53  2^3          child  -             595989    2189819    3.63     16.1  7.1e-16
     8x8x53x8x8  2^3          child  -             373194    2707686    7.25     25.8  6.2e-16
      9x9x9x9x9  3^2          child  -              91933     459998    5.00     25.5  6.9e-16
 10x10x10x10x10  2.5          child  -              87177     585615    6.71     47.6  6.4e-16
 12x12x12x12x12  2^2.3        child  -             218725    1248797    5.71     51.0  7.2e-16
     15x9x7x5x3  3.5          child  -              32668     294025    8.99     15.0  5.5e-16
 15x15x15x15x15  3.5          child  -            1525738    2798975    1.83     24.3  8.0e-16
   16x16x16x8x8  2^4          child  -             352500    1848353    5.24     33.5  3.6e-16
 16x16x16x16x16  2^4          child  -            1352075    4061400    2.93     38.8  3.5e-16
   20x16x12x8x4  2^2.5        child  -             196747    1618222    8.20     26.4  4.2e-16
  24x12x6x12x24  2^3.3        child  -             653725    2166175    3.28     36.0  7.4e-16
    32x16x8x4x2  2^5          child  -             134192    1026191    7.64      9.2  3.6e-16
 32x16x16x16x16  2^5          child  -            3233075    8491724    2.60     34.1  3.9e-16
     53x8x8x8x8  53           child  -             360689    2779508    7.65     26.7  6.7e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           21      0      0   2.93   6.79  12.03    6.22
 ALL                             21      0      0   2.93   6.79  12.03    6.22
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      0      0   2.93   6.07  27.58    6.15
 even column                      5      0      0   3.28   6.71  13.15    6.70
 prime column                     3      0      0   6.79   7.65  10.41    8.15
 odd column                       3      0      0   1.83   5.00   8.99    4.35
 ALL                             21      0      0   2.93   6.79  12.03    6.22
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  11      0      0   2.60   5.24   7.65    4.50
 4097..65536                      8      0      0   4.23   7.26  13.15    7.57
 257..1024                        1      0      0  27.58  27.58  27.58   27.58
 1025..4096                       1      0      0  10.41  10.41  10.41   10.41
 ALL                             21      0      0   2.93   6.79  12.03    6.22
```


worst 10: 15x15x15x15x15 (child 1.83), 32x16x16x16x16 (child 2.60), 16x16x16x16x16 (child 2.93), 24x12x6x12x24 (child 3.28), 8x8x8x8x53 (child 3.63), 4x4x4x4x256 (child 4.23), 9x9x9x9x9 (child 5.00), 16x16x16x8x8 (child 5.24), 12x12x12x12x12 (child 5.71), 10x10x10x10x10 (child 6.71)
best 5: 4x4x4x4x4 (child 27.58), 6x6x6x6x6 (child 13.15), 8x8x8x8x8 (child 12.03), 5x5x5x5x5 (child 10.41), 15x9x7x5x3 (child 8.99)
