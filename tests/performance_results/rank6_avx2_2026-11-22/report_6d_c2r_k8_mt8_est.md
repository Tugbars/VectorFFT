# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2r interleaved, natural, out of place, K=8 contiguous_c2r_k8_mt8_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 4.916..6.201

threaded plans that ran serial (engaged = 0 at both flips): 0


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               2226      29084   10.22     41.4  4.4e-16
     2x4x6x8x10x12  2            -      -              57887    1061856    8.56    246.6  1.1e-15
       3x3x3x3x3x3  3            -      -               5648      39319    5.62     24.6  1.2e-15
       3x5x7x7x5x3  3            -      -              62868     427404    6.16     47.1  1.2e-15
       4x4x4x4x4x4  2^2          -      -               9993     141116    9.75     98.4  4.4e-16
      4x4x4x4x4x53  2^2          -      -             271875     920143    2.11     62.8  1.5e-15
       4x4x4x8x8x8  2^2          -      -              48588     797262    9.75    202.3  6.7e-16
       5x5x5x5x5x5  5            -      -              73213     528740    3.84     59.5  1.2e-15
       6x6x6x6x6x6  2.3          -      -              69525     988956    7.51    208.2  1.6e-15
       8x8x8x4x4x4  2^3          -      -              63000    1116125   11.21    156.0  6.7e-16
       8x8x8x8x8x8  2^3          -      -             497363    2979025    5.55    189.7  8.9e-16
       9x9x9x9x9x9  3^2          -      -            3003625    5587112    1.78     67.3  2.4e-15
 10x10x10x10x10x10  2.5          -      -            3771187   10661381    2.79    105.7  1.6e-15
     12x10x8x6x4x2  2^2.3        -      -             240825    1646506    4.90     59.3  9.5e-16
     16x4x4x4x4x16  2^4          -      -              91625     934081    6.34    228.9  6.7e-16
      53x4x4x4x4x4  53           -      -             155000    1903619    9.91    110.1  1.6e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               16      0      0   2.11   6.25  10.22    5.80
 ALL                             16      0      0   2.11   6.25  10.22    5.80
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   2.11   9.16  11.21    7.16
 prime column                     4      0      0   3.84   5.89   9.91    6.02
 even column                      3      0      0   2.79   4.90   7.51    4.68
 odd column                       1      0      0   1.78   1.78   1.78    1.78
 ALL                             16      0      0   2.11   6.25  10.22    5.80
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   3.84   6.92  11.21    6.36
 > 65536 points                   3      0      0   1.78   2.79   5.55    3.02
 257..1024                        2      0      0   5.62   7.92  10.22    7.58
 1025..4096                       1      0      0   9.75   9.75   9.75    9.75
 ALL                             16      0      0   2.11   6.25  10.22    5.80
```


worst 10: 9x9x9x9x9x9 (- 1.78), 4x4x4x4x4x53 (- 2.11), 10x10x10x10x10x10 (- 2.79), 5x5x5x5x5x5 (- 3.84), 12x10x8x6x4x2 (- 4.90), 8x8x8x8x8x8 (- 5.55), 3x3x3x3x3x3 (- 5.62), 3x5x7x7x5x3 (- 6.16), 16x4x4x4x4x16 (- 6.34), 6x6x6x6x6x6 (- 7.51)
best 5: 8x8x8x4x4x4 (- 11.21), 2x4x2x4x2x4 (- 10.22), 53x4x4x4x4x4 (- 9.91), 4x4x4x8x8x8 (- 9.75), 4x4x4x4x4x4 (- 9.75)
