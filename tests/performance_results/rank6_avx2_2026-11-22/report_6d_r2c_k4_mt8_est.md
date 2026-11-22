# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D r2c interleaved, natural, out of place, K=4 contiguous_r2c_k4_mt8_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 13.016..14.877

threaded plans that ran serial (engaged = 0 at both flips): 2: 2x4x2x4x2x4, 3x3x3x3x3x3


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               4246      32856    5.53     10.9  0.0e+00
     2x4x6x8x10x12  2            -      -              48700     977155   16.11    146.6  4.4e-16
       3x3x3x3x3x3  3            -      -               8739      60353    5.25      7.9  5.2e-16
       3x5x7x7x5x3  3            -      -              29891     541771   13.72     49.5  4.1e-16
       4x4x4x4x4x4  2^2          -      -               7942     204404   22.26     61.9  0.0e+00
      4x4x4x4x4x53  2^2          -      -             202067     873094    3.41     42.2  5.8e-16
       4x4x4x8x8x8  2^2          -      -              29647     576716   16.83    165.8  3.2e-16
       5x5x5x5x5x5  5            -      -              33000     387072   11.31     66.0  4.3e-16
       6x6x6x6x6x6  2.3          -      -              51800     813305   13.37    139.7  6.0e-16
       8x8x8x4x4x4  2^3          -      -              41620     913650   21.28    118.1  2.6e-16
       8x8x8x8x8x8  2^3          -      -             199962    2942124   13.44    236.0  3.4e-16
       9x9x9x9x9x9  3^2          -      -             978050    5367243    5.15    103.3  7.6e-16
 10x10x10x10x10x10  2.5          -      -            1839650    8178768    4.29    108.3  6.4e-16
     12x10x8x6x4x2  2^2.3        -      -             166610    1970715   11.33     42.8  5.4e-16
     16x4x4x4x4x16  2^4          -      -              56163     740050   11.37    186.7  3.5e-16
      53x4x4x4x4x4  53           -      -             104267    2089294   17.67     81.9  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               16      0      0   4.29  12.37  21.28   10.37
 ALL                             16      0      0   4.29  12.37  21.28   10.37
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   3.41  14.77  22.26   11.78
 prime column                     4      0      0   5.25  12.51  17.67   10.96
 even column                      3      0      0   4.29  11.33  13.37    8.66
 odd column                       1      0      0   5.15   5.15   5.15    5.15
 ALL                             16      0      0   4.29  12.37  21.28   10.37
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0  11.31  13.55  21.28   12.50
 > 65536 points                   3      0      0   4.29   5.15  13.44    6.67
 257..1024                        2      0      0   5.25   5.39   5.53    5.39
 1025..4096                       1      0      0  22.26  22.26  22.26   22.26
 ALL                             16      0      0   4.29  12.37  21.28   10.37
```


worst 10: 4x4x4x4x4x53 (- 3.41), 10x10x10x10x10x10 (- 4.29), 9x9x9x9x9x9 (- 5.15), 3x3x3x3x3x3 (- 5.25), 2x4x2x4x2x4 (- 5.53), 5x5x5x5x5x5 (- 11.31), 12x10x8x6x4x2 (- 11.33), 16x4x4x4x4x16 (- 11.37), 6x6x6x6x6x6 (- 13.37), 8x8x8x8x8x8 (- 13.44)
best 5: 4x4x4x4x4x4 (- 22.26), 8x8x8x4x4x4 (- 21.28), 53x4x4x4x4x4 (- 17.67), 4x4x4x8x8x8 (- 16.83), 2x4x6x8x10x12 (- 16.11)
