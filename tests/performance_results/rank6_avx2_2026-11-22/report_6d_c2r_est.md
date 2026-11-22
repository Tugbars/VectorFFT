# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2r interleaved, natural, out of place, K=1_c2r_est  cells: 17 listed, 17 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 7.165..7.500


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            child  -                975      27363   28.02     11.8  3.3e-16
     2x4x6x8x10x12  2            child  -              48701     617154   12.52     36.6  1.1e-15
       3x3x3x3x3x3  3            child  -               3197      36500   11.22      5.4  1.0e-15
       3x5x7x7x5x3  3            child  -              48539     347878    7.12      7.6  1.1e-15
       4x4x4x4x4x4  2^2          child  -               5812     133320   22.82     21.1  4.4e-16
      4x4x4x4x4x53  2^2          child  -             215373     708881    3.09      9.9  1.4e-15
       4x4x4x8x8x8  2^2          child  -              34617     478063   13.61     35.5  6.7e-16
       5x5x5x5x5x5  5            child  -              64405     334656    5.17      8.4  1.1e-15
       6x6x6x6x6x6  2.3          child  -              59918     669306   11.15     30.2  1.6e-15
       8x8x8x4x4x4  2^3          child  -              50386     801654   15.27     24.4  7.8e-16
       8x8x8x8x8x8  2^3          child  -             330353    2498257    7.46     35.7  1.0e-15
       9x9x9x9x9x9  3^2          child  -            1461150    4594274    3.14     17.3  2.2e-15
 10x10x10x10x10x10  2.5          child  -            1374975    6909081    5.02     36.2  1.6e-15
     12x10x8x6x4x2  2^2.3        child  -             226224    1607336    7.09      7.9  1.0e-15
     16x4x4x4x4x16  2^4          child  -              73456     601291    7.91     35.7  7.8e-16
 16x16x16x16x16x16  2^4          child  -           49584513  114897187    2.30     20.3  1.3e-15
      53x4x4x4x4x4  53           child  -             130834    1863561   13.75     16.3  1.9e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           17      0      0   3.09   7.91  22.82    8.38
 ALL                             17      0      0   3.09   7.91  22.82    8.38
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      9      0      0   2.30  12.52  28.02    9.61
 prime column                     4      0      0   5.17   9.17  13.75    8.68
 even column                      3      0      0   5.02   7.09  11.15    7.35
 odd column                       1      0      0   3.14   3.14   3.14    3.14
 ALL                             17      0      0   3.09   7.91  22.82    8.38
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   5.17   9.53  15.27    8.72
 > 65536 points                   4      0      0   2.30   4.08   7.46    4.05
 257..1024                        2      0      0  11.22  19.62  28.02   17.73
 1025..4096                       1      0      0  22.82  22.82  22.82   22.81
 ALL                             17      0      0   3.09   7.91  22.82    8.38
```


worst 10: 16x16x16x16x16x16 (child 2.30), 4x4x4x4x4x53 (child 3.09), 9x9x9x9x9x9 (child 3.14), 10x10x10x10x10x10 (child 5.02), 5x5x5x5x5x5 (child 5.17), 12x10x8x6x4x2 (child 7.09), 3x5x7x7x5x3 (child 7.12), 8x8x8x8x8x8 (child 7.46), 16x4x4x4x4x16 (child 7.91), 6x6x6x6x6x6 (child 11.15)
best 5: 2x4x2x4x2x4 (child 28.02), 4x4x4x4x4x4 (child 22.82), 8x8x8x4x4x4 (child 15.27), 53x4x4x4x4x4 (child 13.75), 4x4x4x8x8x8 (child 13.61)
