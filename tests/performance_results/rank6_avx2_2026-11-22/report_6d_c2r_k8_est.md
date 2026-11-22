# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2r interleaved, natural, out of place, K=8 contiguous_c2r_k8_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 6.315..6.790


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               7889     218814   27.71     11.7  4.4e-16
     2x4x6x8x10x12  2            -      -             415463    4953362   11.89     34.4  1.1e-15
       3x3x3x3x3x3  3            -      -              26814     294284   10.84      5.2  1.2e-15
       3x5x7x7x5x3  3            -      -             386645    2789241    7.21      7.7  1.2e-15
       4x4x4x4x4x4  2^2          -      -              46205    1076031   23.20     21.3  4.4e-16
      4x4x4x4x4x53  2^2          -      -            1840287    5692406    2.52      9.3  1.5e-15
       4x4x4x8x8x8  2^2          -      -             298987    3917619   12.83     32.9  6.7e-16
       5x5x5x5x5x5  5            -      -             521127    2698450    5.18      8.4  1.2e-15
       6x6x6x6x6x6  2.3          -      -             500388    5390325   10.57     28.9  1.6e-15
       8x8x8x4x4x4  2^3          -      -             406025    6433987   15.77     24.2  6.7e-16
       8x8x8x8x8x8  2^3          -      -            2984938   20211506    6.35     31.6  8.9e-16
       9x9x9x9x9x9  3^2          -      -           12656638   37267018    2.94     16.0  2.4e-15
 10x10x10x10x10x10  2.5          -      -           14294312   55662200    3.88     27.9  1.6e-15
     12x10x8x6x4x2  2^2.3        -      -            1855562   12864806    6.92      7.7  9.5e-16
     16x4x4x4x4x16  2^4          -      -             611000    4814418    7.87     34.3  6.7e-16
      53x4x4x4x4x4  53           -      -            1057375   14953212   14.07     16.1  1.6e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               16      0      0   2.94   9.22  23.20    8.62
 ALL                             16      0      0   2.94   9.22  23.20    8.62
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   2.52  12.36  27.71   10.87
 prime column                     4      0      0   5.18   9.02  14.07    8.69
 even column                      3      0      0   3.88   6.92  10.57    6.57
 odd column                       1      0      0   2.94   2.94   2.94    2.94
 ALL                             16      0      0   2.94   9.22  23.20    8.62
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   5.18   9.22  15.77    8.44
 > 65536 points                   3      0      0   2.94   3.88   6.35    4.17
 257..1024                        2      0      0  10.84  19.27  27.71   17.33
 1025..4096                       1      0      0  23.20  23.20  23.20   23.20
 ALL                             16      0      0   2.94   9.22  23.20    8.62
```


worst 10: 4x4x4x4x4x53 (- 2.52), 9x9x9x9x9x9 (- 2.94), 10x10x10x10x10x10 (- 3.88), 5x5x5x5x5x5 (- 5.18), 8x8x8x8x8x8 (- 6.35), 12x10x8x6x4x2 (- 6.92), 3x5x7x7x5x3 (- 7.21), 16x4x4x4x4x16 (- 7.87), 6x6x6x6x6x6 (- 10.57), 3x3x3x3x3x3 (- 10.84)
best 5: 2x4x2x4x2x4 (- 27.71), 4x4x4x4x4x4 (- 23.20), 8x8x8x4x4x4 (- 15.77), 53x4x4x4x4x4 (- 14.07), 4x4x4x8x8x8 (- 12.83)
