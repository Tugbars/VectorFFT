# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2r interleaved, natural, out of place, K=8 contiguous_c2r_k8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 22.748..23.307


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -               9835     133763   13.59      1.8  3.3e-16
  2x2x2x2x2x2x53  2            -      -             112259     566872    4.75      7.1  1.2e-15
   2x3x4x5x6x7x8  2            -      -             531675    7897006   14.75     23.2  1.2e-15
   2x4x8x2x4x8x2  2            -      -             260807    4918625   18.84      8.2  6.7e-16
   3x3x3x3x3x3x3  3            -      -              88925    1035351   11.57      5.5  1.5e-15
   3x5x7x3x5x7x3  3            -      -            1195800   11102193    9.19      8.3  1.3e-15
   4x4x4x4x4x4x4  2^2          -      -             221073    5098150   23.04     20.8  5.6e-16
   4x4x4x4x8x8x8  2^2          -      -            1500562   19415306   12.93     29.7  8.6e-16
   5x5x5x5x5x5x5  5            -      -            2788275   16115700    5.74      9.1  1.1e-15
   6x6x6x6x6x6x6  2.3          -      -            4202825   38362056    9.09     24.1  1.9e-15
   8x8x8x8x8x8x4  2^3          -      -           20068300  176889343    8.80     20.9  8.9e-16
   8x8x8x8x8x8x8  2^3          -      -           37033525  197671349    5.30     23.8  1.1e-15
 10x8x6x4x6x8x10  2.5          -      -           13836763   85732000    6.17     26.4  1.6e-15
  53x2x2x2x2x2x2  53           -      -             303138    3194038   10.52      2.6  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      0      0   5.30   9.85  18.84    9.92
 ALL                             14      0      0   5.30   9.85  18.84    9.92
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   4.75  13.26  23.04   11.21
 prime column                     4      0      0   5.74   9.85  11.57    8.95
 even column                      2      0      0   6.17   7.63   9.09    7.49
 ALL                             14      0      0   5.30   9.85  18.84    9.92
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   5.30   7.49  12.93    7.61
 4097..65536                      4      0      0   9.19  16.79  23.04   15.57
 1025..4096                       3      0      0   4.75  10.52  11.57    8.33
 <= 256 points                    1      0      0  13.59  13.59  13.59   13.59
 ALL                             14      0      0   5.30   9.85  18.84    9.92
```


worst 10: 2x2x2x2x2x2x53 (- 4.75), 8x8x8x8x8x8x8 (- 5.30), 5x5x5x5x5x5x5 (- 5.74), 10x8x6x4x6x8x10 (- 6.17), 8x8x8x8x8x8x4 (- 8.80), 6x6x6x6x6x6x6 (- 9.09), 3x5x7x3x5x7x3 (- 9.19), 53x2x2x2x2x2x2 (- 10.52), 3x3x3x3x3x3x3 (- 11.57), 4x4x4x4x8x8x8 (- 12.93)
best 5: 4x4x4x4x4x4x4 (- 23.04), 2x4x8x2x4x8x2 (- 18.84), 2x3x4x5x6x7x8 (- 14.75), 2x2x2x2x2x2x2 (- 13.59), 4x4x4x4x8x8x8 (- 12.93)
