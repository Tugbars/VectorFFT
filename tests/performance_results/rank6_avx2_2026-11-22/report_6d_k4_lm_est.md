# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2c interleaved, natural, out of place, K=4 lane-major_k4_lm_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 8.315..8.578


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            lm     -               3696      72547   19.57     24.9  3.2e-16
     2x4x6x8x10x12  2            lm     -             310980    2517820    7.87     45.9  5.4e-16
       3x3x3x3x3x3  3            lm     -               5610      90816   16.17     24.7  5.4e-16
       3x5x7x7x5x3  3            lm     -              87293     963556   11.02     33.9  5.4e-16
       4x4x4x4x4x4  2^2          lm     -              26381     364586   13.62     37.3  2.5e-16
      4x4x4x4x4x53  2^2          lm     -             862078    3152472    3.19     19.8  1.1e-15
       4x4x4x8x8x8  2^2          lm     -             201813    1808060    8.69     48.7  5.4e-16
       5x5x5x5x5x5  5            lm     -             106494    1020890    9.50     40.9  4.4e-16
       6x6x6x6x6x6  2.3          lm     -             337440    2807555    7.94     42.9  8.4e-16
       8x8x8x4x4x4  2^3          lm     -             242060    2842013   11.66     40.6  3.3e-16
       8x8x8x8x8x8  2^3          lm     -            2063725   17234812    8.26     45.7  1.2e-15
       9x9x9x9x9x9  3^2          lm     -            5870375   46117787    7.84     34.4  8.8e-16
 10x10x10x10x10x10  2.5          lm     -           12741650   97150643    7.62     31.3  6.0e-16
     12x10x8x6x4x2  2^2.3        lm     -             509910    6969775   13.65     28.0  5.1e-16
     16x4x4x4x4x16  2^4          lm     -             510212    4092456    7.97     41.1  5.2e-16
      53x4x4x4x4x4  53           lm     -             700922    5658172    8.05     24.4  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              16      0      0   7.62   8.47  16.17    9.43
 ALL                             16      0      0   7.62   8.47  16.17    9.43
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   3.19   8.47  19.57    9.04
 prime column                     4      0      0   8.05  10.26  16.17   10.81
 even column                      3      0      0   7.62   7.94  13.65    9.38
 odd column                       1      0      0   7.84   7.84   7.84    7.84
 ALL                             16      0      0   7.62   8.47  16.17    9.43
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   7.87   8.37  13.65    8.45
 > 65536 points                   3      0      0   7.62   7.84   8.26    7.90
 257..1024                        2      0      0  16.17  17.87  19.57   17.79
 1025..4096                       1      0      0  13.62  13.62  13.62   13.62
 ALL                             16      0      0   7.62   8.47  16.17    9.43
```


worst 10: 4x4x4x4x4x53 (lm 3.19), 10x10x10x10x10x10 (lm 7.62), 9x9x9x9x9x9 (lm 7.84), 2x4x6x8x10x12 (lm 7.87), 6x6x6x6x6x6 (lm 7.94), 16x4x4x4x4x16 (lm 7.97), 53x4x4x4x4x4 (lm 8.05), 8x8x8x8x8x8 (lm 8.26), 4x4x4x8x8x8 (lm 8.69), 5x5x5x5x5x5 (lm 9.50)
best 5: 2x4x2x4x2x4 (lm 19.57), 3x3x3x3x3x3 (lm 16.17), 12x10x8x6x4x2 (lm 13.65), 4x4x4x4x4x4 (lm 13.62), 8x8x8x4x4x4 (lm 11.66)
