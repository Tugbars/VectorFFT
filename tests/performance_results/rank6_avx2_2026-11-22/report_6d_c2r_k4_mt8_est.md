# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2r interleaved, natural, out of place, K=4 contiguous_c2r_k4_mt8_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 10.230..12.066

threaded plans that ran serial (engaged = 0 at both flips): 2: 2x4x2x4x2x4, 3x3x3x3x3x3


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               3944      28901    7.25     11.7  4.4e-16
     2x4x6x8x10x12  2            -      -              53380     725805   12.12    133.7  1.1e-15
       3x3x3x3x3x3  3            -      -              13570      39899    2.94      5.1  9.4e-16
       3x5x7x7x5x3  3            -      -              62042     366015    5.59     23.9  1.2e-15
       4x4x4x4x4x4  2^2          -      -               8231     144648   13.35     59.7  5.6e-16
      4x4x4x4x4x53  2^2          -      -             258511     783900    2.63     33.0  1.5e-15
       4x4x4x8x8x8  2^2          -      -              39700     545040   12.63    123.8  6.7e-16
       5x5x5x5x5x5  5            -      -              71006     362053    4.76     30.7  9.4e-16
       6x6x6x6x6x6  2.3          -      -              63280     763555   10.93    114.4  1.6e-15
       8x8x8x4x4x4  2^3          -      -              62547     848970   13.28     78.6  6.7e-16
       8x8x8x8x8x8  2^3          -      -             254850    2661462    9.88    185.2  8.9e-16
       9x9x9x9x9x9  3^2          -      -            1397013    4869012    2.90     72.4  2.4e-15
 10x10x10x10x10x10  2.5          -      -            1914338    8326600    4.11    104.1  1.7e-15
     12x10x8x6x4x2  2^2.3        -      -             188840    1677230    7.62     37.8  9.5e-16
     16x4x4x4x4x16  2^4          -      -              71287     683794    6.93    147.1  6.7e-16
      53x4x4x4x4x4  53           -      -             103600    2008689   15.59     82.4  1.6e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               16      0      0   2.90   7.43  13.35    7.11
 ALL                             16      0      0   2.90   7.43  13.35    7.11
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   2.63  11.00  13.35    8.78
 prime column                     4      0      0   2.94   5.18  15.59    5.91
 even column                      3      0      0   4.11   7.62  10.93    6.99
 odd column                       1      0      0   2.90   2.90   2.90    2.90
 ALL                             16      0      0   2.90   7.43  13.35    7.11
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   4.76   9.27  15.59    8.14
 > 65536 points                   3      0      0   2.90   4.11   9.88    4.90
 257..1024                        2      0      0   2.94   5.10   7.25    4.62
 1025..4096                       1      0      0  13.35  13.35  13.35   13.35
 ALL                             16      0      0   2.90   7.43  13.35    7.11
```


worst 10: 4x4x4x4x4x53 (- 2.63), 9x9x9x9x9x9 (- 2.90), 3x3x3x3x3x3 (- 2.94), 10x10x10x10x10x10 (- 4.11), 5x5x5x5x5x5 (- 4.76), 3x5x7x7x5x3 (- 5.59), 16x4x4x4x4x16 (- 6.93), 2x4x2x4x2x4 (- 7.25), 12x10x8x6x4x2 (- 7.62), 8x8x8x8x8x8 (- 9.88)
best 5: 53x4x4x4x4x4 (- 15.59), 4x4x4x4x4x4 (- 13.35), 8x8x8x4x4x4 (- 13.28), 4x4x4x8x8x8 (- 12.63), 2x4x6x8x10x12 (- 12.12)
