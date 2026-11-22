# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2r interleaved, natural, out of place, K=1_c2r_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 22.811..23.508


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            child  -               1261      16551   12.94      1.8  3.4e-16
  2x2x2x2x2x2x53  2            child  -              15220      70769    4.30      6.5  1.2e-15
   2x3x4x5x6x7x8  2            child  -              63310     962636   15.03     24.4  1.2e-15
   2x4x8x2x4x8x2  2            child  -              32083     590286   18.36      8.3  6.7e-16
   3x3x3x3x3x3x3  3            child  -              11071     128589   11.58      5.5  1.2e-15
   3x5x7x3x5x7x3  3            child  -             146741    1340350    9.11      8.5  1.5e-15
   4x4x4x4x4x4x4  2^2          child  -              26959     632911   23.05     21.3  4.4e-16
   4x4x4x4x8x8x8  2^2          child  -             184757    2369008   12.49     30.2  7.8e-16
   5x5x5x5x5x5x5  5            child  -             348531    2002057    5.73      9.1  1.3e-15
   6x6x6x6x6x6x6  2.3          child  -             463650    4743775   10.05     27.3  1.8e-15
   8x8x8x8x8x8x4  2^3          child  -            2206013   21953693    9.54     23.8  1.1e-15
   8x8x8x8x8x8x8  2^3          child  -            4460037   24694375    5.22     24.7  1.2e-15
 10x8x6x4x6x8x10  2.5          child  -            1411125   10702394    7.33     32.4  1.4e-15
  53x2x2x2x2x2x2  53           child  -              38752     399365    9.83      2.6  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           14      0      0   5.22   9.94  18.36    9.97
 ALL                             14      0      0   5.22   9.94  18.36    9.97
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   4.30  12.72  23.05   11.03
 prime column                     4      0      0   5.73   9.47  11.58    8.78
 even column                      2      0      0   7.33   8.69  10.05    8.58
 ALL                             14      0      0   5.22   9.94  18.36    9.97
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   5.22   8.44  12.49    8.00
 4097..65536                      4      0      0   9.11  16.69  23.05   15.51
 1025..4096                       3      0      0   4.30   9.83  11.58    7.88
 <= 256 points                    1      0      0  12.94  12.94  12.94   12.94
 ALL                             14      0      0   5.22   9.94  18.36    9.97
```


worst 10: 2x2x2x2x2x2x53 (child 4.30), 8x8x8x8x8x8x8 (child 5.22), 5x5x5x5x5x5x5 (child 5.73), 10x8x6x4x6x8x10 (child 7.33), 3x5x7x3x5x7x3 (child 9.11), 8x8x8x8x8x8x4 (child 9.54), 53x2x2x2x2x2x2 (child 9.83), 6x6x6x6x6x6x6 (child 10.05), 3x3x3x3x3x3x3 (child 11.58), 4x4x4x4x8x8x8 (child 12.49)
best 5: 4x4x4x4x4x4x4 (child 23.05), 2x4x8x2x4x8x2 (child 18.36), 2x3x4x5x6x7x8 (child 15.03), 2x2x2x2x2x2x2 (child 12.94), 4x4x4x4x8x8x8 (child 12.49)
