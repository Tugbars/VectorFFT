# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2c interleaved, natural, out of place, K=1_fftw_est  cells: 14 listed, 14 benched, comparator: FFTW 7D (out of place, MEASURE)

control cell 4x4x4x4x4x4x4: 4 readings, 0.905..0.937


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            -      -                896        436    0.48      5.0  1.8e-16
  2x2x2x2x2x2x53  2            -      -              14171      22866    1.58     14.0  5.7e-16
   2x3x4x5x6x7x8  2            -      -              69463      69842    0.99     44.4  4.7e-16
   2x4x8x2x4x8x2  2            -      -              18529      15701    0.84     28.7  2.7e-16
   3x3x3x3x3x3x3  3            -      -               7246       5328    0.73     16.7  5.1e-16
   3x5x7x3x5x7x3  3            -      -              74563      71927    0.96     33.3  5.3e-16
   4x4x4x4x4x4x4  2^2          -      -              32592      29459    0.90     35.2  2.6e-16
   4x4x4x4x8x8x8  2^2          -      -             221900     232733    0.99     50.2  6.3e-16
   5x5x5x5x5x5x5  5            -      -             179364     168656    0.93     35.4  5.5e-16
   6x6x6x6x6x6x6  2.3          -      -             589250     521450    0.88     43.0  8.3e-16
   8x8x8x8x8x8x4  2^3          -      -            2298025    2462781    1.06     45.6  1.3e-15
   8x8x8x8x8x8x8  2^3          -      -            5692125    6328362    1.09     38.7  2.1e-15
 10x8x6x4x6x8x10  2.5          -      -            1979100    1906087    0.94     46.1  6.4e-16
  53x2x2x2x2x2x2  53           -      -              28807      29805    1.03      6.9  5.8e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               14      2     10   0.73   0.95   1.09    0.93
 ALL                             14      2     10   0.73   0.95   1.09    0.93
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      1      5   0.48   0.99   1.58    0.95
 prime column                     4      1      3   0.73   0.95   1.03    0.91
 even column                      2      0      2   0.88   0.91   0.94    0.91
 ALL                             14      2     10   0.73   0.95   1.09    0.93
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      4   0.88   0.96   1.09    0.98
 4097..65536                      4      0      4   0.84   0.93   0.99    0.92
 1025..4096                       3      1      1   0.73   1.03   1.58    1.06
 <= 256 points                    1      1      1   0.48   0.48   0.48    0.48
 ALL                             14      2     10   0.73   0.95   1.09    0.93
```


worst 10: 2x2x2x2x2x2x2 (- 0.48), 3x3x3x3x3x3x3 (- 0.73), 2x4x8x2x4x8x2 (- 0.84), 6x6x6x6x6x6x6 (- 0.88), 4x4x4x4x4x4x4 (- 0.90), 5x5x5x5x5x5x5 (- 0.93), 10x8x6x4x6x8x10 (- 0.94), 3x5x7x3x5x7x3 (- 0.96), 4x4x4x4x8x8x8 (- 0.99), 2x3x4x5x6x7x8 (- 0.99)
best 5: 2x2x2x2x2x2x53 (- 1.58), 8x8x8x8x8x8x8 (- 1.09), 8x8x8x8x8x8x4 (- 1.06), 53x2x2x2x2x2x2 (- 1.03), 2x3x4x5x6x7x8 (- 0.99)
