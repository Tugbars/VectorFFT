# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2c interleaved, natural, out of place, K=8 lane-major_k8_lm_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 13.091..16.588

threaded plans that ran serial (engaged = 0 at both flips): 4: 2x2x2x2x2x2x2, 2x2x2x2x2x2x53, 2x3x4x5x6x7x8, 2x4x8x2x4x8x2


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            lm     -               1762      40433   22.81     20.3  2.2e-16
  2x2x2x2x2x2x53  2            lm     -              82655     498754    5.43     19.3  1.4e-15
   2x3x4x5x6x7x8  2            lm     -             558637    3383362    5.30     44.2  4.3e-16
   2x4x8x2x4x8x2  2            lm     -             107887    1391418   10.49     39.5  4.1e-16
   3x3x3x3x3x3x3  3            lm     -              15413     300364   16.30     63.0  5.8e-16
   3x5x7x3x5x7x3  3            lm     -             161762    2227600   13.70    122.8  4.3e-16
   4x4x4x4x4x4x4  2^2          lm     -              62967     944293   12.87    145.7  4.1e-16
   4x4x4x4x8x8x8  2^2          lm     -             486538    6359106   11.39    183.2  1.4e-15
   5x5x5x5x5x5x5  5            lm     -             249025    3043068   10.14    204.0  4.7e-16
   6x6x6x6x6x6x6  2.3          lm     -            1346488   14868444   10.49    150.5  8.0e-16
   8x8x8x8x8x8x4  2^3          lm     -           10193200  113516856   10.62     82.3  1.7e-15
   8x8x8x8x8x8x8  2^3          lm     -           24508850  255564494    9.20     71.9  2.4e-15
 10x8x6x4x6x8x10  2.5          lm     -            9060225  111185306   11.94     80.6  7.5e-16
  53x2x2x2x2x2x2  53           lm     -              17389     473355   22.83     91.5  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              14      0      0   5.43  11.00  22.81   11.41
 ALL                             14      0      0   5.43  11.00  22.81   11.41
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   5.30  10.56  22.81    9.98
 prime column                     4      0      0  10.14  15.00  22.83   15.08
 even column                      2      0      0  10.49  11.21  11.94   11.19
 ALL                             14      0      0   5.43  11.00  22.81   11.41
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   9.20  10.55  11.94   10.59
 4097..65536                      4      0      0   5.30  11.68  13.70    9.95
 1025..4096                       3      0      0   5.43  16.30  22.83   12.64
 <= 256 points                    1      0      0  22.81  22.81  22.81   22.81
 ALL                             14      0      0   5.43  11.00  22.81   11.41
```


worst 10: 2x3x4x5x6x7x8 (lm 5.30), 2x2x2x2x2x2x53 (lm 5.43), 8x8x8x8x8x8x8 (lm 9.20), 5x5x5x5x5x5x5 (lm 10.14), 6x6x6x6x6x6x6 (lm 10.49), 2x4x8x2x4x8x2 (lm 10.49), 8x8x8x8x8x8x4 (lm 10.62), 4x4x4x4x8x8x8 (lm 11.39), 10x8x6x4x6x8x10 (lm 11.94), 4x4x4x4x4x4x4 (lm 12.87)
best 5: 53x2x2x2x2x2x2 (lm 22.83), 2x2x2x2x2x2x2 (lm 22.81), 3x3x3x3x3x3x3 (lm 16.30), 3x5x7x3x5x7x3 (lm 13.70), 4x4x4x4x4x4x4 (lm 12.87)
