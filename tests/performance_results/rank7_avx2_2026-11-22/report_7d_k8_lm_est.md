# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2c interleaved, natural, out of place, K=8 lane-major_k8_lm_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 15.363..16.348


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            lm     -               1759      75291   42.61     20.4  2.2e-16
  2x2x2x2x2x2x53  2            lm     -              89058     597796    6.07     17.9  1.4e-15
   2x3x4x5x6x7x8  2            lm     -             573737    6846669   11.88     43.0  4.3e-16
   2x4x8x2x4x8x2  2            lm     -             114120    1952133   17.08     37.3  4.1e-16
   3x3x3x3x3x3x3  3            lm     -              27879     570843   20.44     34.8  5.8e-16
   3x5x7x3x5x7x3  3            lm     -             488913    6978981   14.26     40.6  4.3e-16
   4x4x4x4x4x4x4  2^2          lm     -             206540    3332003   16.08     44.4  4.1e-16
   4x4x4x4x8x8x8  2^2          lm     -            2082825   26484000   12.65     42.8  1.4e-15
   5x5x5x5x5x5x5  5            lm     -            1124512   15818081   14.03     45.2  4.7e-16
   6x6x6x6x6x6x6  2.3          lm     -            6184250   87209775   14.01     32.8  8.0e-16
   8x8x8x8x8x8x4  2^3          lm     -           27705563  404101250   14.57     30.3  1.7e-15
   8x8x8x8x8x8x8  2^3          lm     -           56877912  829959131   14.57     31.0  2.4e-15
 10x8x6x4x6x8x10  2.5          lm     -           23776263  355551349   14.94     30.7  7.5e-16
  53x2x2x2x2x2x2  53           lm     -              92340    1982392   21.43     17.2  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              14      0      0  11.88  14.57  21.43   15.38
 ALL                             14      0      0  11.88  14.57  21.43   15.38
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   6.07  14.57  42.61   14.77
 prime column                     4      0      0  14.03  17.35  21.43   17.21
 even column                      2      0      0  14.01  14.47  14.94   14.46
 ALL                             14      0      0  11.88  14.57  21.43   15.38
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0  12.65  14.30  14.94   14.11
 4097..65536                      4      0      0  11.88  15.17  17.08   14.69
 1025..4096                       3      0      0   6.07  20.44  21.43   13.85
 <= 256 points                    1      0      0  42.61  42.61  42.61   42.61
 ALL                             14      0      0  11.88  14.57  21.43   15.38
```


worst 10: 2x2x2x2x2x2x53 (lm 6.07), 2x3x4x5x6x7x8 (lm 11.88), 4x4x4x4x8x8x8 (lm 12.65), 6x6x6x6x6x6x6 (lm 14.01), 5x5x5x5x5x5x5 (lm 14.03), 3x5x7x3x5x7x3 (lm 14.26), 8x8x8x8x8x8x4 (lm 14.57), 8x8x8x8x8x8x8 (lm 14.57), 10x8x6x4x6x8x10 (lm 14.94), 4x4x4x4x4x4x4 (lm 16.08)
best 5: 2x2x2x2x2x2x2 (lm 42.61), 53x2x2x2x2x2x2 (lm 21.43), 3x3x3x3x3x3x3 (lm 20.44), 2x4x8x2x4x8x2 (lm 17.08), 4x4x4x4x4x4x4 (lm 16.08)
