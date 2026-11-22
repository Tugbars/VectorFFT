# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2r interleaved, natural, out of place, K=4 lane-major_c2r_k4_lm_est  cells: 7 listed, 7 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 22.041..22.800


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x3x4x5x6x7x8  2            lm     -             261108    4149645   15.29     47.2  1.1e-15
   4x4x4x4x4x4x4  2^2          lm     -             119523    2646108   21.91     38.4  5.6e-16
   4x4x4x4x8x8x8  2^2          lm     -             862863   10452837   11.81     51.6  8.3e-16
   6x6x6x6x6x6x6  2.3          lm     -            2103437   21215337    9.99     48.2  2.4e-15
   8x8x8x8x8x8x4  2^3          lm     -           12570587   99599862    7.67     33.4  8.9e-16
   8x8x8x8x8x8x8  2^3          lm     -           23458100  130769137    5.56     37.5  1.0e-15
 10x8x6x4x6x8x10  2.5          lm     -            9161487   58390800    6.22     39.9  1.5e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                               7      0      0   5.56   9.99  21.91   10.07
 ALL                              7      0      0   5.56   9.99  21.91   10.07
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0   5.56  11.81  21.91   11.10
 even column                      2      0      0   6.22   8.10   9.99    7.88
 ALL                              7      0      0   5.56   9.99  21.91   10.07
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   5      0      0   5.56   7.67  11.81    7.93
 4097..65536                      2      0      0  15.29  18.60  21.91   18.30
 ALL                              7      0      0   5.56   9.99  21.91   10.07
```


worst 10: 8x8x8x8x8x8x8 (lm 5.56), 10x8x6x4x6x8x10 (lm 6.22), 8x8x8x8x8x8x4 (lm 7.67), 6x6x6x6x6x6x6 (lm 9.99), 4x4x4x4x8x8x8 (lm 11.81), 2x3x4x5x6x7x8 (lm 15.29), 4x4x4x4x4x4x4 (lm 21.91)
best 5: 4x4x4x4x4x4x4 (lm 21.91), 2x3x4x5x6x7x8 (lm 15.29), 4x4x4x4x8x8x8 (lm 11.81), 6x6x6x6x6x6x6 (lm 9.99), 8x8x8x8x8x8x4 (lm 7.67)
