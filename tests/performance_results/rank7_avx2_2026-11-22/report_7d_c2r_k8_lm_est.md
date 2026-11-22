# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2r interleaved, natural, out of place, K=8 lane-major_c2r_k8_lm_est  cells: 7 listed, 7 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 21.217..22.554


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x3x4x5x6x7x8  2            lm     -             550488    8511594   14.30     44.8  1.2e-15
   4x4x4x4x4x4x4  2^2          lm     -             235567    5318033   22.42     38.9  5.6e-16
   4x4x4x4x8x8x8  2^2          lm     -            1922575   21874337   11.30     46.4  8.9e-16
   6x6x6x6x6x6x6  2.3          lm     -            5750637   48467025    8.17     35.2  2.0e-15
   8x8x8x8x8x8x4  2^3          lm     -           27379900  222828463    8.11     30.6  1.0e-15
   8x8x8x8x8x8x8  2^3          lm     -           51395388  324260150    6.30     34.3  1.0e-15
 10x8x6x4x6x8x10  2.5          lm     -           19364163  140299850    7.18     37.7  1.6e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                               7      0      0   6.30   8.17  22.42   10.12
 ALL                              7      0      0   6.30   8.17  22.42   10.12
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0   6.30  11.30  22.42   11.31
 even column                      2      0      0   7.18   7.68   8.17    7.66
 ALL                              7      0      0   6.30   8.17  22.42   10.12
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   5      0      0   6.30   8.11  11.30    8.05
 4097..65536                      2      0      0  14.30  18.36  22.42   17.90
 ALL                              7      0      0   6.30   8.17  22.42   10.12
```


worst 10: 8x8x8x8x8x8x8 (lm 6.30), 10x8x6x4x6x8x10 (lm 7.18), 8x8x8x8x8x8x4 (lm 8.11), 6x6x6x6x6x6x6 (lm 8.17), 4x4x4x4x8x8x8 (lm 11.30), 2x3x4x5x6x7x8 (lm 14.30), 4x4x4x4x4x4x4 (lm 22.42)
best 5: 4x4x4x4x4x4x4 (lm 22.42), 2x3x4x5x6x7x8 (lm 14.30), 4x4x4x4x8x8x8 (lm 11.30), 6x6x6x6x6x6x6 (lm 8.17), 8x8x8x8x8x8x4 (lm 8.11)
