# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2r interleaved, natural, out of place, K=4 lane-major_c2r_k4_lm_mt8_est  cells: 7 listed, 7 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 18.705..20.317

threaded plans that ran serial (engaged = 0 at both flips): 1: 2x3x4x5x6x7x8


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x3x4x5x6x7x8  2            lm     -             247358    1062987    3.92     49.9  1.1e-15
   4x4x4x4x4x4x4  2^2          lm     -              34393     694498   19.70    133.4  5.6e-16
   4x4x4x4x8x8x8  2^2          lm     -             220513    2766200   12.39    202.1  8.3e-16
   6x6x6x6x6x6x6  2.3          lm     -             492875    5568256   11.13    205.5  2.4e-15
   8x8x8x8x8x8x4  2^3          lm     -            4340112   26712250    5.96     96.6  8.9e-16
   8x8x8x8x8x8x8  2^3          lm     -            8268350   33378500    3.67    106.5  1.0e-15
 10x8x6x4x6x8x10  2.5          lm     -            2951262   14656893    4.91    123.7  1.5e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                               7      0      0   3.67   5.96  19.70    7.34
 ALL                              7      0      0   3.67   5.96  19.70    7.34
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      5      0      0   3.67   5.96  19.70    7.31
 even column                      2      0      0   4.91   8.02  11.13    7.39
 ALL                              7      0      0   3.67   5.96  19.70    7.34
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   5      0      0   3.67   5.96  12.39    6.82
 4097..65536                      2      0      0   3.92  11.81  19.70    8.79
 ALL                              7      0      0   3.67   5.96  19.70    7.34
```


worst 10: 8x8x8x8x8x8x8 (lm 3.67), 2x3x4x5x6x7x8 (lm 3.92), 10x8x6x4x6x8x10 (lm 4.91), 8x8x8x8x8x8x4 (lm 5.96), 6x6x6x6x6x6x6 (lm 11.13), 4x4x4x4x8x8x8 (lm 12.39), 4x4x4x4x4x4x4 (lm 19.70)
best 5: 4x4x4x4x4x4x4 (lm 19.70), 4x4x4x4x8x8x8 (lm 12.39), 6x6x6x6x6x6x6 (lm 11.13), 8x8x8x8x8x8x4 (lm 5.96), 10x8x6x4x6x8x10 (lm 4.91)
