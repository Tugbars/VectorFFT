# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D r2c interleaved, natural, out of place, K=8 lane-major_r2c_k8_lm_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 14.468..15.054


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            lm     -             206000    2564231   12.04     95.4  3.5e-16
      4x4x4x4x4  2^2          lm     -               6782     255891   37.71     60.4  0.0e+00
    4x4x4x4x256  2^2          lm     -             466350    4689418   10.04     89.9  4.4e-16
      6x6x6x6x6  2.3          lm     -              53788     950508   17.61     74.7  5.8e-16
      8x8x8x8x8  2^3          lm     -             224212    3344744   14.87     87.7  3.4e-16
     8x8x53x8x8  2^3          lm     -            3757537   32695231    8.60     41.0  9.1e-16
 10x10x10x10x10  2.5          lm     -             781612    8958162   11.34     85.0  5.4e-16
 12x12x12x12x12  2^2.3        lm     -            2678375   23200181    8.59     66.6  7.8e-16
   16x16x16x8x8  2^4          lm     -            4927825   30739318    6.21     38.3  3.0e-16
 16x16x16x16x16  2^4          lm     -           23039600  152174456    6.58     36.4  4.0e-16
   20x16x12x8x4  2^2.5        lm     -            1918450   19514475    9.48     43.3  4.4e-16
  24x12x6x12x24  2^3.3        lm     -            6542425   66039000    9.84     57.6  7.6e-16
 32x16x16x16x16  2^5          lm     -           45500675  346167012    7.09     38.7  4.3e-16
     53x8x8x8x8  53           lm     -            3910837   37215393    8.90     39.4  8.3e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              14      0      0   6.58   9.66  17.61   10.65
 ALL                             14      0      0   6.58   9.66  17.61   10.65
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   6.21   9.32  37.71   10.68
 even column                      5      0      0   8.59   9.84  17.61   10.98
 prime column                     1      0      0   8.90   8.90   8.90    8.90
 ALL                             14      0      0   6.58   9.66  17.61   10.65
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   9      0      0   6.21   8.60  11.34    8.37
 4097..65536                      4      0      0  10.04  13.46  17.61   13.34
 257..1024                        1      0      0  37.71  37.71  37.71   37.71
 ALL                             14      0      0   6.58   9.66  17.61   10.65
```


worst 10: 16x16x16x8x8 (lm 6.21), 16x16x16x16x16 (lm 6.58), 32x16x16x16x16 (lm 7.09), 12x12x12x12x12 (lm 8.59), 8x8x53x8x8 (lm 8.60), 53x8x8x8x8 (lm 8.90), 20x16x12x8x4 (lm 9.48), 24x12x6x12x24 (lm 9.84), 4x4x4x4x256 (lm 10.04), 10x10x10x10x10 (lm 11.34)
best 5: 4x4x4x4x4 (lm 37.71), 6x6x6x6x6 (lm 17.61), 8x8x8x8x8 (lm 14.87), 2x4x8x16x32 (lm 12.04), 10x10x10x10x10 (lm 11.34)
